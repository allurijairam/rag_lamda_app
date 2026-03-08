import sys
from dotenv import load_dotenv
import os
import bs4 
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
import boto3
from langchain_aws import ChatBedrock,BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()
store = {}
def format_docs(docs):
    return "/n/n".join(tex.page_content for tex in docs)

# def get_session_history(session_id:str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

def split_content(content:str):

    return content.split("/n")

def Multi_query(question):

    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    embeddings_model = BedrockEmbeddings(
        client=client,
        region_name=region,
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
    )
    print(1)
    vectordb = Chroma(
    persist_directory=os.getenv("persist_directory_db"), 
    embedding_function=embeddings_model,
    )
    print(2)
    prompt = ChatPromptTemplate.from_messages([("system","Please Generate 4 Variations of the question"),
                                 ("user","{question}")])
    print(3)

    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_CHAT_MODEL_ID", "amazon.nova-micro-v1:0"),
        client=client,
        region=region,
        model_kwargs={"temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))},
    )

    chain = prompt | llm

    content = chain.invoke({'question':question}).content

    questions = split_content(content)
    print(4)
    docs = {}
    for ques in tqdm(questions):
        print(ques)

        similar_data = vectordb.similarity_search(question)
        for dat in similar_data:
            docs[dat.page_content] = dat
    print(5)
    docs2 = [docs[key] for key in docs.keys()]
    print(6)
    print(len(docs2))
    context_fromdb = format_docs(docs2)
    return context_fromdb








def LLM_response_text(question: str, session_id: str, get_session_history):
    """
    based on the question returns answers
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)

    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_CHAT_MODEL_ID", "amazon.nova-micro-v1:0"),
        client=client,
        region=region,
        model_kwargs={"temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))},
    )

    embeddings_model = BedrockEmbeddings(
        client=client,
        region_name=region,
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
    )

    
    vectordb = Chroma(
    persist_directory=os.getenv("persist_directory_db"), 
    embedding_function=embeddings_model,
    )

    docs = vectordb.similarity_search(question)
    
    # context_fromdb = format_docs(docs)
    # print(context_fromdb)
    context_fromdb = Multi_query(question)
    prompt = ChatPromptTemplate.from_messages([("system","you are an expert in understanding lease and answer all the questions about it to the user using {context}"),
                                 MessagesPlaceholder(variable_name="history"),
                                 ("user","{question}")])
    
    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )


    return chain_with_history.invoke({"context":context_fromdb,"question":question},
                                    config={"configurable": {"session_id": session_id}}).content

# LLM_response_text("what are all the charges on the lease?")
# LLM_response_text("where is the building?")
# print(LLM_response_text("what was my first question??","2312"))


