"""
RAG retrieval and LLM response logic using AWS Bedrock and Chroma.

Provides multi-query retrieval and a stateful chat chain with session history.
Per-session vector stores are persisted under persist_directory_db/<session_id>.
"""

import os
import shutil
import warnings

import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()


def format_docs(docs):
    """Concatenate document page contents with a fixed separator."""
    return "/n/n".join(doc.page_content for doc in docs)


def split_content(content: str):
    """Split a string by newline."""
    return content.split("\n")


def _get_embeddings():
    """Shared Bedrock embeddings client."""
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    return BedrockEmbeddings(
        client=client,
        region_name=region,
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
    )


def _session_persist_dir(session_id):
    """Directory where this session's Chroma data is stored."""
    base = os.getenv("persist_directory_db", "Data/db")
    return os.path.join(base, str(session_id))


def save_pdf_to_vectorstore(pdf_path: str, session_id: str):
    """
    Load PDF from pdf_path, chunk, embed with Bedrock, and persist to Chroma
    in a directory unique to this session (per user).
    """
    if not session_id:
        raise ValueError("session_id is required to store file per user")
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    persist_dir = _session_persist_dir(session_id)
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def get_vectorstore_for_session(session_id: str):
    """
    Load this session's Chroma vector store from disk if it exists.
    Returns None if no persisted store for this session.
    """
    if not session_id:
        return None
    persist_dir = _session_persist_dir(session_id)
    if not os.path.isdir(persist_dir):
        return None
    embeddings = _get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def delete_session_store(session_id: str):
    """Remove this session's persisted vector store directory (e.g. after 10 min inactivity)."""
    if not session_id:
        return
    persist_dir = _session_persist_dir(session_id)
    if os.path.isdir(persist_dir):
        try:
            shutil.rmtree(persist_dir)
            print(f"Vector store for session {session_id} deleted")
        except OSError:
            pass
    else:
        print(f"Vector store for session {session_id} not found")

def is_about_jairam(history,question,llm,DB_store,session_id):
    """Check if the history is about Jairam."""
    response = llm.invoke(f"""Given this conversation history:
                {history}

                And this new question: "{question}".
                
                Is the user asking about a person named Jairam in the new question then yes else if they are asking about a document user uploaded (user uploaded a document: {session_id in DB_store }) then the answer is no .
    Reply with one word: yes or no. don't say anything else.""")


    
    if response.content == "yes":
        return True
    else:
        return False
def Multi_query(question, DB_store=None, session_id=None,get_session_history=None,llm=None):
    """
    Generate multiple question variations via LLM, retrieve docs for each,
    deduplicate by content, and return formatted context from the session's vector store.
    """
    if is_about_jairam(get_session_history(session_id),question,llm,DB_store,session_id):
        context_file_path = os.getenv("JAIRAM_ALLURI_CONTEXT_FILE")
        with open(context_file_path, "r") as file:
            context = file.read()

        return context
   
    if DB_store is None or session_id not in DB_store:
        return " No context provided "
    vectordb, _ = DB_store[session_id]
    if vectordb is None:
        return " No context provided "

    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Please Generate 4 Variations of the question"),
        ("user", "{question}"),
    ])
    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_CHAT_MODEL_ID", "amazon.nova-micro-v1:0"),
        client=client,
        region=region,
        model_kwargs={"temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))},
    )
    chain = prompt | llm
    content = chain.invoke({"question": question}).content
    questions = split_content(content)
    docs = {}
    for ques in tqdm(questions):
        similar_data = vectordb.similarity_search(question)
        for dat in similar_data:
            docs[dat.page_content] = dat
    docs2 = list(docs.values())
    context_fromdb = format_docs(docs2)
    DB_store[session_id][1] = datetime.now()
    return context_fromdb


def LLM_response_text(question: str, session_id: str, get_session_history, DB_store=None):
    """
    Answer the question using RAG context (from session's vector store) and conversation history.
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_CHAT_MODEL_ID", "amazon.nova-micro-v1:0"),
        client=client,
        region=region,
        model_kwargs={"temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))},
    )
    #### use information from get_session_history to answer the question. if it is about Jairam then use Jairam's persistent DB
    #### else use the context from the session's vector store.

    
    context_fromdb = Multi_query(question, DB_store=DB_store, session_id=session_id,get_session_history=get_session_history,llm=llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are here to answer questions about why you should Hire Jairam Alluri in a positive manner if asked. "
         "for example: 1. what is Jairam's expertise in. 2. what is Jairam's visa status. etc"
        "or ignore all that if not askedf about Jairam and answer any other questions user asks about the context provided to the user using or if user asks anything else then still use the same context {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}"),
    ])
    # print(get_session_history(session_id),"hiii")


    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history.invoke(
        {"context": context_fromdb, "question": question},
        config={"configurable": {"session_id": session_id}},
    ).content
