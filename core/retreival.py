"""
RAG retrieval and LLM response logic using AWS Bedrock and Chroma.

Provides multi-query retrieval and a stateful chat chain with session history.
"""

import os
import warnings

import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()


def format_docs(docs):
    """
    Concatenate document page contents with a fixed separator.

    Args:
        docs: Iterable of document-like objects with .page_content.

    Returns:
        Single string of all page contents joined by "/n/n".
    """
    return "/n/n".join(doc.page_content for doc in docs)


def split_content(content: str):
    """
    Split a string by the literal "/n" delimiter.

    Args:
        content: String that may contain "/n" as separator.

    Returns:
        List of substrings.
    """
    return content.split("\n")


def _get_embeddings_model():
    """Build Bedrock embeddings model from env config."""
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    return BedrockEmbeddings(
        client=client,
        region_name=region,
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
    )


def build_in_memory_vectorstore(documents):
    """
    Create an in-memory Chroma vectorstore from a list of documents.

    Uses Bedrock embeddings from env. No persistence; store is lost when process exits.

    Args:
        documents: List of LangChain Document objects (e.g. from a PDF loader + splitter).

    Returns:
        Chroma vectorstore instance (in-memory).
    """
    embeddings_model = _get_embeddings_model()
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
    )


def Multi_query(question, vectordb=None):
    """
    Generate multiple question variations via LLM, retrieve docs for each,
    deduplicate by content, and return formatted context.

    Uses Bedrock for embeddings and chat. Reads config from env (AWS_REGION,
    persist_directory_db, model IDs, temperature).

    Args:
        question: User question string.
        vectordb: Optional Chroma vectorstore to search. If None, uses the persistent DB.

    Returns:
        Formatted context string from retrieved documents.
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    embeddings_model = _get_embeddings_model()
    if vectordb is None:
        vectordb = Chroma(
            persist_directory=os.getenv("persist_directory_db"),
            embedding_function=embeddings_model,
        )
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
    docs2 = [docs[key] for key in docs.keys()]
    context_fromdb = format_docs(docs2)
    return context_fromdb


def LLM_response_text(question: str, session_id: str, get_session_history, session_vectordb=None):
    """
    Answer the question using RAG context and conversation history.

    Retrieves context via Multi_query (from session_vectordb if provided, else
    persistent DB), then runs a Bedrock chat chain with history.

    Args:
        question: User question string.
        session_id: Session identifier for conversation history.
        get_session_history: Callable(session_id) -> ChatMessageHistory.
        session_vectordb: Optional in-memory Chroma for this session (e.g. uploaded PDF). Used for 10 min then discarded.

    Returns:
        The LLM response text (answer string).
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_CHAT_MODEL_ID", "amazon.nova-micro-v1:0"),
        client=client,
        region=region,
        model_kwargs={"temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))},
    )
    context_fromdb = Multi_query(question, vectordb=session_vectordb)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "you are an expert in understanding lease and answer all the questions about it to the user using {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}"),
    ])
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
