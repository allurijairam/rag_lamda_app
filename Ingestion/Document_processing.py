"""
Document ingestion: load PDFs from a directory, split, embed with Bedrock, and persist to Chroma.

Config driven by env: data_path_variable, persist_directory_db, AWS_REGION,
BEDROCK_EMBEDDING_MODEL_ID, CHUNK_SIZE, CHUNK_OVERLAP.
"""

import os
import sys
import warnings

import boto3
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()


def main(data_pth=os.getenv("data_path_variable")):
    """
    Load PDFs from data_pth, split into chunks, embed with Bedrock, and persist to Chroma.

    Args:
        data_pth: Directory path containing PDFs. Defaults to env var data_path_variable.
    """
    loader = PyPDFDirectoryLoader(path=data_pth)
    data = loader.load()
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "0"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splitted_docs = splitter.split_documents(data)
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    embeddings_model = BedrockEmbeddings(
        client=client,
        region_name=region,
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
    )
    vectorstore = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embeddings_model,
        persist_directory=os.getenv("persist_directory_db"),
    )
    vectorstore.as_retriever()


if __name__ == "__main__":
    sys.exit(main())
