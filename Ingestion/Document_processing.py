import sys
from dotenv import load_dotenv
import os
import bs4 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
import boto3
from langchain_aws import ChatBedrock,BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings 

warnings.filterwarnings("ignore")

load_dotenv()


def main(data_pth=os.getenv("data_path_variable")):
        loader = PyPDFDirectoryLoader(path=data_pth)
        data = loader.load()
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "0"))
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted_docs = splitter.split_documents(data)

        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        embeddings_model = BedrockEmbeddings(
            client=client,
            region_name=region,
            model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
        )
        
        vectorstore = Chroma.from_documents(documents=splitted_docs,embedding=embeddings_model,persist_directory=os.getenv("persist_directory_db"))

        retriever = vectorstore.as_retriever()

        # print(retriever.get_relevant_documents("what is the address?"))

    


if __name__ =="__main__":
    sys.exit(main())