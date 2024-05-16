from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv()

# Accessing the environment variables
genai_api_key = os.getenv("GENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def initialize_embeddings(model_name="BAAI/bge-base-en-v1.5"):
    embed_model = FastEmbedEmbeddings(model_name=model_name)
    return embed_model

def create_vectorstore(documents, embed_model, persist_directory="chroma_db", collection_name="rag"):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vectorstore

def initialize_chat_model(temperature=0, model_name="mixtral-8x7b-32768", api_key=None):
    chat_model = ChatGroq(temperature=temperature, model_name=model_name, api_key=api_key)
    return chat_model

# Load and process documents
documents = load_documents("data/Civil-code.pdf")
chunked_documents = split_documents(documents)
print("The length of chunked documents is:", len(chunked_documents))

# Configure Generative AI
genai.configure(api_key=genai_api_key)

# Initialize Embeddings
embed_model = initialize_embeddings()

# Create Vector Store
vectorstore = create_vectorstore(chunked_documents, embed_model)

# Initialize Chat Model
chat_model = initialize_chat_model(api_key=groq_api_key)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


