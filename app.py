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

loader = UnstructuredPDFLoader("data/Civil-code.pdf")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(documents)

print("The length of chunked documents is:", len(chunked_documents))

genai.configure(api_key = genai_api_key)

# Initialize Embeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create and persist a Chroma vector database from the chunked documents
vectorstore = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embed_model,
    persist_directory="chroma_db",  # Local mode with in-memory storage only
    collection_name="rag"
)

chat_model = ChatGroq(temperature=0,
                      model_name="mixtral-8x7b-32768",
                      api_key=groq_api_key)

retriever=vectorstore.as_retriever(search_kwargs={'k': 3})


