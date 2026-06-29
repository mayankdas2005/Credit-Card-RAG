from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List
import os
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")

def upsert_to_vector_db(processed_documents: List[Document]):
    # Initialize your embedding model vector weights
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=KEY)
    
    # Load and save collection artifacts to disk locally
    db = Chroma.from_documents(
        documents=processed_documents,
        embedding=embeddings,
        persist_directory= os.path.join(ROOT_DIR, "credit_card_rag_db")
    )
    print(f"Successfully indexed {len(processed_documents)} rich metadata chunks.")
    return db

