from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List
import os


KEY = os.environ["GEMINI_API_KEY"]


def upsert_to_vector_db(processed_documents: List[Document]):
    # Initialize your embedding model vector weights
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", api_key=KEY)
    
    # Load and save collection artifacts to disk locally
    db = Chroma.from_documents(
        documents=processed_documents,
        embedding=embeddings,
        persist_directory="./credit_card_rag_db"
    )
    print(f"Successfully indexed {len(processed_documents)} rich metadata chunks.")
    return db

