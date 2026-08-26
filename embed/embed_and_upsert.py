from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")

# The free-tier embed_content quota is per-minute and far lower than a single document's worth of
# chunks can require in one call. Batch + retry-with-backoff instead of sending everything at once.
EMBED_BATCH_SIZE = 40


@retry(stop=stop_after_attempt(8), wait=wait_exponential_jitter(initial=2, max=60))
def _add_batch(db: Chroma, texts: List[str], metadatas: List[dict]):
    db.add_texts(texts=texts, metadatas=metadatas)


def upsert_to_vector_db(processed_documents: List[Document]):
    # Initialize your embedding model vector weights
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=KEY)

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=os.path.join(ROOT_DIR, "credit_card_rag_db")
    )

    for i in range(0, len(processed_documents), EMBED_BATCH_SIZE):
        batch = processed_documents[i:i + EMBED_BATCH_SIZE]
        _add_batch(db, [d.page_content for d in batch], [d.metadata for d in batch])
        print(f"  Embedded batch {i // EMBED_BATCH_SIZE + 1}/{-(-len(processed_documents) // EMBED_BATCH_SIZE)}")

    print(f"Successfully indexed {len(processed_documents)} rich metadata chunks.")
    return db

