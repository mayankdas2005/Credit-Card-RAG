import os 
from processing.parse_pdf import extract_markdown_from_pdf
from processing.create_metadata import create_global_catalog_discovery, build_rich_vector_documents
from embed.embed_and_upsert import upsert_to_vector_db
from processing.ingestion_cache import (
    load_cache, 
    save_cache, 
    is_file_indexed, 
    get_cached_catalog, 
    add_catalog_to_cache, 
    mark_file_indexed
)

def pre_parse_all_pdfs(pdf_dir: str, md_dir: str):
    """Check all PDFs in the pdf_dir. Parse to md if the md file does not exist."""
    print("Checking for PDF files to pre-parse...")
    if not os.path.exists(pdf_dir):
        print(f"Warning: PDF directory {pdf_dir} does not exist.")
        return
        
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    for filename in pdfs:
        pdf_path = os.path.join(pdf_dir, filename)
        md_filename = filename.replace(".pdf", ".md")
        md_path = os.path.join(md_dir, md_filename)
        
        if os.path.exists(md_path):
            print(f"-> Markdown file for '{filename}' already exists. Skipping parsing.")
        else:
            print(f"-> Parsing '{filename}' to markdown on the fly...")
            extract_markdown_from_pdf(pdf_path, md_dir)

def run_ingestion_pipeline():
    ROOT_DIR = os.path.dirname(__file__)
    ROOT_DATA_DIR = os.path.join(ROOT_DIR, "data")
    PDF_DIR = os.path.join(ROOT_DATA_DIR, "pdfs")
    MD_DIR = os.path.join(ROOT_DATA_DIR, "parsed_md")
    CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")
    CACHE_EXAMPLE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.example.json")

    # Seed main cache from example template if missing
    if not os.path.exists(CACHE_PATH) and os.path.exists(CACHE_EXAMPLE_PATH):
        try:
            import shutil
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            shutil.copy(CACHE_EXAMPLE_PATH, CACHE_PATH)
            print("Seeded ingestion cache from example template.")
        except Exception as e:
            print(f"Warning: Failed to seed cache from template: {e}")

    os.makedirs(MD_DIR, exist_ok=True)
    
    # Step 1: Pre-parse any unparsed PDFs
    pre_parse_all_pdfs(PDF_DIR, MD_DIR)

    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdfs:
        return f"No documents found in {PDF_DIR}"
    
    # Load metadata and indexing cache
    cache_data = load_cache(CACHE_PATH)
    
    for filename in pdfs:
        print(f"\n========================================================")
        print(f"Processing File Target: {filename}")
        print(f"========================================================")
        
        # Step A: Check if the file is already fully indexed
        if is_file_indexed(filename, cache_data):
            print(f"--> File '{filename}' has already been fully indexed. Skipping entire ingestion.")
            continue
            
        md_filename = filename.replace(".pdf", ".md")
        md_filepath = os.path.join(MD_DIR, md_filename)
        
        if not os.path.exists(md_filepath):
            print(f"Error: Markdown file {md_filepath} not found even after parsing phase. Skipping.")
            continue
            
        with open(md_filepath, "r", encoding="utf-8") as f:
            extracted_md = f.read()

        # Step B: Auto-Detect Bank Details & Cards (Pass 1)
        catalog_data = get_cached_catalog(filename, cache_data)
        if catalog_data:
            print("Starting Pass 1: Loading bank details and card catalog from cache...")
        else:
            print("Starting Pass 1: Auto-Detecting Bank Details & Card Frameworks via LLM...")
            catalog_data = create_global_catalog_discovery(extracted_md)
            if catalog_data and catalog_data.cards_found:
                add_catalog_to_cache(filename, catalog_data, cache_data)
                save_cache(cache_data, CACHE_PATH)
                print("Pass 1: Discovery successful. Saved catalog to cache.")

        if not catalog_data or not catalog_data.cards_found:
            print(f"Warning: Discovered no reliable parameters from internal structural parsing. Skipping {filename}.")
            continue
            
        bank_name = catalog_data.bank_name
        bank_id = catalog_data.bank_id
        cards_lookup = {card.card_id: card for card in catalog_data.cards_found}

        # Step C: Generate metadata-stamped chunks (Pass 2)
        print("Starting Pass 2: Generating rich classification metadata schemas...")
        card_docs = build_rich_vector_documents(extracted_md, bank_name, bank_id, cards_lookup)
        print(f"Generated {len(card_docs)} metadata-stamped documents.")

        # Step D: Sync chunks to local Chroma DB index
        if card_docs:
            print(f"Final Assembly: Syncing {len(card_docs)} total chunks to local Chroma DB index...")
            vector_store = upsert_to_vector_db(processed_documents=card_docs)
            
            # Mark file as fully indexed and save cache
            mark_file_indexed(filename, cache_data)
            save_cache(cache_data, CACHE_PATH)
            print(f"Successfully processed and indexed: {filename}")
        else:
            print(f"No valid elements derived during this pipeline execution sequence for {filename}.")

if __name__ == "__main__":
    run_ingestion_pipeline()


