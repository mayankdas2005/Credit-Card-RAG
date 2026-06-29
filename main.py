import os 
from processing.parse_pdf import extract_markdown_from_pdf
from processing.create_metadata import create_global_catalog_discovery, build_rich_vector_documents
from embed.embed_and_upsert import upsert_to_vector_db



def run_ingestion_pipeline():
    ROOT_DIR = os.path.dirname(__file__)
    ROOT_DATA_DIR = os.path.join(ROOT_DIR, "data")
    PDF_DIR = os.path.join(ROOT_DATA_DIR, "pdfs")
    MD_DIR = os.path.join(ROOT_DATA_DIR, "parsed_md")

    os.makedirs(MD_DIR, exist_ok=True)

    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdfs:
        return f"No documents found in {PDF_DIR}"
    
    all_staged_documents = []
    for filename in pdfs:
        filepath = os.path.join(PDF_DIR, filename)
        
        print(f"Processing File Target: {filename}")
        
        extracted_md = extract_markdown_from_pdf(filepath, MD_DIR)

        print("Starting Pass 1: Auto-Detecting Bank Details & Card Frameworks...")
        catalog_data = create_global_catalog_discovery(extracted_md)

        if not catalog_data or not catalog_data.cards_found:
            print(f"Warning: Discovered no reliable parameters from internal structural parsing. Skipping {filename}.")
            continue
            
        bank_name = catalog_data.bank_name
        bank_id = catalog_data.bank_id
        cards_lookup = {card.card_id: card for card in catalog_data.cards_found}


        print("Starting Pass 2: Generating rich classification metadata schemas...")
        card_docs = build_rich_vector_documents(extracted_md, bank_name, bank_id, cards_lookup)
        print(f"Generated {len(card_docs)} metadata-stamped documents.")

        all_staged_documents.extend(card_docs)
        print(f"Staged file components for execution: {filename}")

        if all_staged_documents:
            print(f"\nFinal Assembly: Syncing {len(all_staged_documents)} total chunks to local Chroma DB index...")
            vector_store = upsert_to_vector_db(processed_documents=all_staged_documents)
            print("Ingestion engine processing cycle successfully complete!")
        else:
            print("No valid elements derived during this pipeline execution sequence.")

if __name__=="__main__":
    run_ingestion_pipeline()





