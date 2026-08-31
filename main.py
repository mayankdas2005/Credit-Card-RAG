import os
import json
from processing.parse_pdf import extract_markdown_from_pdf
from processing.create_metadata import create_global_catalog_discovery, build_rich_vector_documents, create_reward_rate_catalog, reconcile_duplicate_cards
from processing.models import RewardRateProfile
from embed.embed_and_upsert import upsert_to_vector_db
from processing.ingestion_cache import (
    load_cache,
    save_cache,
    is_file_indexed,
    get_cached_catalog,
    add_catalog_to_cache,
    mark_file_indexed,
    is_reward_file_indexed,
    mark_reward_file_indexed,
    add_reward_profiles_to_cache,
    record_merge_groups,
    is_bank_reconciled,
    mark_bank_reconciled
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

def run_card_reconciliation():
    """Detect and merge duplicate card identities that arose because Pass 1 catalog discovery runs
    independently per source document with no shared registry — the same real card can end up with
    two disconnected card_id/card_name entries if two documents for the same bank describe it
    slightly differently (e.g. one abbreviates the name). Only runs for banks backed by 2+ source
    documents, and only reruns when that document set changes (idempotent via a fingerprint check),
    so this is a cheap no-op on the common single-document-per-bank case."""
    ROOT_DIR = os.path.dirname(__file__)
    CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")
    cache_data = load_cache(CACHE_PATH)

    catalogs = cache_data.get("discovered_catalogs", {})

    # Group every discovered card, across every source file, by bank_id
    cards_by_bank = {}
    bank_names = {}
    for filename, catalog in catalogs.items():
        bank_id = catalog.get("bank_id")
        bank_names[bank_id] = catalog.get("bank_name")
        for card in catalog.get("cards_found", []):
            cards_by_bank.setdefault(bank_id, []).append({**card, "source_file": filename})

    for bank_id, cards in cards_by_bank.items():
        source_files = {c["source_file"] for c in cards}
        if len(source_files) < 2:
            continue  # nothing to reconcile — only one document has ever contributed cards for this bank

        if is_bank_reconciled(bank_id, cache_data):
            continue

        print(f"\n--- Card Reconciliation Target: {bank_names.get(bank_id, bank_id)} "
              f"({len(cards)} cards across {len(source_files)} documents) ---")
        result = reconcile_duplicate_cards(bank_names.get(bank_id, bank_id), cards)

        if result and result.merge_groups:
            record_merge_groups(result.merge_groups, cache_data)
            for group in result.merge_groups:
                print(f"  Merged {group.alias_card_ids} -> canonical '{group.canonical_card_id}' ({group.reasoning})")
        else:
            print("  No duplicate cards found.")

        mark_bank_reconciled(bank_id, cache_data)
        save_cache(cache_data, CACHE_PATH)


def run_reward_ingestion_pipeline():
    """Generic, bank-agnostic ingestion of rewards-program PDFs. Any bank's reward-program PDF dropped
    into data/pdfs/rewards/ is discovered and extracted the same way — no bank names hardcoded."""
    ROOT_DIR = os.path.dirname(__file__)
    REWARDS_PDF_DIR = os.path.join(ROOT_DIR, "data", "pdfs", "rewards")
    REWARDS_MD_DIR = os.path.join(ROOT_DIR, "data", "parsed_md", "rewards")
    CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")

    if not os.path.exists(REWARDS_PDF_DIR):
        print(f"No rewards-program PDF directory found at {REWARDS_PDF_DIR}. Skipping reward ingestion.")
        return

    os.makedirs(REWARDS_MD_DIR, exist_ok=True)
    pre_parse_all_pdfs(REWARDS_PDF_DIR, REWARDS_MD_DIR)

    pdfs = [f for f in os.listdir(REWARDS_PDF_DIR) if f.endswith(".pdf")]
    if not pdfs:
        print("No rewards-program PDFs found. Skipping reward ingestion.")
        return

    cache_data = load_cache(CACHE_PATH)

    for filename in pdfs:
        print(f"\n--- Reward Ingestion Target: {filename} ---")

        if is_reward_file_indexed(filename, cache_data):
            print(f"--> Rewards file '{filename}' already indexed. Skipping.")
            continue

        md_filename = filename.replace(".pdf", ".md")
        md_filepath = os.path.join(REWARDS_MD_DIR, md_filename)
        if not os.path.exists(md_filepath):
            print(f"Error: Markdown file {md_filepath} not found even after parsing phase. Skipping.")
            continue

        with open(md_filepath, "r", encoding="utf-8") as f:
            extracted_md = f.read()

        # Discover which bank/cards this reward document covers using the same generic Pass-1 pattern
        # already used for MITC docs — no card vocabulary is hardcoded anywhere in this path.
        catalog_data = create_global_catalog_discovery(extracted_md)
        if not catalog_data or not catalog_data.cards_found:
            print(f"Warning: could not discover any cards in {filename}. Skipping.")
            continue

        # If this bank already has a card catalog from a previously-ingested MITC document, reuse its
        # card_id list instead of the ids freshly discovered from this reward document's own text.
        # LLM-generated slugs aren't guaranteed to converge across two different documents describing
        # the same card (e.g. "magnus-credit-card" vs "axis-bank-magnus-credit-card") — reusing the
        # already-established ids keeps reward_profiles joinable with the MITC-sourced card/fee data.
        existing_catalog = next(
            (c for c in cache_data.get("discovered_catalogs", {}).values()
             if c.get("bank_id") == catalog_data.bank_id),
            None
        )
        if existing_catalog:
            valid_card_ids = [c["card_id"] for c in existing_catalog.get("cards_found", [])]
            print(f"Reusing {len(valid_card_ids)} previously-discovered card IDs for bank_id='{catalog_data.bank_id}'.")
        else:
            valid_card_ids = [c.card_id for c in catalog_data.cards_found]

        reward_catalog = create_reward_rate_catalog(extracted_md, valid_card_ids)

        if reward_catalog and reward_catalog.profiles:
            add_reward_profiles_to_cache(reward_catalog.profiles, cache_data)
            print(f"Successfully extracted {len(reward_catalog.profiles)} reward profiles from {filename}")
        else:
            print(f"Warning: no reward-rate figures were extractable from {filename}.")

        mark_reward_file_indexed(filename, cache_data)
        save_cache(cache_data, CACHE_PATH)


def load_manual_reward_seeds():
    """Merge manually curated reward-rate seed files (data/reward_seeds/*.json) into the cache, using
    the exact same schema a PDF extraction would produce. Lets a bank be covered before its reward PDF
    is sourced. A genuine PDF-extracted profile always takes precedence over a manual seed for the same card."""
    ROOT_DIR = os.path.dirname(__file__)
    SEED_DIR = os.path.join(ROOT_DIR, "data", "reward_seeds")
    CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")

    if not os.path.exists(SEED_DIR):
        return

    seed_files = [f for f in os.listdir(SEED_DIR) if f.endswith(".json")]
    if not seed_files:
        return

    cache_data = load_cache(CACHE_PATH)
    existing_profiles = cache_data.get("reward_profiles", {})

    for filename in seed_files:
        seed_path = os.path.join(SEED_DIR, filename)
        try:
            with open(seed_path, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
            profiles = [RewardRateProfile.model_validate(p) for p in seed_data.get("profiles", [])]
        except Exception as e:
            print(f"Warning: Failed to parse reward seed {filename}: {e}")
            continue

        # Don't let a manual seed clobber a genuinely PDF-extracted profile for the same card
        profiles = [
            p for p in profiles
            if existing_profiles.get(p.card_id, {}).get("source") != "pdf_extracted"
        ]

        add_reward_profiles_to_cache(profiles, cache_data)
        print(f"Loaded {len(profiles)} manually seeded reward profiles from {filename}")

    save_cache(cache_data, CACHE_PATH)


if __name__ == "__main__":
    run_ingestion_pipeline()
    run_card_reconciliation()
    run_reward_ingestion_pipeline()
    load_manual_reward_seeds()


