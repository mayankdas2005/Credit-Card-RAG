import os
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from agent.state import AgentState

# Load API key
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")

from processing.ingestion_cache import load_cache, get_card_alias_group

def resolve_card_ids(selected_card_id: str, cache_path: str) -> List[str]:
    """Find all valid card IDs from cache that share core name tokens with selected_card_id."""
    if not selected_card_id:
        return []

    cache_data = load_cache(cache_path)

    # Extract unique card IDs from the cache
    all_card_ids = []
    for catalog in cache_data.get("discovered_catalogs", {}).values():
        for card in catalog.get("cards_found", []):
            all_card_ids.append(card["card_id"])

    all_card_ids = list(set(all_card_ids))

    # The router is constrained to only select slugs from the valid card-id list it's given, so in
    # the common case selected_card_id is already an exact, unambiguous match — use it directly.
    # Falling through to fuzzy token matching below strips tier words ("elite"/"prime"/"select"/etc)
    # that are often a card's ONLY distinguishing feature (e.g. "SBI Card ELITE" vs "SBI Card PRIME"),
    # which would otherwise broaden the filter to every card sharing the same bank.
    if selected_card_id in all_card_ids:
        # Expand to the full alias group: the same real card can be tagged with a different card_id
        # in Chroma depending on which source document a given chunk came from (see card_aliases /
        # the reconciliation pass), so search on all known identities, not just this one.
        return get_card_alias_group(selected_card_id, cache_data)

    # Tokenize and filter out generic card/bank terms or common qualifiers
    generic_terms = {
        "card", "cards", "bank", "premium", "elite", "prime", "select", 
        "private", "legacy", "gold", "metal", "edition", "activ", "plus", 
        "money", "save", "click", "limit", "limited", "ltd", "co"
    }
    
    target_tokens = [
        t for t in selected_card_id.lower().replace("-", " ").split()
        if t not in generic_terms and len(t) > 2
    ]
    
    if not target_tokens:
        # Fallback to direct substring matching if no specific keywords remain
        target = selected_card_id.lower()
        return [cid for cid in all_card_ids if target in cid.lower() or cid.lower() in target]
        
    # Match any card IDs that contain at least one of our core name tokens
    matches = []
    for cid in all_card_ids:
        cid_lower = cid.lower().replace("-", " ")
        if any(token in cid_lower for token in target_tokens):
            matches.append(cid)
            
    return matches

def retrieve_from_chroma(query: str, card_ids: List[str] = None) -> List[Any]:
    """Helper to query Chroma DB, optionally filtering by card_id slug(s)."""
    db_directory = os.path.join(ROOT_DIR, "credit_card_rag_db")
    if not os.path.exists(db_directory):
        print(f"Warning: Vector DB directory not found at {db_directory}")
        return []
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=KEY)
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    
    if card_ids:
        if len(card_ids) == 1:
            print(f"Retrieving from Chroma with metadata filter: card_id='{card_ids[0]}'")
            return db.similarity_search(query, k=4, filter={"card_id": card_ids[0]})
        else:
            print(f"Retrieving from Chroma with metadata filter: card_id IN {card_ids}")
            return db.similarity_search(query, k=4, filter={"card_id": {"$in": card_ids}})
    else:
        print("Retrieving from Chroma via standard similarity search...")
        return db.similarity_search(query, k=5)

def retriever_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve document chunks from Chroma DB based on current state and query."""
    print("--- Running Retriever Node ---")

    messages = state.get("messages", [])
    if not messages:
        return {"retrieved_docs": [], "retrieval_grounded": False}

    latest_query = messages[-1].content
    intent = state.get("intent")
    selected_card_id = state.get("selected_card_id")

    # If the user is asking a contextual question about a specific card, filter by that card.
    # Comparison follow-ups (e.g. "which one has the lower fee?") often have no card names in the
    # latest message at all — the router resolves selected_card_id from conversation history, but
    # without this filter the search falls through to a blind unfiltered query with nothing to
    # anchor on, pulling in unrelated cards instead of the one actually being discussed.
    wants_card_specific = bool(selected_card_id and intent in ["general", "eligibility", "comparison"])
    card_ids_filter = None
    if wants_card_specific:
        CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")
        card_ids_filter = resolve_card_ids(selected_card_id, CACHE_PATH)

    if wants_card_specific and card_ids_filter:
        docs = retrieve_from_chroma(latest_query, card_ids=card_ids_filter)
        if docs:
            print(f"Retrieved {len(docs)} card-specific document chunks.")
            return {"retrieved_docs": docs, "retrieval_grounded": True}
        print(f"No card-specific chunks found for '{selected_card_id}'. Falling back to unfiltered search (unconfirmed for this card).")
    elif wants_card_specific:
        print(f"Card ID '{selected_card_id}' is not in database. Falling back to unfiltered search (unconfirmed for this card).")

    # Either no specific card was requested, or the card-specific search came up empty —
    # run a broad unfiltered search rather than silently returning nothing.
    docs = retrieve_from_chroma(latest_query)
    grounded = bool(docs) and not wants_card_specific
    print(f"Retrieved {len(docs)} document chunks (grounded={grounded}).")
    return {
        "retrieved_docs": docs,
        "retrieval_grounded": grounded
    }
