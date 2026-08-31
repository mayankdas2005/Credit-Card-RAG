import json
import os
from typing import Dict, Any, Optional, List
from processing.models import BankAndCardCatalog, RewardRateProfile, CardMergeGroup

def _default_cache() -> Dict[str, Any]:
    return {
        "fully_indexed_files": [],
        "discovered_catalogs": {},
        "fully_indexed_reward_files": [],
        "reward_profiles": {},
        "card_aliases": {},
        "reconciled_banks": {}
    }

def load_cache(cache_path: str) -> Dict[str, Any]:
    """Load the ingestion cache from the specified JSON file path."""
    if not os.path.exists(cache_path):
        return _default_cache()
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}. Initializing a clean cache.")
        return _default_cache()

    # Backfill keys for caches written before the reward-profile extension existed.
    cache_data.setdefault("fully_indexed_reward_files", [])
    cache_data.setdefault("reward_profiles", {})
    cache_data.setdefault("card_aliases", {})
    cache_data.setdefault("reconciled_banks", {})
    return cache_data

def save_cache(cache_data: Dict[str, Any], cache_path: str) -> None:
    """Save the current state of the ingestion cache to the JSON file path."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Failed to save cache to {cache_path}: {e}")

def is_file_indexed(filename: str, cache_data: Dict[str, Any]) -> bool:
    """Check if a PDF file has already been fully processed and indexed in the Chroma DB."""
    return filename in cache_data.get("fully_indexed_files", [])

def get_cached_catalog(filename: str, cache_data: Dict[str, Any]) -> Optional[BankAndCardCatalog]:
    """Retrieve the cached BankAndCardCatalog object for a file, if it exists."""
    catalog_dict = cache_data.get("discovered_catalogs", {}).get(filename)
    if catalog_dict:
        try:
            if hasattr(BankAndCardCatalog, "model_validate"):
                return BankAndCardCatalog.model_validate(catalog_dict)
            else:
                return BankAndCardCatalog.parse_obj(catalog_dict)
        except Exception as e:
            print(f"Warning: Failed to parse cached catalog for {filename}: {e}")
            return None
    return None

def add_catalog_to_cache(filename: str, catalog: BankAndCardCatalog, cache_data: Dict[str, Any]) -> None:
    """Add a BankAndCardCatalog object to the discovered_catalogs section of the cache."""
    if "discovered_catalogs" not in cache_data:
        cache_data["discovered_catalogs"] = {}
    
    if hasattr(catalog, "model_dump"):
        cache_data["discovered_catalogs"][filename] = catalog.model_dump()
    else:
        cache_data["discovered_catalogs"][filename] = catalog.dict()

def mark_file_indexed(filename: str, cache_data: Dict[str, Any]) -> None:
    """Mark a file as fully indexed by adding it to the fully_indexed_files list in the cache."""
    if "fully_indexed_files" not in cache_data:
        cache_data["fully_indexed_files"] = []
    if filename not in cache_data["fully_indexed_files"]:
        cache_data["fully_indexed_files"].append(filename)


def is_reward_file_indexed(filename: str, cache_data: Dict[str, Any]) -> bool:
    """Check if a rewards-program PDF has already been processed into reward_profiles."""
    return filename in cache_data.get("fully_indexed_reward_files", [])


def mark_reward_file_indexed(filename: str, cache_data: Dict[str, Any]) -> None:
    """Mark a rewards-program PDF as fully processed."""
    if "fully_indexed_reward_files" not in cache_data:
        cache_data["fully_indexed_reward_files"] = []
    if filename not in cache_data["fully_indexed_reward_files"]:
        cache_data["fully_indexed_reward_files"].append(filename)


def add_reward_profiles_to_cache(profiles: List[RewardRateProfile], cache_data: Dict[str, Any]) -> None:
    """Merge a list of RewardRateProfile objects into the reward_profiles cache section, keyed by card_id.
    Works identically regardless of whether profiles came from PDF extraction or a manual seed file."""
    if "reward_profiles" not in cache_data:
        cache_data["reward_profiles"] = {}

    for profile in profiles:
        payload = profile.model_dump() if hasattr(profile, "model_dump") else profile.dict()
        cache_data["reward_profiles"][payload["card_id"]] = payload


def get_all_reward_profiles(cache_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the full card_id -> reward profile dict, regardless of source (PDF or manual seed)."""
    return cache_data.get("reward_profiles", {})


def get_canonical_card_id(card_id: str, cache_data: Dict[str, Any]) -> str:
    """Resolve a card_id to its canonical identity if it's a known alias, otherwise return unchanged."""
    return cache_data.get("card_aliases", {}).get(card_id, card_id)


def get_card_alias_group(card_id: str, cache_data: Dict[str, Any]) -> List[str]:
    """Return every card_id (self, canonical, and sibling aliases) known to refer to the same real
    card as the given card_id. Used to widen a Chroma metadata filter so it finds chunks regardless
    of which alias a given source document happened to tag them with."""
    aliases = cache_data.get("card_aliases", {})
    canonical = aliases.get(card_id, card_id)
    group = {card_id, canonical}
    group.update(alias for alias, canon in aliases.items() if canon == canonical)
    return sorted(group)


def record_merge_groups(merge_groups: List[CardMergeGroup], cache_data: Dict[str, Any]) -> None:
    """Store a bank's reconciled duplicate-card groups as alias_card_id -> canonical_card_id entries."""
    if "card_aliases" not in cache_data:
        cache_data["card_aliases"] = {}
    for group in merge_groups:
        payload = group.model_dump() if hasattr(group, "model_dump") else group.dict()
        for alias in payload["alias_card_ids"]:
            if alias != payload["canonical_card_id"]:
                cache_data["card_aliases"][alias] = payload["canonical_card_id"]


def get_bank_reconciliation_fingerprint(bank_id: str, cache_data: Dict[str, Any]) -> List[str]:
    """The sorted list of source filenames currently contributing cards to a given bank_id — used to
    detect when reconciliation needs to be rerun because a new document was added for that bank."""
    files = [
        fname for fname, catalog in cache_data.get("discovered_catalogs", {}).items()
        if catalog.get("bank_id") == bank_id
    ]
    return sorted(files)


def is_bank_reconciled(bank_id: str, cache_data: Dict[str, Any]) -> bool:
    """Check whether the current set of source files for this bank has already been reconciled."""
    current_fingerprint = get_bank_reconciliation_fingerprint(bank_id, cache_data)
    return cache_data.get("reconciled_banks", {}).get(bank_id) == current_fingerprint


def mark_bank_reconciled(bank_id: str, cache_data: Dict[str, Any]) -> None:
    if "reconciled_banks" not in cache_data:
        cache_data["reconciled_banks"] = {}
    cache_data["reconciled_banks"][bank_id] = get_bank_reconciliation_fingerprint(bank_id, cache_data)
