import json
import os
from typing import Dict, Any, Optional
from processing.models import BankAndCardCatalog

def load_cache(cache_path: str) -> Dict[str, Any]:
    """Load the ingestion cache from the specified JSON file path."""
    if not os.path.exists(cache_path):
        return {
            "fully_indexed_files": [],
            "discovered_catalogs": {}
        }
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}. Initializing a clean cache.")
        return {
            "fully_indexed_files": [],
            "discovered_catalogs": {}
        }

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
