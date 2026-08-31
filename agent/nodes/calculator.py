import os
from typing import Dict, Any
from agent.state import AgentState
from processing.ingestion_cache import load_cache, get_all_reward_profiles, get_canonical_card_id

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")


def calculator_node(state: AgentState) -> Dict[str, Any]:
    """Execute mathematical calculations to compare reward yields for the user spends.

    Fully data-driven: ranks every card_id that has a reward profile in the cache (whether sourced
    from an ingested rewards-program PDF or a manual seed file), no bank or card list hardcoded here.
    Cards without a reward profile are excluded from ranking rather than guessed at.
    """
    print("--- Running Calculator Node ---")

    intent = state.get("intent")
    user_spends = state.get("user_spends")

    if not user_spends or intent != "spend_optimization":
        print("No category spends found or intent is not optimization. Skipping calculation.")
        return {"calculation_results": None}

    cache_data = load_cache(CACHE_PATH)
    reward_profiles = get_all_reward_profiles(cache_data)

    # Flat lookup of discovered cards (name + confirmed fee) across every ingested bank.
    # bank_name lives on the parent catalog, not on each card dict in cards_found, so it's
    # stamped onto each card here rather than looked up later.
    discovered_cards = {}
    all_banks = set()
    for catalog in cache_data.get("discovered_catalogs", {}).values():
        bank_name = catalog.get("bank_name")
        all_banks.add(bank_name)
        for card in catalog.get("cards_found", []):
            discovered_cards[card["card_id"]] = {**card, "bank_name": bank_name}

    if not reward_profiles:
        print("No reward profiles available (no rewards PDFs or seed files ingested yet). Skipping calculation.")
        return {"calculation_results": None}

    results = []
    covered_banks = set()

    for card_id, profile in reward_profiles.items():
        # A reward profile's card_id might be an alias of a card whose fee/name data was recorded
        # under a different (canonical) card_id from a different source document — fall back to the
        # canonical id so fee lookups still resolve instead of silently degrading to "unconfirmed".
        card_meta = discovered_cards.get(card_id) or discovered_cards.get(get_canonical_card_id(card_id, cache_data))
        card_name = card_meta.get("card_name") if card_meta else card_id.replace("-", " ").title()

        card_bank_name = card_meta.get("bank_name") if card_meta else None

        cached_fee = card_meta.get("annual_fee") if card_meta else None
        if cached_fee is not None:
            annual_fee = cached_fee
            fee_confirmed = True
        else:
            # No confirmed fee anywhere in the ingested MITC catalog for this card — use 0 purely as an
            # arithmetic placeholder and flag it, rather than fabricating a plausible-looking number.
            annual_fee = 0.0
            fee_confirmed = False

        category_rates = profile.get("category_rates", {})
        base_rate = profile.get("base_rate", 1.0)
        point_value = profile.get("point_value_inr", 1.0)

        total_points = 0.0
        spend_breakdown = {}
        for category, monthly_spend in user_spends.items():
            multiplier = category_rates.get(category, base_rate)
            annual_category_spend = monthly_spend * 12
            category_points = (annual_category_spend / 100.0) * multiplier
            total_points += category_points
            spend_breakdown[category] = {
                "annual_spend": annual_category_spend,
                "multiplier": multiplier,
                "points_earned": category_points
            }

        rewards_cash_value = total_points * point_value
        net_annual_benefit = rewards_cash_value - annual_fee

        if card_bank_name:
            covered_banks.add(card_bank_name)

        results.append({
            "card_id": card_id,
            "card_name": card_name,
            "annual_fee": annual_fee,
            "fee_confirmed": fee_confirmed,
            "annual_points": total_points,
            "rewards_cash_value": rewards_cash_value,
            "net_annual_benefit": net_annual_benefit,
            "spend_breakdown": spend_breakdown,
            "point_value": point_value
        })

    sorted_results = sorted(results, key=lambda x: x["net_annual_benefit"], reverse=True)

    uncovered_banks = sorted(b for b in all_banks if b and b not in covered_banks)
    if sorted_results:
        print(f"Calculation complete over {len(sorted_results)} cards. Top card: {sorted_results[0]['card_name']} "
              f"(Net Benefit: {sorted_results[0]['net_annual_benefit']:.2f} INR)")
    if uncovered_banks:
        print(f"Note: no reward-rate data yet for: {', '.join(uncovered_banks)} — excluded from ranking.")

    return {
        "calculation_results": {
            "sorted_recommendations": sorted_results,
            "input_spends": user_spends,
            "uncovered_banks": uncovered_banks
        }
    }
