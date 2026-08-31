"""
Evaluation test cases for the credit-card RAG advisor.

Ground truth for fee figures below was pulled directly from cache/ingestion_cache.json's
discovered_catalogs (the LLM-extracted fee data from the source MITC PDFs), not guessed.

Each case is:
  id: short unique identifier
  category: groups cases in the report (grounded_fact / hallucination_guardrail / comparison /
            spend_optimization / eligibility / out_of_domain / greeting / multi_turn)
  turns: list of user messages sent in order on the same conversation thread; checks apply to
         the response to the LAST turn only (earlier turns just build up context/memory)
  checks: list of check dicts, all of which must pass for the case to pass. Types:
    - contains_any: {"phrases": [...]}      -> at least one phrase (case-insensitive) is in the answer
    - contains_all: {"phrases": [...]}      -> every phrase is in the answer
    - not_contains: {"phrases": [...]}      -> none of the phrases are in the answer
    - intent_equals: {"value": "..."}       -> final state's detected intent matches exactly
"""

HEDGE_PHRASES = [
    "do not have verified", "don't have verified", "not have verified",
    "cannot confirm", "can't confirm", "not confirmed", "unconfirmed",
    "not stated in", "no verified", "not verified", "don't have",
    "do not have", "unable to confirm", "not documented",
]

CASES = [
    {
        "id": "greeting_basic",
        "category": "greeting",
        "turns": ["Hi there"],
        "checks": [
            {"type": "intent_equals", "value": "greeting"},
            {"type": "contains_any", "phrases": ["credit card", "advisor", "help you"]},
        ],
    },
    {
        "id": "sbi_elite_fee_grounded",
        "category": "grounded_fact",
        "turns": ["What is the annual fee for the SBI Card ELITE?"],
        "checks": [
            {"type": "contains_any", "phrases": ["4,999", "4999"]},
        ],
    },
    {
        "id": "sbi_prime_fee_grounded",
        "category": "grounded_fact",
        "turns": ["What is the annual fee for the SBI Card PRIME?"],
        "checks": [
            {"type": "contains_any", "phrases": ["2,999", "2999"]},
        ],
    },
    {
        "id": "hdfc_infinia_fee_grounded",
        "category": "grounded_fact",
        "turns": ["What is the annual fee for the HDFC Infinia credit card?"],
        "checks": [
            {"type": "contains_any", "phrases": ["10,000", "10000"]},
        ],
    },
    {
        "id": "hdfc_regalia_gold_fee_grounded",
        "category": "grounded_fact",
        "turns": ["What is the annual fee of the HDFC Regalia Gold card?"],
        "checks": [
            {"type": "contains_any", "phrases": ["2,500", "2500"]},
        ],
    },
    {
        "id": "axis_buzz_fee_grounded",
        "category": "grounded_fact",
        "turns": ["What is the annual fee for the Axis Bank Buzz credit card?"],
        "checks": [
            {"type": "contains_any", "phrases": ["750"]},
        ],
    },
    {
        "id": "indusind_pinnacle_fee_hedge",
        "category": "hallucination_guardrail",
        "turns": ["What is the annual fee for the IndusInd Pinnacle card?"],
        "checks": [
            {"type": "contains_any", "phrases": HEDGE_PHRASES},
            {"type": "not_contains", "phrases": ["the annual fee is rs", "the annual fee is ₹"]},
        ],
    },
    {
        "id": "indusind_celesta_fee_hedge",
        "category": "hallucination_guardrail",
        "turns": ["What is the annual fee for the IndusInd Celesta card?"],
        "checks": [
            {"type": "contains_any", "phrases": HEDGE_PHRASES},
        ],
    },
    {
        "id": "hdfc_diners_black_fee_hedge",
        "category": "hallucination_guardrail",
        "turns": ["What is the annual fee for the HDFC Diners Club Black card?"],
        "checks": [
            {"type": "contains_any", "phrases": HEDGE_PHRASES},
        ],
    },
    {
        "id": "comparison_sbi_elite_vs_prime",
        "category": "comparison",
        "turns": ["Compare the SBI Card ELITE and SBI Card PRIME"],
        "checks": [
            {"type": "contains_all", "phrases": ["4,999", "2,999"]},
        ],
    },
    {
        "id": "multiturn_followup_lower_fee",
        "category": "multi_turn",
        "turns": [
            "Compare the SBI Card ELITE and SBI Card PRIME",
            "Which one has the lower annual fee?",
        ],
        "checks": [
            {"type": "contains_any", "phrases": ["prime"]},
        ],
    },
    {
        "id": "spend_optimization_general",
        "category": "spend_optimization",
        "turns": ["I spend 20000 a month on dining and 15000 on travel, which card should I use to maximize rewards?"],
        "checks": [
            {"type": "intent_equals", "value": "spend_optimization"},
            {"type": "contains_any", "phrases": ["celesta", "crest", "club vistara", "atlas", "horizon", "olympus", "miles"]},
        ],
    },
    {
        "id": "spend_optimization_uncovered_bank_disclosure",
        "category": "spend_optimization",
        "turns": ["I spend 25000 on travel and 10000 on dining every month, what's my best card?"],
        "checks": [
            {"type": "contains_any", "phrases": ["hdfc"]},
            {"type": "contains_any", "phrases": ["not covered", "excluded", "not yet available", "not included", "no reward", "does not include", "doesn't include", "does not yet include", "doesn't yet include", "was not available"]},
        ],
    },
    {
        "id": "eligibility_no_fabricated_income_rule",
        "category": "eligibility",
        "turns": ["Am I eligible for the SBI Card ELITE if I earn 12 lakhs a year?"],
        "checks": [
            {"type": "intent_equals", "value": "eligibility"},
            {"type": "contains_any", "phrases": ["4,999", "4999"]},
            {"type": "not_contains", "phrases": ["you are eligible", "you qualify", "yes, you are"]},
        ],
    },
    {
        "id": "out_of_domain_weather",
        "category": "out_of_domain",
        "turns": ["What's the weather like today?"],
        "checks": [
            {"type": "intent_equals", "value": "out_of_domain"},
            {"type": "contains_any", "phrases": ["weather", "forecast"]},
        ],
    },
]
