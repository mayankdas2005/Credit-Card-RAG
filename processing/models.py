from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal


class CardIdentity(BaseModel):
    card_id: str = Field(..., description="Unique URL-safe slug, e.g., 'bank-card-slug'")
    card_name: str = Field(..., description="Full official name of the card")
    joining_fee: Optional[float] = Field(
        None,
        description="Joining fee, ONLY if an explicit numeric figure appears in the text for this card. "
                    "Use 0.0 only if the text explicitly states the card is free / fee-waived. "
                    "Use null if no joining fee figure is stated anywhere in the text — do not guess."
    )
    annual_fee: Optional[float] = Field(
        None,
        description="Annual renewal fee, ONLY if an explicit numeric figure appears in the text for this card. "
                    "Use 0.0 only if the text explicitly states the annual fee is free / waived / NIL. "
                    "Use null if no annual fee figure is stated anywhere in the text — do not guess or assume 0."
    )
    fee_waiver_threshold: Optional[float] = Field(
        None,
        description="Spend threshold required to waive the annual fee, ONLY if explicitly stated. "
                    "Use null if not stated — do not guess."
    )

class BankAndCardCatalog(BaseModel):
    bank_name: str = Field(
        ..., 
        description="Official name of the bank found in the text, e.g., 'Generic Bank'"
    )
    bank_id: str = Field(
        ..., 
        description="A clean, lowercase url-safe slug based on the bank name, e.g., 'generic-bank'"
    )
    cards_found: List[CardIdentity] = Field(..., description="List of all credit cards found in this section.")


class ChunkFeatureExtractor(BaseModel):
    applicable_card_id: str = Field(..., description="Match this text to one of the discovered card_ids. Use 'generic' if it's broad boilerplate text.")
    benefit_type: str = Field("general", description="Must be one of: lounge_access, cashback, reward_multiplier, fees_and_charges, legal_terms")
    spend_category: str = Field("general", description="Main spend vertical: travel, dining, fuel, shopping, utilities, general")
    partner_merchant: str = Field("none", description="Co-branded partner name (e.g., Marriott, Swiggy) or 'none'")


class BatchItemClassification(BaseModel):
    paragraph_index: int = Field(..., description="The index of the paragraph being classified.")
    applicable_card_ids: List[str] = Field(
        ...,
        description="ALL card IDs from the provided list that this paragraph's content applies to. "
                    "Many paragraphs (e.g. a shared table row covering several cards) apply to multiple cards at once — "
                    "list every one of them. Use ['generic'] ONLY if the content is truly bank-wide boilerplate or "
                    "applies uniformly to literally all cards, not just because multiple cards happen to be mentioned."
    )
    benefit_type: str = Field("general", description="lounge_access, cashback, reward_multiplier, fees_and_charges, legal_terms")
    spend_category: str = Field("general", description="travel, dining, fuel, shopping, utilities, general")
    partner_merchant: str = Field("none", description="Co-branded partner name or 'none'")

class BatchClassificationResponse(BaseModel):
    results: List[BatchItemClassification]


class RewardRateProfile(BaseModel):
    """A single card's reward-earning structure, sourced either from an ingested rewards-program
    PDF or a manually curated seed file when no such document has been sourced yet."""
    card_id: str = Field(..., description="Must match a card_id already known from the bank's MITC/card catalog.")
    reward_unit: str = Field(..., description="Unit the reward is earned in, e.g. 'points', 'cashback_percent', 'miles'.")
    point_value_inr: float = Field(..., description="INR cash value of a single reward unit when redeemed.")
    category_rates: Dict[str, float] = Field(
        ...,
        description="Reward multiplier per spend vertical, keyed by: travel, dining, fuel, shopping, utilities, general. "
                    "Only include categories with an explicitly stated rate."
    )
    base_rate: float = Field(..., description="Default/base earn rate applied to spend categories not explicitly listed.")
    source: Literal["pdf_extracted", "manual_seed"] = Field(..., description="Where this profile's numbers came from.")


class RewardCatalogResponse(BaseModel):
    """Structured output wrapper for reward-rate extraction over a single rewards-program document."""
    profiles: List[RewardRateProfile]


class CardMergeGroup(BaseModel):
    """A group of card_ids discovered across different source documents that refer to the exact
    same real-world card product (e.g. one document abbreviated the name differently than another)."""
    canonical_card_id: str = Field(
        ...,
        description="The most complete/descriptive card_id among the group to keep as the canonical identity."
    )
    alias_card_ids: List[str] = Field(
        ...,
        description="Other card_ids in the group that refer to the same real card as canonical_card_id. "
                    "Must not include canonical_card_id itself."
    )
    reasoning: str = Field(..., description="One-sentence justification for why these are the same card.")


class CardReconciliationResponse(BaseModel):
    """Structured output for a duplicate-detection pass over one bank's full discovered card list."""
    merge_groups: List[CardMergeGroup] = Field(
        default_factory=list,
        description="Groups of card_ids to merge. Only include a group when confident it's the EXACT same "
                    "product — never merge genuinely distinct cards/tiers/co-branded variants just because "
                    "their names are similar."
    )