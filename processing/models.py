from pydantic import BaseModel, Field
from typing import List


class CardIdentity(BaseModel):
    card_id: str = Field(..., description="Unique URL-safe slug, e.g., 'bank-card-slug'")
    card_name: str = Field(..., description="Full official name of the card")
    joining_fee: float = Field(0.0, description="Joining fee. Use 0.0 if free or not found.")
    annual_fee: float = Field(0.0, description="Annual renewal fee. Use 0.0 if not found.")
    fee_waiver_threshold: float = Field(0.0, description="Spend threshold required to waive the annual fee. Use 0.0 if none.")

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
    applicable_card_id: str = Field(..., description="The card ID from the provided list, or 'generic'.")
    benefit_type: str = Field("general", description="lounge_access, cashback, reward_multiplier, fees_and_charges, legal_terms")
    spend_category: str = Field("general", description="travel, dining, fuel, shopping, utilities, general")
    partner_merchant: str = Field("none", description="Co-branded partner name or 'none'")

class BatchClassificationResponse(BaseModel):
    results: List[BatchItemClassification]