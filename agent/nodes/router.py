import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState

# Load API key
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")

class QueryAnalysis(BaseModel):
    """The structured analysis of the user query."""
    intent: str = Field(
        ..., 
        description="One of: 'spend_optimization' (maximizing points/reward logic), "
                    "'eligibility' (salary/eligibility requirements), "
                    "'comparison' (comparing fees, premium status, features), "
                    "'general' (general queries requiring database lookup), "
                    "'greeting' (hello, hi, greetings, or casual chatter where no information retrieval is needed), "
                    "'out_of_domain' (questions completely unrelated to credit cards, finance, or banking, e.g., general knowledge, sports, history)."
    )
    user_income: Optional[float] = Field(
        None, 
        description="The annual income or salary explicitly mentioned in rupees. Null if not specified."
    )
    user_spends: Optional[Dict[str, float]] = Field(
        None, 
        description="Monthly spends categorized by verticals: 'travel', 'dining', 'fuel', 'shopping', 'utilities', 'general'. Null if not specified."
    )
    selected_card_id: Optional[str] = Field(
        None, 
        description="The url-safe slug of the card being referenced (e.g. 'legend', 'pioneer-private', 'celesta'). Null if not explicitly mentioned."
    )

def load_valid_card_slugs() -> list:
    """Dynamically load the list of all registered card slugs from the ingestion cache."""
    from processing.ingestion_cache import load_cache
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ingestion_cache.json")
    cache_data = load_cache(CACHE_PATH)
    
    slugs = []
    for catalog in cache_data.get("discovered_catalogs", {}).values():
        for card in catalog.get("cards_found", []):
            slugs.append(card["card_id"])
    return sorted(list(set(slugs)))

def intent_router(state: AgentState) -> Dict[str, Any]:
    """Analyze conversation history, categorize user intent, and extract profile details."""
    print("--- Running Intent Router Node ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=KEY)
    structured_llm = llm.with_structured_output(QueryAnalysis, method="json_mode")

    # Format history for context
    history_text = ""
    for msg in state.get("messages", []):
        sender = "User" if msg.type == "human" else "AI"
        history_text += f"{sender}: {msg.content}\n"
        
    # Dynamically resolve valid database slugs for Entity Resolution
    valid_slugs = load_valid_card_slugs()
    valid_slugs_str = ", ".join(valid_slugs) if valid_slugs else "No registered cards found"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an expert intent router for a credit card advisor.\n"
            "Analyze the conversation history and classify the user's latest query.\n"
            "Identify vertical categories for user spends: 'travel', 'dining', 'fuel', 'shopping', 'utilities', 'general'.\n"
            f"If the user mentions or references a credit card, you MUST select its exact slug from this list of valid slugs in our database: [{valid_slugs_str}].\n"
            "If the card is not in the list or is a general query, set selected_card_id to null.\n"
            "Provide output in strict accordance with the QueryAnalysis JSON schema."
        ),
        HumanMessage(f"Conversation History:\n{history_text}")
    ])

    analysis_chain = (prompt | structured_llm).with_retry(stop_after_attempt=4, wait_exponential_jitter=True)
    
    try:
        analysis = analysis_chain.invoke({})
        print(f"Detected Intent: {analysis.intent}")
        print(f"Extracted parameters: Income={analysis.user_income}, Spends={analysis.user_spends}, CardID={analysis.selected_card_id}")
        
        # If no selected_card_id is extracted but one was stored in state previously, keep the state's context
        updated_card_id = analysis.selected_card_id or state.get("selected_card_id")
        
        return {
            "intent": analysis.intent,
            "user_income": analysis.user_income,
            "user_spends": analysis.user_spends,
            "selected_card_id": updated_card_id
        }
    except Exception as e:
        print(f"Warning: Router Node failed: {e}. Defaulting to general intent.")
        return {
            "intent": "general"
        }
