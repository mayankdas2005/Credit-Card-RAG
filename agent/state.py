from typing import Annotated, TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the RAG agent graph, tracked across conversational turns."""
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[str]                  # 'spend_optimization', 'eligibility', 'comparison', 'general'
    user_income: Optional[float]
    user_spends: Optional[Dict[str, float]]  # e.g., {'travel': 20000.0, 'dining': 5000.0}
    retrieved_docs: List[Any]
    retrieval_grounded: Optional[bool]     # True = card-specific filtered hit, False = unfiltered fallback/no match
    calculation_results: Optional[Dict[str, Any]]
    selected_card_id: Optional[str]        # Context tracking for contextual follow-up questions
