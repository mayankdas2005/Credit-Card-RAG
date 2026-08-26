from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes.router import intent_router
from agent.nodes.retriever import retriever_node
from agent.nodes.calculator import calculator_node
from agent.nodes.synthesizer import synthesizer_node

def route_after_router(state: AgentState) -> str:
    """Decide whether to route to the retriever node or straight to the synthesizer for greetings and out of domain queries."""
    if state.get("intent") in ["greeting", "out_of_domain"]:
        return "synthesizer"
    return "retriever"

def route_after_retrieval(state: AgentState) -> str:
    """Decide whether to route to the calculator tool node or straight to the synthesizer."""
    if state.get("intent") == "spend_optimization":
        return "calculator"
    return "synthesizer"

# Initialize stateful graph
workflow = StateGraph(AgentState)

# Add modular action nodes
workflow.add_node("router", intent_router)
workflow.add_node("retriever", retriever_node)
workflow.add_node("calculator", calculator_node)
workflow.add_node("synthesizer", synthesizer_node)

# Construct routing flows
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "retriever": "retriever",
        "synthesizer": "synthesizer"
    }
)

workflow.add_conditional_edges(
    "retriever",
    route_after_retrieval,
    {
        "calculator": "calculator",
        "synthesizer": "synthesizer"
    }
)

workflow.add_edge("calculator", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile graph with thread checkpointers for conversational memory
memory = MemorySaver()
rag_agent = workflow.compile(checkpointer=memory)
