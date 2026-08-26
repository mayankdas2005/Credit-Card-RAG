import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from agent.state import AgentState

# Load API key
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Combine context, state, and calculation values to draft the final response to the user."""
    print("--- Running Synthesizer Node ---")
    
    intent = state.get("intent")
    user_income = state.get("user_income")
    user_spends = state.get("user_spends")
    retrieved_docs = state.get("retrieved_docs", [])
    retrieval_grounded = state.get("retrieval_grounded", False)
    calc_results = state.get("calculation_results")
    messages = state.get("messages", [])

    # 1. Format the retrieved document context
    doc_context = ""
    for idx, doc in enumerate(retrieved_docs):
        doc_context += f"[Document {idx+1}]\n{doc.page_content}\nMetadata: {doc.metadata}\n\n"

    if not retrieved_docs:
        doc_context = "No documents were retrieved for this query — the knowledge base has no grounded information here."
    elif not retrieval_grounded:
        doc_context += (
            "\nNOTE: These documents were NOT found via a confirmed card-specific match — they are a broader "
            "fallback search and may not be specific to the exact card the user is asking about."
        )
        
    # 2. Format the calculation results context
    calc_context = "No calculations performed for this query."
    recommended_card_id = state.get("selected_card_id")
    
    if calc_results:
        recs = calc_results["sorted_recommendations"]
        if recs:
            recommended_card_id = recs[0]["card_id"] # Capture the top recommended card
            calc_context = "Ranked Card Recommendations (based on programmatic calculations):\n"
            for rank, r in enumerate(recs[:3]):
                fee_note = "" if r.get("fee_confirmed", True) else " (NOT stated in source documents — estimated only, tell the user this is unconfirmed)"
                calc_context += (
                    f"Rank {rank+1}: {r['card_name']} (ID: {r['card_id']})\n"
                    f"  - Annual Fee: {r['annual_fee']:.2f} INR{fee_note}\n"
                    f"  - Annual Points Earned: {r['annual_points']:.2f}\n"
                    f"  - Points Cash Value: {r['rewards_cash_value']:.2f} INR (at value {r['point_value']} per point)\n"
                    f"  - Net Annual Benefit: {r['net_annual_benefit']:.2f} INR (Rewards minus Annual Fee)\n\n"
                )
            uncovered_banks = calc_results.get("uncovered_banks") or []
            if uncovered_banks:
                calc_context += (
                    f"IMPORTANT: Reward-rate data is not yet available for these banks, so their cards could NOT be "
                    f"included in this ranking: {', '.join(uncovered_banks)}. You MUST tell the user this ranking only "
                    f"covers banks with sourced reward data and does not yet include those banks.\n"
                )

    # 3. Create the prompt instruction
    if intent == "greeting":
        system_instruction = (
            "You are an elite, friendly financial credit card advisor.\n"
            "The user has just greeted you or initiated casual chatter.\n"
            "Respond in a warm, welcoming, and professional manner, briefly explaining how you can assist them "
            "(e.g., finding the best cards for their salary, optimizing category spending to maximize reward point value, "
            "or comparing card fees/lounge access rules)."
        )
    elif intent == "out_of_domain":
        system_instruction = (
            "You are an elite, friendly financial credit card advisor.\n"
            "The user is asking a question completely unrelated to credit cards, personal finance, or banking.\n"
            "Task: Answer their question politely using your general knowledge, and add a brief, friendly remark "
            "reminding them that your primary expertise is helping them optimize and compare credit cards."
        )
    else:
        system_instruction = (
            "Your task is to answer the user's latest query accurately using the provided contexts (Document Context & Calculation Context).\n"
            "Rules:\n"
            "1. If calculations are present, ALWAYS cite them to explain the card rankings and net benefits in detail.\n"
            "2. If eligibility is queried, verify the user's income against the document requirements.\n"
            "3. Keep your advice professional, structured in clear markdown, and grounded in the retrieved details.\n"
            "4. If a card is recommended, explain how its reward point value can be maximized (e.g. transfers to hotels/airlines or airmiles).\n"
            "5. Do not hallucinate calculations. Strictly use the values provided in the Calculation Context.\n"
            "6. If a card's Annual Fee is marked as NOT stated in source documents, you MUST tell the user that figure is an estimate, "
            "not confirmed from the bank's official terms, and advise them to verify it before applying. Never present an unconfirmed fee as a fact.\n"
            "7. If the Document Context says no documents were retrieved, or notes that results are an unconfirmed fallback, you MUST NOT "
            "state specific facts (fees, rates, eligibility criteria, benefit details) as if they are confirmed for that card. Explicitly tell "
            "the user you don't have verified, document-grounded information for that specific point, and suggest they check the official "
            "source. This applies to every factual claim, not just fees."
        )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(system_instruction),
        HumanMessage(
            f"User Profile Info:\n"
            f"  - Income: {user_income if user_income else 'Not specified'}\n"
            f"  - Spends: {user_spends if user_spends else 'Not specified'}\n\n"
            f"Document Context (Chroma DB):\n{doc_context}\n"
            f"Calculation Context (Python Tool Node):\n{calc_context}\n\n"
            f"Conversation History:\n"
            f"{''.join([f'{m.type.upper()}: {m.content}\n' for m in messages])}"
        )
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=KEY).with_retry(
        stop_after_attempt=4, wait_exponential_jitter=True
    )
    response = llm.invoke(prompt.format_messages())
    
    # Extract content text safely (handles string or block lists)
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        content_str = "".join(text_parts)
    else:
        content_str = str(content)
        
    ai_message = AIMessage(content=content_str)
    
    return {
        "messages": [ai_message],
        "selected_card_id": recommended_card_id
    }
