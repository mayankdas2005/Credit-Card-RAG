import sys
import os

# Ensure the root directory is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from agent.rag_agent import rag_agent
from langchain_core.messages import HumanMessage

def print_banner():
    print("\n" + "="*60)
    print("      💳  STATEFUL CREDIT CARD ADVISOR (LangGraph RAG)  💳")
    print("="*60)
    print("Welcome! I can help you optimize your reward points, recommend")
    print("the best cards based on your salary or categories, and compare")
    print("card fees and lounge details.")
    print("\nCommands:")
    print("  - Type 'exit' or 'quit' to close the app.")
    print("  - Type 'clear' to reset conversation history (new thread).")
    print("="*60 + "\n")

def run_chat_app():
    print_banner()
    
    # Initialize thread configuration for stateful memory checkpointers
    thread_counter = 1
    config = {"configurable": {"thread_id": f"session-{thread_counter}"}}
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if user_input.lower() == "clear":
            thread_counter += 1
            config = {"configurable": {"thread_id": f"session-{thread_counter}"}}
            print("\n[SYSTEM] Conversation history cleared. Started a new chat session.\n")
            continue
            
        print("\n[SYSTEM] Processing request through LangGraph...")
        
        try:
            # Invoke the LangGraph agent state machine
            state = rag_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # The last message is the AI synthesizer output
            message_list = state.get("messages",[])
            last_message = message_list[-1]
            print(f"\nAdvisor:\n{last_message.content}\n")
            print("-" * 60 + "\n")
            
        except Exception as e:
            print(f"\nError: An issue occurred during graph execution: {e}\n")

if __name__ == "__main__":
    run_chat_app()
