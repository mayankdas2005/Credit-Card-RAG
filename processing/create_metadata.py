from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from processing.models import BankAndCardCatalog, ChunkFeatureExtractor, BatchClassificationResponse
import os
from dotenv import load_dotenv
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")


def create_global_catalog_discovery(raw_pdf_text: str) -> BankAndCardCatalog:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=KEY)
    structured_llm = llm.with_structured_output(BankAndCardCatalog, method="json_mode")

    prompt_text = """You are an elite financial data analyst.
1. Read the provided text to identify which Bank issued this document.
2. Extract a clean name (bank_name) and a safe lowercase hyphenated slug identifier (bank_id).
3. Extract all distinct credit cards mentioned in the text along with their basic fees.

Document Text:
{text}"""

    discoveries_prompt = ChatPromptTemplate.from_template(prompt_text)

    discovery_chain = discoveries_prompt | structured_llm

    print("---------------Implementing single-call processing - Stage 1 of metadata pipeline-------------")

    try:
        final_response = discovery_chain.invoke({"text": raw_pdf_text})
    except Exception as e:
        print(f"------------Critical error occured during Stage 1 : {e}-------------") 
        final_response = None

    return final_response


def build_rich_vector_documents(raw_pdf_text: str, bank_name: str, bank_id: str, global_catalog: dict) -> list[Document]:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=KEY)
    structured_llm = llm.with_structured_output(BatchClassificationResponse, method="json_mode")

    classification_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("You are an elite financial data analyst classification assistant."),
        HumanMessage(
            "Analyze the list of paragraphs from {bank_name} below. Each paragraph has a 'Paragraph Index' prefix.\n"
            "For each paragraph, classify which specific card it belongs to. You MUST select exclusively from this list of discovered card IDs: {valid_keys}.\n"
            "If a paragraph applies generally to all cards or is boilerplate legal jargon, map it to 'generic'.\n"
            "Also determine the benefit_type, spend_category, and partner_merchant for each paragraph according to their descriptions.\n"
            "Respond ONLY with the structured BatchClassificationResponse containing the classification result for each paragraph index.\n\n"
            "Paragraphs:\n{paragraphs_text}"
        )
    ])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.create_documents([raw_pdf_text])

    classification_chain = classification_prompt | structured_llm

    valid_keys = list(global_catalog.keys())
    
    # We group paragraphs into batches of 15
    batch_size = 15
    batch_inputs = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        paragraphs_text = ""
        for relative_idx, doc in enumerate(batch_docs):
            global_idx = i + relative_idx
            paragraphs_text += f"Paragraph Index {global_idx}:\n{doc.page_content}\n\n"
        
        batch_inputs.append({
            "bank_name": bank_name,
            "valid_keys": valid_keys,
            "paragraphs_text": paragraphs_text
        })

    print(f"--> Initiating Pass 2 batch pipeline across {len(batch_inputs)} API requests (batching 15 paragraphs per request)...")

    # Run the batch operations using langchain's batch method with concurrency limit
    batch_results = classification_chain.batch(batch_inputs, config={"max_concurrency": 3}, return_exceptions=True)

    final_documents = []

    for batch_idx, analysis_response in enumerate(batch_results):
        if isinstance(analysis_response, Exception):
            print(f"Warning: Batch {batch_idx} failed with error: {analysis_response}")
            continue
        if not analysis_response or not analysis_response.results:
            continue    
        
        for item in analysis_response.results:
            idx = item.paragraph_index
            if idx < 0 or idx >= len(documents):
                print(f"Warning: Model returned index {idx} out of range in batch {batch_idx}")
                continue
            
            document = documents[idx]
            matched_card = global_catalog.get(item.applicable_card_id)
            chunk_id = f"{bank_id}-{item.applicable_card_id}-{idx}"

            metadata = {
                "bank_id": bank_id,
                "bank_name": bank_name,
                "card_id": matched_card.card_id if matched_card else "generic-terms",
                "card_name": matched_card.card_name if matched_card else f"{bank_name} General Terms",
                "chunk_id": chunk_id,
                
                "joining_fee": matched_card.joining_fee if matched_card else 0.0,
                "annual_fee": matched_card.annual_fee if matched_card else 0.0,
                "fee_waiver_threshold": matched_card.fee_waiver_threshold if matched_card else 0.0,
                
                "benefit_type": item.benefit_type,
                "spend_category": item.spend_category,
                "partner_merchant": item.partner_merchant
            }

            inflated_page_content = (
                f"Bank: {metadata['bank_name']} | Card: {metadata['card_name']} | "
                f"Category: {metadata['spend_category']} | Context: {document.page_content}"
            )

            final_doc = Document(
                page_content=inflated_page_content,
                metadata=metadata
            )
            final_documents.append(final_doc)
    
    return final_documents


            
