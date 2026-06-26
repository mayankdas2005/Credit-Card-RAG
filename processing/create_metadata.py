from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from processing.models import LocalCardDiscoveries, ChunkFeatureExtractor
import os

KEY = os.environ["GEMINI_API_KEY"]


def create_global_catalog_discovery(raw_pdf_text: str, bank_name: str, bank_id: str) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro", api_key=KEY)
    structured_llm = llm.with_structured_output(LocalCardDiscoveries)

    text_splitter = TokenTextSplitter(chunk_size=4000, chunk_overlap=200)
    documents = text_splitter.create_documents([raw_pdf_text])

    discoveries_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("You are a financial data auditor. Analyze this document slice from {bank_name} and extract structural details for any credit card explicitly named here."),
        HumanMessage("{text}")
    ])

    discovery_chain = discoveries_prompt | structured_llm

    global_catalog = {}
    for doc in documents:
        try:
            result = discovery_chain.invoke({"bank_name": bank_name, "text":doc.page_content})

            for card in result.cards_found:
                if card.card_id not in global_catalog:
                    global_catalog[card.card_id] = card
                else:
                    existing = global_catalog[card.card_id]
                    if card.joining_fee > 0: 
                        existing.joining_fee = card.joining_fee
                    if card.annual_fee > 0: 
                        existing.annual_fee = card.annual_fee
                    if card.fee_waiver_threshold > 0: 
                        existing.fee_waiver_threshold = card.fee_waiver_threshold
        except Exception as e:
            print(f"Some error occured {e}, skipping this document")
            continue
    return global_catalog


def build_rich_vector_documents(raw_pdf_text: str, bank_name: str, bank_id: str, global_catalog: dict) -> list[Document]:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=KEY)
    structured_llm = llm.with_structured_output(ChunkFeatureExtractor)

    classification_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "Analyze the following paragraph text from {bank_name}.\n"
            "Classify which specific card it belongs to. You MUST select exclusively from this list of discovered card IDs: {valid_keys}.\n"
            "If the text applies generally to all cards or is boilerplate legal jargon, reply with 'generic'."
        ),
        HumanMessage("{paragraph}")
    ])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.create_documents([raw_pdf_text])

    classification_chain = classification_prompt | structured_llm

    valid_keys = list(global_catalog.keys())
    final_documents = []

    for idx, doc in documents:
        if len(doc) < 40:
            continue
            
        try:
            result = classification_chain.invoke(
                {
                "bank_name":bank_name, 
                "valid_keys":valid_keys, 
                "paragraph": doc.page_content 
                }
            )
        except Exception:
            continue

        matched_card = global_catalog.get(result.applicable_card_id)

        metadata = {
            "bank_id": bank_id,
            "bank_name": bank_name,
            "card_id": matched_card.card_id if matched_card else "generic-terms",
            "card_name": matched_card.card_name if matched_card else f"{bank_name} General Terms",
            "chunk_id": f"{bank_id}-{result.applicable_card_id}-{idx}",
            
            "joining_fee": matched_card.joining_fee if matched_card else 0.0,
            "annual_fee": matched_card.annual_fee if matched_card else 0.0,
            "fee_waiver_threshold": matched_card.fee_waiver_threshold if matched_card else 0.0,
            
            "benefit_type": result.benefit_type,
            "spend_category": result.spend_category,
            "partner_merchant": result.partner_merchant
        }

        inflated_page_content = (
            f"Bank: {metadata['bank_name']} | Card: {metadata['card_name']} | "
            f"Category: {metadata['spend_category']} | Context: {doc.page_content}"
        )

        final_doc = Document(
            page_content=inflated_page_content,
            metadata=metadata
        )
        final_documents.append(final_doc)
    
    return final_documents

            

