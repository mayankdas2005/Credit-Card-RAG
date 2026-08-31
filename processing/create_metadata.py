from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from processing.models import BankAndCardCatalog, BatchClassificationResponse, RewardCatalogResponse, CardReconciliationResponse
import os
from dotenv import load_dotenv


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
3. Extract all distinct credit cards mentioned in the text.
4. For each card, only report joining_fee / annual_fee / fee_waiver_threshold if an explicit numeric
   figure for that specific fee appears in the text. Many "Most Important Terms & Conditions" documents
   cover interest rates, late fees, and forex mark-up WITHOUT ever stating joining/annual fees — in that
   case, leave those fields null for every card rather than defaulting to 0. Never guess or infer a fee
   figure that is not explicitly written in the text.

Document Text:
{text}"""

    discoveries_prompt = ChatPromptTemplate.from_template(prompt_text)

    # Free-tier Gemini quota is easy to exceed with these batched pipelines; retry with backoff
    # instead of silently dropping the call and losing data.
    discovery_chain = (discoveries_prompt | structured_llm).with_retry(
        stop_after_attempt=6, wait_exponential_jitter=True
    )

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

    # NOTE: must use ("role", "template string") tuples here, NOT literal SystemMessage/HumanMessage
    # objects — ChatPromptTemplate.from_messages() only performs {variable} substitution on tuple-style
    # entries. Passing raw Message objects (as this code did before) sends the literal, unsubstituted
    # "{bank_name}"/"{valid_keys}"/"{paragraphs_text}" text to the LLM, which is the actual reason every
    # paragraph in the entire corpus was classified as "generic" — the model never saw real content.
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite financial data analyst classification assistant."),
        ("human",
            "Analyze the list of paragraphs from {bank_name} below. Each paragraph has a 'Paragraph Index' prefix.\n"
            "For each paragraph, list EVERY specific card it belongs to. You MUST select exclusively from this list of "
            "discovered card IDs: {valid_keys}. Many paragraphs are shared table rows or clauses covering several cards "
            "at once (e.g. a single interest-rate row listing 6 card names) — list all of them, not just one.\n"
            "Only map a paragraph to ['generic'] if it is truly bank-wide boilerplate or applies uniformly to literally all cards.\n"
            "Also determine the benefit_type, spend_category, and partner_merchant for each paragraph according to their descriptions.\n"
            "Respond ONLY with the structured BatchClassificationResponse containing the classification result for each paragraph index.\n\n"
            "Paragraphs:\n{paragraphs_text}"
        )
    ])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.create_documents([raw_pdf_text])

    classification_chain = (classification_prompt | structured_llm).with_retry(
        stop_after_attempt=6, wait_exponential_jitter=True
    )

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

    # Low concurrency + retry-with-backoff (above) keeps us under the free-tier per-minute request quota
    # instead of bursting into 429s and silently dropping batches.
    batch_results = classification_chain.batch(batch_inputs, config={"max_concurrency": 2}, return_exceptions=True)

    final_documents = []

    # Safety valve: if a single paragraph gets tagged to more cards than this, it's almost certainly
    # a bank-wide clause rather than a genuine per-card table row — fall back to one generic doc
    # instead of duplicating the chunk dozens of times. Not tied to any specific bank's card count.
    MAX_CARD_FANOUT = 15

    def make_document(document, card_id, card_name, matched_card, item, idx):
        chunk_id = f"{bank_id}-{card_id}-{idx}"
        metadata = {
            "bank_id": bank_id,
            "bank_name": bank_name,
            "card_id": card_id,
            "card_name": card_name,
            "chunk_id": chunk_id,

            # Chroma metadata rejects None, so unknown fees are stored as the string "unknown"
            # rather than 0.0 (which would falsely read as a confirmed fee-waived card).
            "joining_fee": matched_card.joining_fee if matched_card and matched_card.joining_fee is not None else "unknown",
            "annual_fee": matched_card.annual_fee if matched_card and matched_card.annual_fee is not None else "unknown",
            "fee_waiver_threshold": matched_card.fee_waiver_threshold if matched_card and matched_card.fee_waiver_threshold is not None else "unknown",

            "benefit_type": item.benefit_type,
            "spend_category": item.spend_category,
            "partner_merchant": item.partner_merchant
        }

        inflated_page_content = (
            f"Bank: {bank_name} | Card: {card_name} | "
            f"Category: {metadata['spend_category']} | Context: {document.page_content}"
        )

        return Document(page_content=inflated_page_content, metadata=metadata)

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
            card_ids = [cid for cid in (item.applicable_card_ids or []) if cid and cid != "generic"]

            if not card_ids or len(card_ids) > MAX_CARD_FANOUT:
                final_documents.append(
                    make_document(document, "generic-terms", f"{bank_name} General Terms", None, item, idx)
                )
                continue

            matched_any = False
            for card_id in card_ids:
                matched_card = global_catalog.get(card_id)
                if not matched_card:
                    continue
                matched_any = True
                final_documents.append(
                    make_document(document, matched_card.card_id, matched_card.card_name, matched_card, item, idx)
                )

            if not matched_any:
                # None of the model's card_ids matched the known catalog — don't drop the paragraph.
                final_documents.append(
                    make_document(document, "generic-terms", f"{bank_name} General Terms", None, item, idx)
                )

    return final_documents


def create_reward_rate_catalog(raw_text: str, valid_card_ids: list[str]) -> RewardCatalogResponse:
    """Generic, bank-agnostic extraction pass for a rewards-program document. Mirrors the shape of
    create_global_catalog_discovery: one LLM call per document, driven entirely by that document's own
    dynamically-discovered card_ids — no bank or card names hardcoded here."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=KEY)
    structured_llm = llm.with_structured_output(RewardCatalogResponse, method="json_mode")

    prompt_text = """You are an elite financial data analyst extracting reward-earning structures from a
credit card rewards-program document.

For each card, ONLY report a RewardRateProfile if the document explicitly states numeric reward-earning
figures for that card (e.g. "2 points per Rs.100 on dining", "5% cashback on fuel", "4 miles per Rs.150").
Do not guess or infer rates that are not explicitly written in the text. If a card is mentioned but no
earn-rate figures are given for it anywhere in the text, omit it entirely rather than inventing a profile.

You MUST use card_id values exclusively from this list of valid card IDs for this bank: {valid_keys}

Document Text:
{text}"""

    discoveries_prompt = ChatPromptTemplate.from_template(prompt_text)
    discovery_chain = (discoveries_prompt | structured_llm).with_retry(
        stop_after_attempt=6, wait_exponential_jitter=True
    )

    print("---------------Implementing reward-rate extraction pass-------------")

    try:
        final_response = discovery_chain.invoke({"text": raw_text, "valid_keys": valid_card_ids})
    except Exception as e:
        print(f"------------Critical error occured during reward-rate extraction: {e}-------------")
        final_response = None

    return final_response


def reconcile_duplicate_cards(bank_name: str, cards: list[dict]) -> CardReconciliationResponse:
    """Given every card discovered for one bank across (possibly multiple) source documents, identify
    genuine duplicates — the same real-world product that was independently slugged/named differently
    by two separate ingestion runs — versus distinct products that merely share a similar name (e.g.
    different co-branded tiers). One LLM call per bank; no bank/card names hardcoded here.

    This exists because Pass 1 catalog discovery runs independently per document with no shared card
    registry, so the same physical card can end up with two disconnected card_id identities if it's
    described (or abbreviated) slightly differently across two documents for the same bank."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=KEY)
    structured_llm = llm.with_structured_output(CardReconciliationResponse, method="json_mode")

    cards_listing = "\n".join(
        f"- card_id={c['card_id']!r}, card_name={c['card_name']!r}, "
        f"annual_fee={c.get('annual_fee')}, source_file={c.get('source_file')!r}"
        for c in cards
    )

    prompt_text = """You are auditing a credit card catalog for {bank_name} that was assembled from
multiple source documents processed independently. Because each document was processed separately,
the SAME real-world card product may appear more than once under different card_id/card_name values
(for example, one document might abbreviate "SBI Card AURUM" as just "AURUM").

Your task: identify groups of entries below that refer to the EXACT same real-world card product.

Do NOT merge cards that are genuinely different products or tiers, even if their names look similar —
for example "SBI Card ELITE" and "IndiGo SBI Card ELITE" are different co-branded products and must
NOT be merged, nor should "SBI Card MILES" and "SBI Card MILES ELITE". Only merge when you are
confident it's the same product — matching or compatible annual_fee figures are a good corroborating
signal, but a name that is a strict abbreviation/subset of another entry's name is the main signal.
If you are not highly confident two entries are the same product, leave them unmerged.

For each confirmed duplicate group, pick the more complete/descriptive card_id as canonical.

Cards:
{cards_listing}"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    reconciliation_chain = (prompt | structured_llm).with_retry(
        stop_after_attempt=6, wait_exponential_jitter=True
    )

    print(f"---------------Implementing card-identity reconciliation pass for {bank_name}-------------")

    try:
        final_response = reconciliation_chain.invoke({"bank_name": bank_name, "cards_listing": cards_listing})
    except Exception as e:
        print(f"------------Critical error occured during card reconciliation: {e}-------------")
        final_response = None

    return final_response or CardReconciliationResponse(merge_groups=[])
