# Bugs, Root Causes, and Fixes

A running build log of real defects found in this project — what broke, why it was actually
happening (not just the surface symptom), and how it was fixed. Kept for reference and because
most of these are more interesting than the feature list: several looked like one thing on first
diagnosis and turned out to be something else entirely.

---

## Ingestion & Metadata Extraction

### 1. 100% of chunks classified as "generic-terms" — LangChain prompt-templating gotcha

**Symptom:** Every chunk ingested into Chroma (363 of them, across all 5 MITC documents) was
tagged `card_id: "generic-terms"`. Any query that filtered retrieval by a specific card
(`eligibility` intent, or `general` intent + a named card) silently retrieved zero documents.

**First diagnosis (incomplete):** Assumed the problem was that `BatchItemClassification` only
allowed a single `applicable_card_id: str` per paragraph, forcing the classifier to pick one
owner even for paragraphs that were shared table rows covering many cards at once. Fixed the
schema to `applicable_card_ids: List[str]` and added logic to emit one duplicated Document per
matched card. **This did not fix the bug** — a full rerun still produced 100% generic chunks.

**Real root cause:** The classification prompt in `build_rich_vector_documents` was built as:

```python
ChatPromptTemplate.from_messages([
    SystemMessage("..."),
    HumanMessage("...{bank_name}...{valid_keys}...{paragraphs_text}...")
])
```

`ChatPromptTemplate.from_messages()` only performs `{variable}` substitution on
`("role", "template_string")` **tuple** entries. Passing literal `SystemMessage`/`HumanMessage`
objects means LangChain does **not** substitute `{}` placeholders inside them. The LLM was
receiving the literal, unsubstituted text `{bank_name}`, `{valid_keys}`, `{paragraphs_text}` on
every single classification call, for the entire corpus — it never saw real paragraph text or
the valid card-ID list, so "generic" was the only classification it could ever produce.

**Fix:** Converted the prompt to tuple form: `[("system", "..."), ("human", "...{var}...")]`.
Verified by re-running classification directly on real IndusInd text — went from 100% generic to
correctly returning specific card_ids (`pioneer-private`, `indulge`, `crest`, etc.).

**File:** `processing/create_metadata.py`

---

### 2. Fee data silently defaulted to 0.0 instead of "unknown"

**Symptom:** The agent stated a confirmed ₹0 annual fee for premium IndusInd cards where the
source MITC document never states a fee figure at all.

**Root cause:** `CardIdentity.joining_fee` / `annual_fee` / `fee_waiver_threshold` were typed
`float = 0.0`. A field that was simply never mentioned in the document and a field that was
explicitly confirmed as ₹0 were indistinguishable — both round-tripped through the schema as
`0.0`.

**Fix:** Changed all three fields to `Optional[float] = None`, strengthened the field
descriptions with explicit "do not guess a figure that isn't written in the text" language, and
threaded a `fee_confirmed: bool` flag from the data layer through the calculator/retriever into
the synthesizer's prompt rules, instructing the model to caveat unconfirmed fees instead of
stating them as fact.

**Files:** `processing/models.py`, `agent/nodes/calculator.py`, `agent/nodes/synthesizer.py`

---

### 3. Rate-limit batch drops during Pass-2 classification

**Symptom:** Some paragraphs from large documents were silently missing from the final chunk set
with no error surfaced.

**Root cause:** `classification_chain.batch(..., return_exceptions=True)` caught 429 errors per
batch and just `continue`d past them — a transient rate-limit hit meant that batch's paragraphs
were permanently dropped, not retried.

**Fix:** Wrapped all LLM chains (`discovery_chain`, `classification_chain`, the reward-rate
extraction chain, the router's `analysis_chain`, the synthesizer's direct `llm.invoke()`) with
`.with_retry(stop_after_attempt=6, wait_exponential_jitter=True)`, and reduced
`classification_chain.batch()`'s `max_concurrency` from 3 to 2 to stay further under the
free-tier per-minute request quota.

**File:** `processing/create_metadata.py`, `agent/nodes/router.py`, `agent/nodes/synthesizer.py`

---

## Embedding & Rate Limiting

### 4. Embedding rate limit (429, per-minute)

**Symptom:** `upsert_to_vector_db` crashed partway through a large document with a 429 on
`embed_content`.

**Root cause:** `Chroma.from_documents()` embedded an entire document's chunk list in one call —
for a 200+ chunk document this single call requested far more embeddings per minute than the
free-tier `embed_content` per-minute quota allows.

**Fix:** Rewrote `upsert_to_vector_db` to batch in groups of 40 (`EMBED_BATCH_SIZE`), each batch
wrapped in a `@retry(stop_after_attempt(8), wait_exponential_jitter(initial=2, max=60))`
decorated helper, using `Chroma(...).add_texts()` per batch instead of one `from_documents()` call.

**File:** `embed/embed_and_upsert.py`

### 5. Embedding daily quota exhaustion (operational constraint, not a code bug)

**Symptom:** After the per-minute fix, a full ingestion rerun succeeded on 2 of 5 files then hit
Gemini's free-tier **daily** `embed_content` cap (1000 requests/day) — not fixable by
retry/backoff since it's a full-day cap, not a transient limit. Recurred again later the same
day during reward-PDF ingestion and during eval harness runs (each Chroma query is itself an
embedding call).

**Resolution:** No code fix applies here — waited for the daily quota to reset and resumed.
Ingestion (`main.py`) and the eval harness are both idempotent/cache-aware, so a blocked run can
always be safely rerun without reprocessing already-completed work.

---

## Retrieval & Hallucination Guardrail

### 6. `comparison` intent never triggered card-specific retrieval filtering

**Symptom:** A multi-turn follow-up like *"Which one has the lower annual fee?"* (referring back
to two cards named in the previous turn) retrieved completely unrelated cards. The router
correctly resolved `selected_card_id` from conversation history, but the answer used fee figures
from cards that were never discussed.

**Root cause:** `retriever_node`'s gate —
`wants_card_specific = bool(selected_card_id and intent in ["general", "eligibility"])` — never
included `"comparison"`. A comparison follow-up with no card names in the literal query text fell
straight through to a blind unfiltered similarity search, which has nothing to anchor on and
pulls in semantically-adjacent but wrong documents.

**Fix:** Added `"comparison"` to the intent list gating `wants_card_specific`.

**File:** `agent/nodes/retriever.py`

**How it was found:** the eval harness (see below) — a multi-turn test case failed, tracing the
retriever log for that turn showed it ran an unfiltered search despite a resolved card_id.

### 7. `resolve_card_ids` stripped tier words that were the card's only distinguishing feature

**Symptom:** Even after fix #6, filtered retrieval for "SBI Card PRIME" style queries could still
match dozens of unrelated SBI-family cards instead of the specific one.

**Root cause:** `resolve_card_ids`'s fuzzy-matching fallback stripped tokens like `"elite"`,
`"prime"`, `"select"`, `"gold"` as generic bank/qualifier noise before token-matching. For cards
where the *entire* differentiator is the tier name (plain "SBI Card ELITE" vs "SBI Card PRIME",
no co-brand prefix), stripping that word left only `"sbi"` as the match token — which matches
essentially every SBI-issued card, co-branded or not.

**Fix:** Added an exact-match short-circuit: since the router is already constrained to only
return slugs from the real valid-card-id list, `selected_card_id` is almost always already an
exact match — check for that first and only fall back to fuzzy token matching if it isn't found.

**File:** `agent/nodes/retriever.py`

---

## Calculator / Reward Optimization

### 8. Calculator's `covered_banks` never matched `all_banks` (bank_id vs bank_name mismatch)

**Symptom:** A spend-optimization query correctly ranked IndusInd cards (Celesta, Crest, Club
Vistara Explorer) at the top, while simultaneously reporting "IndusInd Bank — excluded from
ranking" as an uncovered bank in the same response. Contradictory output.

**Root cause:** Two bugs compounding:
1. `covered_banks` was populated from `card_meta.get("bank_id")` (a slug like `"indusind-bank"`),
   while `all_banks` was populated from `catalog.get("bank_name")` (a display string like
   `"IndusInd Bank"`). The two sets used different namespaces and could never intersect.
2. Even after aligning the field name, `bank_name` doesn't exist on individual card dicts inside
   a catalog's `cards_found` list — it only exists on the parent catalog object. The lookup was
   always `None` regardless of which field name was used.

**Fix:** When building the flat `discovered_cards` lookup, stamp each card with its parent
catalog's `bank_name` at construction time (`{**card, "bank_name": bank_name}`), then use that
consistently for both `all_banks` and `covered_banks`.

**File:** `agent/nodes/calculator.py`

**How it was found:** manually testing a spend-optimization query and noticing the ranked cards
and the "uncovered" disclaimer contradicted each other.

---

## Data Consistency — Cross-Document Card Identity

### 9. Cross-document card_id slug drift (reward-program PDF vs. MITC catalog)

**Symptom:** After ingesting Axis Bank's EDGE Rewards PDF, 6 of its 9 extracted reward profiles
came back with `bank = UNKNOWN` when cross-referenced against the already-ingested Axis MITC
catalog — even though the cards themselves (Magnus, Reserve, Burgundy Private, etc.) were already
present in that catalog under different slugs (`magnus-credit-card` in the MITC catalog vs.
`axis-bank-magnus-credit-card` from the independently-run reward-doc discovery pass).

**Root cause:** `run_reward_ingestion_pipeline` ran `create_global_catalog_discovery` fresh on
each reward document's own text, with no memory of card_ids already discovered for that bank
from a prior MITC ingestion. LLM-generated slugs aren't guaranteed to converge across two
different documents describing the same card — a document that spells out "Axis Bank Magnus
Credit Card" in full versus one that just says "Magnus Credit Card" can slugify differently even
though it's the same physical card.

**Fix:** Before running reward-rate extraction, check whether `discovered_catalogs` already has
an entry for the same `bank_id`; if so, reuse that catalog's existing card_id list as the
`valid_card_ids` constraint instead of the freshly-discovered one. Falls back to fresh discovery
only for genuinely new banks.

**File:** `main.py` (`run_reward_ingestion_pipeline`)

**Residual limitation:** even with the valid-id list passed explicitly, 1 of 23 total reward
profiles still didn't match (`axis-bank-rewards-credit-card` vs. the catalog's
`rewards-credit-card`) — structured-output constraints bias the LLM toward compliance but don't
guarantee it. Left as a documented, gracefully-degrading edge case (the card just doesn't get
fee/name cross-referenced; it isn't misattributed to the wrong bank).

### 10. Same root cause, pre-existing in the manually-authored IndusInd seed file

**Symptom:** 3 of 10 IndusInd reward profiles (`indulge`, `club-vistara-explorer`,
`pioneer-private`) resolved to `bank = UNKNOWN`, inherited from a prior session's manual
migration of a hardcoded dict into `data/reward_seeds/indusind.json`.

**Root cause:** Same slug-drift problem as #9, just introduced by hand instead of by an LLM —
the actual MITC-discovered card_ids are `indulge-credit-card`, `club-vistara-explorer-credit-card`,
`pioneer-private-credit-card`.

**Fix:** Corrected the 3 card_ids in the seed file directly.

**File:** `data/reward_seeds/indusind.json`

---

## Known Issue — Not Yet Fixed

### 11. Card-identity duplication across MITC documents for the same bank

**Symptom:** SBI's AURUM card exists as two entirely separate catalog entries:
`sbi-card-aurum` (annual_fee ₹9,999, waiver not recorded) from one MITC file, and `aurum`
(annual_fee ₹9,999, waiver ₹12,00,000) from a second MITC file — same physical card, same fee,
different waiver figure recorded, two disconnected identities.

**Root cause:** Same class of problem as #9/#10 — Pass 1 catalog discovery runs independently per
file with no cross-file card registry, and this time the *card_name itself* also differs
("SBI Card AURUM" vs. the abbreviated "AURUM"), so even name-based dedup doesn't catch it for
free.

**Scope check performed:** ran a fuzzy substring match across all 213 discovered (bank, card_name)
pairs looking for other instances. Found 45 "candidate" matches, but nearly all were false
positives — legitimately distinct co-branded variants that share a base name (e.g. "SBI Card
ELITE" vs. "IndiGo SBI Card ELITE" are different real products, not duplicates). Only the AURUM
case was manually confirmed as a genuine duplicate identity so far. Not yet a widespread problem,
but nothing currently prevents or detects it, and it will resurface as more documents get added
for banks already in the system.

**Fix:** Rather than rewriting/re-tagging existing Chroma chunks (expensive, and would require
re-embedding — a cost this project can't always afford given the free-tier daily quota), added a
non-destructive **alias-resolution layer**:

1. A new LLM reconciliation pass, `reconcile_duplicate_cards` (`processing/create_metadata.py`),
   given every card discovered for one bank across all its source documents, identifies genuine
   duplicates and outputs `{canonical_card_id, alias_card_ids, reasoning}` groups. Deliberately
   LLM-based rather than a string-similarity heuristic — the earlier scope-check already showed a
   naive substring match produces far more false positives (distinct co-branded variants sharing a
   base name) than true positives, and merging two genuinely different cards would be worse than
   leaving them unmerged. The LLM is explicitly instructed to only merge when confident, using
   annual_fee agreement as a corroborating signal.
2. Results are stored as a `card_aliases: {alias_id: canonical_id}` map in the ingestion cache
   (`processing/ingestion_cache.py`), plus a `reconciled_banks` fingerprint (the sorted list of
   source files that contributed to a bank) so reconciliation only reruns when a bank's document
   set actually changes.
3. Wired into `main.py` as `run_card_reconciliation()`, running after MITC ingestion, once per bank
   that has 2+ contributing source documents (single-document banks have nothing to reconcile, so
   are skipped entirely — no wasted LLM calls).
4. Applied downstream at query time rather than at ingestion time:
   - `retriever.py`'s `resolve_card_ids` expands an exact-matched card_id to its full alias group
     (`get_card_alias_group`) before building the Chroma metadata filter, so a query resolves to
     both `aurum` and `sbi-card-aurum` regardless of which identity the router happened to pick or
     which identity a given chunk was tagged with at ingestion time.
   - `calculator.py` canonicalizes a reward profile's card_id (`get_canonical_card_id`) before
     looking up its fee/name metadata, so a reward profile authored against one identity still
     resolves correctly against MITC fee data recorded under the other.

**Verified:** ran the reconciliation pass — it correctly identified `aurum` → `sbi-card-aurum`
("AURUM is a clear abbreviation for SBI Card AURUM and they share the same annual fee") and
correctly did *not* merge any of the ~115 other SBI cards despite many sharing similar tier names.
Confirmed `resolve_card_ids("aurum", ...)` and `resolve_card_ids("sbi-card-aurum", ...)` both now
return `['aurum', 'sbi-card-aurum']` symmetrically, while unrelated cards like `sbi-card-elite`
are returned unexpanded, unaffected by the change. A full live Chroma query confirmation is
pending the daily embedding quota reset (still exhausted as of this fix), but the alias-resolution
logic itself needs no API calls and is fully verified.

**Files:** `processing/models.py` (`CardMergeGroup`, `CardReconciliationResponse`),
`processing/create_metadata.py` (`reconcile_duplicate_cards`), `processing/ingestion_cache.py`
(`card_aliases` cache section + `get_canonical_card_id`/`get_card_alias_group`/`record_merge_groups`/
`is_bank_reconciled`/`mark_bank_reconciled`), `main.py` (`run_card_reconciliation`),
`agent/nodes/retriever.py`, `agent/nodes/calculator.py`
