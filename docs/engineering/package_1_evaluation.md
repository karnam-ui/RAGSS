# Engineering Record: Package 1 — Evaluation Correctness & Test Reliability

## 1. Problem Statement
The retrieval evaluation harness in PaperTrail (`evaluation/`) contained several correctness bugs, mathematical inconsistencies, misleading claims, and architectural leaks:
1. **False Abstention Claim**: The harness claimed to evaluate out-of-domain query abstention, but unconditionally set `unanswerable_flagged = True` even when the retriever returned irrelevant documents (false positives).
2. **Mathematical Inconsistencies Across Metrics**: `compute_recall_at_k` deduplicated retrieved IDs, whereas `compute_hit_at_k` sliced the raw list before deduplication and `compute_mrr` did not deduplicate at all. This caused contradictions such as `Recall@K > 0` while `Hit@K == 0`, and distorted reciprocal rank positions.
3. **Cross-Level Target ID Pollution**: `extract_document_ids` fell back to `parent_id` or `document_id` when evaluating at `--level chunk`. This compared parent IDs with expected chunk IDs, creating artificial zero hits or ID contamination.
4. **Fixture Keyword Collisions**: `build_fixture_retriever` used an over-broad check (`"transduction" in query`) which caused `q10_unanswerable_quantum` ("qubit state transduction") to retrieve a transformer document, while still reporting "Handled cleanly: True".
5. **Incomplete JSON & CLI Serialization**: `evaluate.py` failed to export precision, hits, false positives, clean abstentions, and latency to `benchmark_results.json`.

---

## 2. Observed Existing Behavior vs. Root Cause

| Area | Existing Behavior | Root Cause |
| :--- | :--- | :--- |
| **Unanswerable Tracking** | `q10_unanswerable_quantum` retrieved `doc_1_paper1_transformers_p0`, yet CLI displayed `✓ Handled cleanly: True` and aggregate reported `Correctly Handled: 3/3`. | Hardcoded `unanswerable_flagged = True` in `evaluator.py:146` regardless of retrieved documents. |
| **Hit@K vs Recall@K** | If `retrieved = ["d1", "d1", "d2"]`, `k=2`, target `["d2"]`, `compute_recall_at_k` was `1.0`, but `compute_hit_at_k` was `0.0`. | `compute_hit_at_k` did `set(retrieved_ids[:k])` on raw slice before deduplication, whereas Recall deduplicated first. |
| **MRR Ranking** | If duplicate items appeared before the hit (e.g. `["d1", "d1", "target"]`), MRR assigned rank 3 (`0.3333`) instead of unique rank 2 (`0.5`). | `compute_mrr` iterated over raw `retrieved_ids` without deduplication or support for cutoff $K$. |
| **ID Extraction** | When evaluated at `chunk` level, missing `chunk_id` fell back to `parent_id`. | Permissive fallback in `extract_document_ids` instead of strict level enforcement. |
| **Fixture Query 10** | Fixture retriever returned `doc_1_paper1_transformers_p0_c0` for query about quantum computing. | Substring match `"transduction"` collided with "qubit state transduction". |

---

## 3. Decisions & Rationale

### A. Strict Retrieval-Only Boundary for Unanswerables
- **Decision**: Define retrieval-level unanswerable performance strictly in terms of **Clean Abstention** (retriever returned 0 documents: True Negative) vs. **Retrieval False Positive** (retriever returned documents for an out-of-corpus query).
- **Rationale**: A retrieval harness observes retrieved documents, not generated text. Claiming "abstention" when an LLM has not even been invoked is intellectually dishonest. Generation abstention must be evaluated in an end-to-end generation suite.
- **Alternatives Considered**: 
  - *Confidence thresholding*: Require retrievers to output similarity scores. Rejected for now because standard LangChain `Retriever.retrieve()` returns `List[Document]` without scores.

### B. Shared Order-Preserving Deduplication Helper
- **Decision**: Centralize all ID deduplication in `_deduplicate_preserve_order(ids, max_items=k)`.
- **Rationale**: An entity retrieved multiple times (e.g., from hybrid BM25 + dense fusion) occupies only one rank slot in standard IR evaluation. By sharing the exact same deduplication helper across Recall@K, Precision@K, Hit@K, and MRR@K, we mathematically guarantee that $\text{Hit@K} = 1.0 \iff \text{Recall@K} > 0.0$.

### C. Preserving Rank Slots for Missing IDs
- **Decision**: If a document lacks an ID for the requested `target_id_level`, assign `__unidentified_<level>_<rank>__`.
- **Rationale**: If an unidentifiable document is simply dropped from the list, subsequent valid documents are promoted up the ranking, artificially inflating Precision and MRR. By assigning an unmatchable placeholder, the document occupies its true rank position, correctly penalizing precision without polluting IDs across levels.

---

## 4. Implementation Summary

1. `evaluation/metrics.py`:
   - Added `_deduplicate_preserve_order`.
   - Updated `compute_recall_at_k`, `compute_precision_at_k`, `compute_hit_at_k`, and `compute_mrr` (with cutoff `k`).
   - Extended `QueryEvalResult` with `precision_at_k`, `empty_retrieval`, and `is_false_positive`.
   - Extended `AggregatedMetrics` with mean precision, mean hits, latency, clean abstentions, false positives, and `to_dict()`.
2. `evaluation/evaluator.py`:
   - Validated `target_id_level in ("chunk", "parent", "document")`.
   - Strict ID extraction avoiding cross-level fallback.
   - Grounded unanswerable tracking (`empty_retrieval` vs `is_false_positive`).
3. `evaluation/evaluate.py`:
   - Fixed keyword rule in `build_fixture_retriever` (`"sequence transduction"`).
   - Truthful CLI reporting (`✓ Clean abstention` vs `✗ Retrieval false positive`).
   - Comprehensive JSON payload export.
4. `pytest.ini`:
   - Added `pythonpath = .` to guarantee project root imports in all testing environments.
5. `tests/test_metrics.py` & `tests/test_eval_harness.py`:
   - 30 comprehensive, deterministic tests covering deduplication invariants, cutoff $K$, target ID levels, and false positive tracking.
6. `evaluation/README.md`:
   - Rewritten with rigorous IR metric formulas, level contracts, and retrieval vs. generation boundaries.

---

## 5. Validation Results

- **Unit & Harness Tests**: 30 passed in `0.04s`.
- **Full Non-Integration Suite**: 44 passed in `2.84s` (`pytest -m "not integration"`).
- **Fixture Evaluation**:
  - Answerable: 9 queries (Mean R@1: 0.8889, R@3: 1.0000, R@5: 1.0000, MRR: 1.0000).
  - Unanswerable: 3 queries (Clean Abstentions: 3/3, False Positives: 0/3).
  - Latency: 0.00 ms.

---

## 6. Known Limitations
- The fixture evaluation mode is an offline sanity check for deterministic CI/CD, not proof of live embedding model quality.
- The retrieval harness does not evaluate answer text fidelity or hallucination; that belongs in Package 4.

---

## 7. Interview Concepts & Questions

### Key Concepts
1. **Recall@K vs. Precision@K vs. MRR**: How each penalizes rank and entity coverage.
2. **Deduplication in IR**: Why entity deduplication must precede rank truncation.
3. **Retrieval Abstention vs. Generation Abstention**: Why measuring "abstention" in a retrieval harness without score thresholds is misleading.
4. **Rank Invariants**: Why dropping malformed documents inflates downstream metrics.

### Practice Questions
- *Q1: Why is Hit@K mathematically tied to Recall@K, and how can careless deduplication break this invariant?*
- *Q2: If a hybrid retriever returns 5 chunks from the same paper, how should document-level Precision@5 be calculated?*
- *Q3: What is the difference between an IR false positive and an LLM hallucination?*
