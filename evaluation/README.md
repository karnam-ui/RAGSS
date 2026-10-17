# RAGSS Retrieval Evaluation Harness

This directory contains the reproducible retrieval evaluation harness, versioned benchmark dataset, and metric calculation engine for the RAGSS hybrid retrieval pipeline.

---

## 1. Directory Structure

```
evaluation/
├── retrieval_dataset.jsonl   # Versioned ground-truth evaluation dataset (12 queries)
├── metrics.py                # Standalone IR metrics: Recall@K, Precision@K, Hit@K, MRR@K
├── evaluator.py              # Core harness engine: dataset loading, execution, report aggregation
├── evaluate.py               # Standalone CLI runner (fixture & live modes)
├── benchmark_results.json    # Standardized benchmark evaluation outputs
└── README.md                 # Methodology, metric definitions, and command reference
```

---

## 2. Benchmark Dataset (`retrieval_dataset.jsonl`)

The dataset is stored in standard JSONL format. Each record contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `query_id` | `str` | Stable, deterministic query identifier (e.g., `q01_trans_core`). |
| `question` | `str` | Research question formulated for retrieval. |
| `expected_chunk_ids` | `list[str]` | Ground-truth child chunk identifiers (`doc_X_paperY_pZ_cW`). |
| `expected_parent_ids` | `list[str]` | Ground-truth parent page identifiers (`doc_X_paperY_pZ`). |
| `expected_document_ids` | `list[str]` | Ground-truth document identifiers (`doc_X_paperY`). |
| `relevant_sources` | `list[str]` | Canonical source PDF file names. |
| `is_answerable` | `bool` | `true` if answerable from indexed corpus; `false` for unanswerable/out-of-domain queries. |
| `category` | `str` | Query type: `single_paper`, `specific_passage`, `multi_paper`, `confusable_terminology`, `unanswerable`. |
| `rationale` | `str` | Human annotation rationale citing exact paper section. |

### Dataset Composition (12 Queries)
- **Single-Paper Overview (3):** Core architecture & contributions (`q01_trans_core`, `q03_bert_overview`, `q05_rag_core`).
- **Specific Passage / Detail (3):** Exact mathematical definition or training objective (`q02_trans_math`, `q04_bert_mlm`, `q06_rag_hybrid`).
- **Multi-Paper Synthesis (2):** Requires retrieving evidence across two distinct papers (`q07_cross_attention_vs_mlm`, `q08_cross_trans_rag`).
- **Confusable Terminology (1):** Parametric vs. non-parametric retrieval memory (`q09_confusable_parametric`).
- **Unanswerable / Out-of-Corpus (3):** Quantum error correction, banking interest rates, transmembrane protein folding (`q10`, `q11`, `q12`).

---

## 3. Evaluation Contracts & Metric Definitions

### Target ID Levels & Rank Invariant
Evaluation can be performed at three granularity levels (`--level`):
1. **`chunk`**: Evaluates raw child chunks (`expected_chunk_ids`).
2. **`parent`**: Evaluates expanded parent documents / pages (`expected_parent_ids`). This matches the output contract of `Retriever.retrieve()`.
3. **`document`**: Evaluates whole source documents (`expected_document_ids`).

> [!IMPORTANT]
> **No Cross-Level ID Pollution**: If a retriever outputs documents lacking identifiers for the requested level (e.g., evaluating chunk level on a retriever that expands to parents without retaining chunk metadata), the harness assigns a non-matching placeholder (`__unidentified_<level>_<rank>__`). This preserves the document's rank slot to properly penalize Precision and MRR without falsely matching ground-truth IDs from other levels.

### Deduplication Invariant
In Information Retrieval evaluation, rank positions represent distinct documents. If a retriever returns duplicate entries (e.g., from hybrid rank fusion or multiple chunks resolving to the same parent), IDs are deduplicated preserving the rank of their *first* occurrence. This ensures strict mathematical consistency:
$$\text{Hit@K} = 1.0 \iff \text{Recall@K} > 0.0$$

### Metric Calculations

#### Recall@K
$$\text{Recall@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{|\text{Relevant}|}$$
Fraction of annotated ground-truth target IDs retrieved in the top $K$ unique items.

#### Precision@K
$$\text{Precision@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{K}$$
Fraction of top $K$ retrieved slots containing relevant items.

#### Hit@K
$$\text{Hit@K} = \begin{cases} 1.0 & \text{if } |\text{Retrieved}_{1..K} \cap \text{Relevant}| \ge 1 \\ 0.0 & \text{otherwise} \end{cases}$$

#### Mean Reciprocal Rank (MRR / MRR@K)
$$\text{MRR@K} = \frac{1}{|Q_{\text{ans}}|} \sum_{i=1}^{|Q_{\text{ans}}|} \text{RR}_i$$
Where $\text{RR}_i = \frac{1}{\text{rank}_i}$ for the first relevant document found within rank $K$, and $0.0$ if no relevant document appears within the top $K$.

---

## 4. Unanswerable Query Evaluation: Retrieval vs. Generation

> [!NOTE]
> **Strict Retrieval-Only Boundary**: Retrieval evaluation and answer-generation evaluation are distinct tasks. A retrieval harness observes retrieved document IDs, not generated natural-language text.

For out-of-corpus queries (`is_answerable = false`, `expected_ids = []`):
- **Clean Abstention (Empty Retrieval / True Negative)**: The retriever returned zero documents (`len(retrieved_ids) == 0`).
- **Retrieval False Positive**: The retriever returned documents despite no corpus document being relevant.

Unanswerable queries are excluded from Recall and MRR averages to prevent metric inflation. Downstream generation abstention (the LLM refusing to answer when context is absent or insufficient) is evaluated separately in the end-to-end generation evaluation suite.

---

## 5. Running Evaluation

### Fast Offline Fixture Mode (CI/CD Default)
Executes in `< 0.05` seconds without loading PyTorch weights or writing to disk:
```bash
python evaluation/evaluate.py --mode fixture
```

### Chunk-Level Fixture Evaluation
```bash
python evaluation/evaluate.py --mode fixture --level chunk
```

### Live End-to-End Retrieval Mode
Indexes the 3 sample research papers using `bge-small-en-v1.5`, BM25, and `ms-marco-MiniLM-L-6-v2` reranker:
```bash
python evaluation/evaluate.py --mode live --level parent --k 5
```

### Export Detailed JSON Results
```bash
python evaluation/evaluate.py --mode fixture --output-json evaluation/benchmark_results.json
```
