# RAGSS Retrieval Evaluation Harness

This directory contains the reproducible retrieval evaluation harness, versioned benchmark dataset, and metric calculation engine for the RAGSS hybrid retrieval pipeline.

---

## 1. Directory Structure

```
evaluation/
├── retrieval_dataset.jsonl   # Versioned ground-truth evaluation dataset (12 queries)
├── metrics.py                # Standalone IR metrics: Recall@K, MRR, Precision@K, Hit@K
├── evaluator.py              # Core harness engine: dataset loading, execution, report aggregation
├── evaluate.py               # Standalone CLI runner (fixture & live modes)
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

## 3. Metric Definitions

### Recall@K
$$\text{Recall@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{|\text{Relevant}|}$$
Fraction of annotated ground-truth target IDs retrieved in the top $K$ results. Handled safely when multiple targets exist (e.g., multi-paper queries).

### Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the 1-indexed position of the *first* relevant document returned for query $i$. Returns $0.0$ if no relevant result is retrieved.

### Unanswerable Query Handling
Unanswerable queries (`is_answerable: false`) are strictly separated from Recall and MRR computations. They do not inflate retrieval performance. The harness evaluates whether out-of-domain queries are flagged, return low confidence, or cleanly trigger downstream abstention.

---

## 4. Running Evaluation

### Fast Offline Fixture Mode (CI/CD Default)
Executes in `< 0.1` seconds without loading PyTorch weights or writing to disk:
```bash
python evaluation/evaluate.py --mode fixture
```

### Live End-to-End Retrieval Mode
Indexes the 3 sample research papers using `bge-small-en-v1.5`, BM25, and `ms-marco-MiniLM-L-6-v2` reranker:
```bash
python evaluation/evaluate.py --mode live --level parent --k 5
```

Export detailed JSON results:
```bash
python evaluation/evaluate.py --mode fixture --output-json evaluation/results.json
```
