"""
Deterministic tests for evaluation harness loading, schema validation, level contracts, and abstention reporting.
"""

import json
import sys
from pathlib import Path
import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import RetrievalEvaluator
from evaluation.evaluate import build_fixture_retriever

DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_dataset.jsonl"


def test_dataset_file_exists_and_has_records():
    assert DATASET_PATH.exists(), f"Missing dataset file at: {DATASET_PATH}"
    evaluator = RetrievalEvaluator(dataset_path=DATASET_PATH)
    assert len(evaluator.dataset) >= 10, "Evaluation dataset must contain at least 10 annotated queries"


def test_dataset_schema_and_integrity():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip() and not line.startswith("#")]

    query_ids = set()
    answerable_count = 0
    unanswerable_count = 0

    for rec in records:
        assert "query_id" in rec
        assert "question" in rec
        assert "is_answerable" in rec
        assert "category" in rec
        assert "rationale" in rec

        # Unique query IDs
        assert rec["query_id"] not in query_ids, f"Duplicate query_id: {rec['query_id']}"
        query_ids.add(rec["query_id"])

        if rec["is_answerable"]:
            answerable_count += 1
            assert len(rec.get("expected_chunk_ids", [])) > 0 or len(rec.get("expected_parent_ids", [])) > 0
            assert len(rec.get("relevant_sources", [])) > 0
        else:
            unanswerable_count += 1
            assert len(rec.get("expected_chunk_ids", [])) == 0
            assert len(rec.get("relevant_sources", [])) == 0

    assert answerable_count >= 5, "Dataset must have answerable benchmark questions"
    assert unanswerable_count >= 2, "Dataset must have unanswerable benchmark questions"


def test_evaluator_validates_target_level():
    with pytest.raises(ValueError, match="Invalid target_id_level"):
        RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="paragraph")


def test_extract_document_ids_strictly_enforces_level():
    evaluator_chunk = RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="chunk")
    evaluator_parent = RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="parent")
    evaluator_doc = RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="document")

    # Document that only has parent_id (e.g. from parent expansion)
    parent_only_doc = Document(
        page_content="Content of parent page",
        metadata={"parent_id": "doc_1_p0", "document_id": "doc_1", "source": "paper1.pdf"},
    )
    # Document that has chunk_id
    chunk_doc = Document(
        page_content="Content of chunk",
        metadata={
            "chunk_id": "doc_1_p0_c0",
            "parent_id": "doc_1_p0",
            "document_id": "doc_1",
            "source": "paper1.pdf",
        },
    )

    # Chunk level: parent_only_doc must NOT fall back to parent_id 'doc_1_p0'
    chunk_ids = evaluator_chunk.extract_document_ids([parent_only_doc, chunk_doc])
    assert chunk_ids[0] == "__unidentified_chunk_0__"
    assert chunk_ids[1] == "doc_1_p0_c0"

    # Parent level: extracts parent_id for both
    parent_ids = evaluator_parent.extract_document_ids([parent_only_doc, chunk_doc])
    assert parent_ids[0] == "doc_1_p0"
    assert parent_ids[1] == "doc_1_p0"

    # Document level: extracts document_id
    doc_ids = evaluator_doc.extract_document_ids([parent_only_doc, chunk_doc])
    assert doc_ids[0] == "doc_1"
    assert doc_ids[1] == "doc_1"


def test_evaluator_runs_deterministically_in_fixture_mode():
    evaluator = RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="parent")
    fixture_retriever = build_fixture_retriever()

    metrics = evaluator.evaluate(retriever_fn=fixture_retriever, max_k=5)

    assert metrics.total_queries == len(evaluator.dataset)
    assert metrics.num_answerable > 0
    assert metrics.num_unanswerable > 0

    # Sanity checks on aggregated scores
    assert 0.0 <= metrics.mean_recall_at_1 <= 1.0
    assert 0.0 <= metrics.mean_recall_at_3 <= 1.0
    assert 0.0 <= metrics.mean_recall_at_5 <= 1.0
    assert 0.0 <= metrics.mean_precision_at_5 <= 1.0
    assert 0.0 <= metrics.mean_mrr <= 1.0

    # In fixture mode with clean keyword mapping, all unanswerables return []
    assert metrics.unanswerable_empty_retrieval_count == metrics.num_unanswerable
    assert metrics.unanswerable_false_positive_count == 0
    assert metrics.unanswerable_handled_count == metrics.num_unanswerable

    # Markdown table generation check
    table_str = metrics.summary_table()
    assert "| **Total Evaluated Queries** |" in table_str
    assert "| **Mean Reciprocal Rank (MRR)** |" in table_str
    assert "| **Unanswerable Clean Abstentions (Empty Retrieval)** |" in table_str


def test_evaluator_unanswerable_false_positive_tracking():
    """Verifies that returning irrelevant documents for an unanswerable query is flagged as a False Positive."""
    evaluator = RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="parent")

    # Mock retriever that always returns an irrelevant document, even for out-of-corpus queries
    leaky_doc = Document(
        page_content="Irrelevant text",
        metadata={"parent_id": "irrelevant_parent_1", "document_id": "irrelevant_doc"},
    )

    def leaky_retriever(query: str, k: int):
        return [leaky_doc]

    metrics = evaluator.evaluate(retriever_fn=leaky_retriever, max_k=5)

    # All unanswerables received documents -> 0 clean abstentions, all false positives
    assert metrics.unanswerable_empty_retrieval_count == 0
    assert metrics.unanswerable_false_positive_count == metrics.num_unanswerable

    for res in metrics.per_query_results:
        if not res.is_answerable:
            assert res.is_false_positive is True
            assert res.empty_retrieval is False
            assert res.unanswerable_flagged is False


def test_evaluator_rejects_missing_dataset():
    with pytest.raises(FileNotFoundError):
        RetrievalEvaluator(dataset_path="/tmp/non_existent_dataset_path.jsonl")
