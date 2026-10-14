"""
Deterministic tests for evaluation harness loading, schema validation, and reporting.
"""

import json
from pathlib import Path
import pytest

from evaluation.evaluator import RetrievalEvaluator
from evaluation.evaluate import build_fixture_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def test_evaluator_runs_deterministically_in_fixture_mode():
    evaluator = RetrievalEvaluator(dataset_path=DATASET_PATH, target_id_level="chunk")
    fixture_retriever = build_fixture_retriever()

    metrics = evaluator.evaluate(retriever_fn=fixture_retriever, max_k=5)

    assert metrics.total_queries == len(evaluator.dataset)
    assert metrics.num_answerable > 0
    assert metrics.num_unanswerable > 0

    # Sanity checks on aggregated scores
    assert 0.0 <= metrics.mean_recall_at_1 <= 1.0
    assert 0.0 <= metrics.mean_recall_at_3 <= 1.0
    assert 0.0 <= metrics.mean_recall_at_5 <= 1.0
    assert 0.0 <= metrics.mean_mrr <= 1.0

    # Unanswerable queries must not be counted in answerable metrics
    assert metrics.unanswerable_handled_count == metrics.num_unanswerable

    # Markdown table generation check
    table_str = metrics.summary_table()
    assert "| **Total Evaluated Queries** |" in table_str
    assert "| **Mean Reciprocal Rank (MRR)** |" in table_str


def test_evaluator_rejects_missing_dataset():
    with pytest.raises(FileNotFoundError):
        RetrievalEvaluator(dataset_path="/tmp/non_existent_dataset_path.jsonl")
