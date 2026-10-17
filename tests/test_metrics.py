"""
Deterministic unit tests for Information Retrieval evaluation metrics.
Validates mathematical correctness, deduplication invariants, and edge cases.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    AggregatedMetrics,
    QueryEvalResult,
    compute_hit_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)


class TestRecallAtK:
    def test_perfect_single_target_hit_at_rank_1(self):
        # Retrieved: ['d1', 'd2', 'd3'], Target: ['d1']
        assert compute_recall_at_k(["d1", "d2", "d3"], ["d1"], k=1) == 1.0
        assert compute_recall_at_k(["d1", "d2", "d3"], ["d1"], k=3) == 1.0

    def test_single_target_hit_at_rank_3(self):
        # Hit occurs at position 3 (index 2)
        retrieved = ["d9", "d8", "d1", "d4"]
        relevant = ["d1"]
        assert compute_recall_at_k(retrieved, relevant, k=1) == 0.0
        assert compute_recall_at_k(retrieved, relevant, k=2) == 0.0
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_multi_target_partial_recall(self):
        # Targets: ['d1', 'd2']. Retrieved: ['d1', 'd9', 'd8', 'd2']
        retrieved = ["d1", "d9", "d8", "d2"]
        relevant = ["d1", "d2"]
        assert compute_recall_at_k(retrieved, relevant, k=1) == 0.5
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.5
        assert compute_recall_at_k(retrieved, relevant, k=4) == 1.0

    def test_empty_relevant_ids_returns_zero(self):
        assert compute_recall_at_k(["d1", "d2"], [], k=3) == 0.0

    def test_empty_retrieved_ids_returns_zero(self):
        assert compute_recall_at_k([], ["d1"], k=3) == 0.0

    def test_k_zero_or_negative_returns_zero(self):
        assert compute_recall_at_k(["d1"], ["d1"], k=0) == 0.0
        assert compute_recall_at_k(["d1"], ["d1"], k=-2) == 0.0

    def test_duplicate_retrieved_ids_deduplicated_safely(self):
        # Duplicate 'd1' shouldn't inflate recall
        retrieved = ["d1", "d1", "d1"]
        relevant = ["d1", "d2"]
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_duplicate_relevant_ids_in_ground_truth_handled(self):
        # Target set has duplicate 'd1', unique relevant target count is 1
        retrieved = ["d1", "d2"]
        relevant = ["d1", "d1"]
        assert compute_recall_at_k(retrieved, relevant, k=1) == 1.0


class TestMeanReciprocalRank:
    def test_first_result_is_relevant(self):
        assert compute_mrr(["d1", "d2", "d3"], ["d1"]) == 1.0

    def test_second_result_is_relevant(self):
        assert compute_mrr(["d9", "d1", "d3"], ["d1"]) == 0.5

    def test_third_result_is_relevant(self):
        assert pytest.approx(compute_mrr(["d9", "d8", "d1"], ["d1"]), 0.0001) == 1.0 / 3.0

    def test_multiple_relevant_items_uses_first_rank(self):
        retrieved = ["d9", "d2", "d8", "d1"]
        relevant = ["d1", "d2"]
        assert compute_mrr(retrieved, relevant) == 0.5

    def test_no_relevant_items_returns_zero(self):
        assert compute_mrr(["d7", "d8", "d9"], ["d1", "d2"]) == 0.0

    def test_empty_lists_return_zero(self):
        assert compute_mrr([], ["d1"]) == 0.0
        assert compute_mrr(["d1"], []) == 0.0

    def test_mrr_cutoff_k(self):
        # Hit is at rank 3
        retrieved = ["d9", "d8", "d1", "d4"]
        relevant = ["d1"]
        # Cutoff at k=2 ignores rank 3 hit
        assert compute_mrr(retrieved, relevant, k=2) == 0.0
        # Cutoff at k=3 includes rank 3 hit
        assert pytest.approx(compute_mrr(retrieved, relevant, k=3), 0.0001) == 1.0 / 3.0
        assert compute_mrr(retrieved, relevant, k=0) == 0.0

    def test_mrr_deduplication_preserves_unique_rank(self):
        # Duplicate 'd9' should not push 'd1' from rank 2 to rank 3
        retrieved = ["d9", "d9", "d1"]
        relevant = ["d1"]
        assert compute_mrr(retrieved, relevant) == 0.5


class TestPrecisionAtK:
    def test_precision_hand_calculated(self):
        # k=4, 2 relevant out of top 4 unique -> 2/4 = 0.5
        retrieved = ["d1", "d9", "d2", "d8"]
        relevant = ["d1", "d2", "d3"]
        assert compute_precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_empty_returns_zero(self):
        assert compute_precision_at_k([], ["d1"], k=3) == 0.0
        assert compute_precision_at_k(["d1"], [], k=3) == 0.0

    def test_precision_k_larger_than_retrieved(self):
        # Retrieved 1 doc (relevant), evaluated at k=5 -> precision is 1/5 = 0.2
        assert compute_precision_at_k(["d1"], ["d1"], k=5) == 0.2


class TestHitAtK:
    def test_hit_binary_values(self):
        assert compute_hit_at_k(["d1", "d2"], ["d1"], k=1) == 1.0
        assert compute_hit_at_k(["d2", "d1"], ["d1"], k=1) == 0.0
        assert compute_hit_at_k(["d2", "d1"], ["d1"], k=2) == 1.0
        assert compute_hit_at_k(["d9", "d8"], ["d1"], k=5) == 0.0

    def test_hit_consistent_with_recall_under_duplicates(self):
        # When duplicate IDs precede the relevant hit
        retrieved = ["d9", "d9", "d1"]
        relevant = ["d1"]
        # Top 2 unique items are ['d9', 'd1']
        recall = compute_recall_at_k(retrieved, relevant, k=2)
        hit = compute_hit_at_k(retrieved, relevant, k=2)
        assert recall == 1.0
        assert hit == 1.0  # Mathematical invariant: hit == 1.0 whenever recall > 0


class TestDataModels:
    def test_query_eval_result_to_dict(self):
        res = QueryEvalResult(
            query_id="q01",
            question="test question",
            is_answerable=True,
            expected_ids=["d1"],
            retrieved_ids=["d1"],
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            precision_at_1=1.0,
            precision_at_3=0.3333,
            precision_at_5=0.2,
            mrr=1.0,
            hit_at_1=1.0,
            hit_at_3=1.0,
            hit_at_5=1.0,
            has_any_hit=True,
            empty_retrieval=False,
            is_false_positive=False,
            unanswerable_flagged=False,
            latency_ms=1.23,
        )
        d = res.to_dict()
        assert d["query_id"] == "q01"
        assert d["recall_at_1"] == 1.0
        assert d["precision_at_5"] == 0.2
        assert d["has_any_hit"] is True

    def test_aggregated_metrics_table_and_dict(self):
        metrics = AggregatedMetrics(
            total_queries=2,
            num_answerable=1,
            num_unanswerable=1,
            mean_recall_at_1=1.0,
            mean_recall_at_3=1.0,
            mean_recall_at_5=1.0,
            mean_precision_at_1=1.0,
            mean_precision_at_3=0.3333,
            mean_precision_at_5=0.2,
            mean_hit_at_1=1.0,
            mean_hit_at_3=1.0,
            mean_hit_at_5=1.0,
            mean_mrr=1.0,
            zero_hit_query_count=0,
            zero_hit_query_ids=[],
            unanswerable_empty_retrieval_count=1,
            unanswerable_false_positive_count=0,
            mean_latency_ms=2.5,
        )
        table = metrics.summary_table()
        assert "Unanswerable Clean Abstentions (Empty Retrieval)" in table
        assert "Unanswerable Retrieval False Positives" in table

        d = metrics.to_dict()
        assert d["num_answerable"] == 1
        assert d["unanswerable_empty_retrieval_count"] == 1
        assert d["unanswerable_false_positive_count"] == 0
