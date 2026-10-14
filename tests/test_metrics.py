"""
Deterministic unit tests for Information Retrieval evaluation metrics.
Validates mathematical correctness against hand-calculated ground-truth examples.
"""

import pytest
from evaluation.metrics import (
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
        # Top 1 contains 'd1' -> 1/2 = 0.5
        assert compute_recall_at_k(retrieved, relevant, k=1) == 0.5
        # Top 3 contains 'd1' -> 1/2 = 0.5
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.5
        # Top 4 contains 'd1' and 'd2' -> 2/2 = 1.0
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


class TestMeanReciprocalRank:
    def test_first_result_is_relevant(self):
        # Rank 1 -> 1/1 = 1.0
        assert compute_mrr(["d1", "d2", "d3"], ["d1"]) == 1.0

    def test_second_result_is_relevant(self):
        # Rank 2 -> 1/2 = 0.5
        assert compute_mrr(["d9", "d1", "d3"], ["d1"]) == 0.5

    def test_third_result_is_relevant(self):
        # Rank 3 -> 1/3 ≈ 0.333333
        assert pytest.approx(compute_mrr(["d9", "d8", "d1"], ["d1"]), 0.0001) == 1.0 / 3.0

    def test_multiple_relevant_items_uses_first_rank(self):
        # First relevant is 'd2' at rank 2; 'd1' at rank 4 is ignored for MRR
        retrieved = ["d9", "d2", "d8", "d1"]
        relevant = ["d1", "d2"]
        assert compute_mrr(retrieved, relevant) == 0.5

    def test_no_relevant_items_returns_zero(self):
        assert compute_mrr(["d7", "d8", "d9"], ["d1", "d2"]) == 0.0

    def test_empty_lists_return_zero(self):
        assert compute_mrr([], ["d1"]) == 0.0
        assert compute_mrr(["d1"], []) == 0.0


class TestPrecisionAtK:
    def test_precision_hand_calculated(self):
        # k=4, 2 relevant out of top 4 -> 2/4 = 0.5
        retrieved = ["d1", "d9", "d2", "d8"]
        relevant = ["d1", "d2", "d3"]
        assert compute_precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_empty_returns_zero(self):
        assert compute_precision_at_k([], ["d1"], k=3) == 0.0
        assert compute_precision_at_k(["d1"], [], k=3) == 0.0


class TestHitAtK:
    def test_hit_binary_values(self):
        assert compute_hit_at_k(["d1", "d2"], ["d1"], k=1) == 1.0
        assert compute_hit_at_k(["d2", "d1"], ["d1"], k=1) == 0.0
        assert compute_hit_at_k(["d2", "d1"], ["d1"], k=2) == 1.0
        assert compute_hit_at_k(["d9", "d8"], ["d1"], k=5) == 0.0
