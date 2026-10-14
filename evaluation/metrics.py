"""
Information Retrieval (IR) Evaluation Metrics.

Provides deterministic, standalone implementations of:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Precision@K
- Hit@K
- Unanswerable Query Abstention / False Positive Tracking
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Recall@K: fraction of annotated relevant IDs present in the top K retrieved IDs.
    
    Recall@K = |Retrieved[:K] ∩ Relevant| / |Relevant|

    Edge cases:
    - If relevant_ids is empty: returns 0.0.
    - If retrieved_ids is empty: returns 0.0.
    - k <= 0: returns 0.0.
    - Duplicate IDs in retrieved_ids: deduplicated while preserving top-K order.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    if not target_set:
        return 0.0

    # Deduplicate retrieved IDs while preserving order up to k unique items
    seen: Set[str] = set()
    top_k_unique: List[str] = []
    for doc_id in retrieved_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            top_k_unique.append(doc_id)
            if len(top_k_unique) == k:
                break

    hits = sum(1 for doc_id in top_k_unique if doc_id in target_set)
    return float(hits) / float(len(target_set))


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of the first relevant document in retrieved_ids.
    
    Rank is 1-indexed.
    Returns 0.0 if no relevant document is found in retrieved_ids or if relevant_ids is empty.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in target_set:
            return 1.0 / float(rank)

    return 0.0


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Precision@K: fraction of top K retrieved IDs that are relevant.
    
    Precision@K = |Retrieved[:K] ∩ Relevant| / K
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    # Deduplicate retrieved IDs preserving rank order
    seen: Set[str] = set()
    top_k_unique: List[str] = []
    for doc_id in retrieved_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            top_k_unique.append(doc_id)
            if len(top_k_unique) == k:
                break

    hits = sum(1 for doc_id in top_k_unique if doc_id in target_set)
    return float(hits) / float(k)


def compute_hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Hit@K (Binary): 1.0 if at least one relevant document is in top K, else 0.0.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0
    target_set: Set[str] = set(relevant_ids)
    top_k_slice = set(retrieved_ids[:k])
    return 1.0 if bool(top_k_slice.intersection(target_set)) else 0.0


@dataclass
class QueryEvalResult:
    query_id: str
    question: str
    is_answerable: bool
    expected_ids: List[str]
    retrieved_ids: List[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    has_any_hit: bool
    unanswerable_flagged: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "is_answerable": self.is_answerable,
            "expected_ids": self.expected_ids,
            "retrieved_ids": self.retrieved_ids,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "has_any_hit": self.has_any_hit,
            "unanswerable_flagged": self.unanswerable_flagged,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class AggregatedMetrics:
    total_queries: int
    num_answerable: int
    num_unanswerable: int
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_mrr: float
    zero_hit_query_count: int
    zero_hit_query_ids: List[str]
    unanswerable_handled_count: int
    per_query_results: List[QueryEvalResult] = field(default_factory=list)

    def summary_table(self) -> str:
        """Produces a clean Markdown table summarizing aggregate performance."""
        return (
            "| Metric | Value |\n"
            "| :--- | :--- |\n"
            f"| **Total Evaluated Queries** | {self.total_queries} |\n"
            f"| **Answerable Queries** | {self.num_answerable} |\n"
            f"| **Unanswerable / Out-of-Corpus Queries** | {self.num_unanswerable} |\n"
            f"| **Mean Recall@1** | {self.mean_recall_at_1:.4f} |\n"
            f"| **Mean Recall@3** | {self.mean_recall_at_3:.4f} |\n"
            f"| **Mean Recall@5** | {self.mean_recall_at_5:.4f} |\n"
            f"| **Mean Reciprocal Rank (MRR)** | {self.mean_mrr:.4f} |\n"
            f"| **Zero-Hit Queries (Miss Rate)** | {self.zero_hit_query_count}/{self.num_answerable} |\n"
            f"| **Correctly Handled Unanswerables** | {self.unanswerable_handled_count}/{self.num_unanswerable} |\n"
        )
