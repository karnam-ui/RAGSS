"""
Information Retrieval (IR) Evaluation Metrics.

Provides deterministic, mathematically consistent implementations of:
- Recall@K
- Precision@K
- Hit@K
- Mean Reciprocal Rank (MRR / MRR@K)
- Retrieval False Positive & Empty Retrieval (Clean Abstention) Tracking
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


def _deduplicate_preserve_order(ids: List[str], max_items: Optional[int] = None) -> List[str]:
    """
    Deduplicates a list of ID strings while strictly preserving initial appearance rank.
    If max_items is specified and > 0, stops after accumulating max_items unique entries.
    """
    if max_items is not None and max_items <= 0:
        return []

    seen: Set[str] = set()
    unique_items: List[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
            if max_items is not None and len(unique_items) == max_items:
                break
    return unique_items


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Recall@K: fraction of annotated relevant IDs present in the top K retrieved IDs.
    
    Recall@K = |UniqueRetrieved[:K] ∩ Relevant| / |Relevant|

    Edge cases:
    - If relevant_ids is empty: returns 0.0.
    - If retrieved_ids is empty: returns 0.0.
    - If k <= 0: returns 0.0.
    - Duplicate IDs in retrieved_ids: deduplicated while preserving top-K unique order.
    - Duplicate IDs in relevant_ids: handled via set conversion (unique relevant targets).
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    if not target_set:
        return 0.0

    top_k_unique = _deduplicate_preserve_order(retrieved_ids, max_items=k)
    hits = sum(1 for doc_id in top_k_unique if doc_id in target_set)
    return float(hits) / float(len(target_set))


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Precision@K: fraction of top K retrieved slots that are relevant.
    
    Precision@K = |UniqueRetrieved[:K] ∩ Relevant| / K

    Edge cases:
    - If relevant_ids is empty: returns 0.0.
    - If retrieved_ids is empty: returns 0.0.
    - If k <= 0: returns 0.0.
    - Denominator is fixed at k per standard IR evaluation definition.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    if not target_set:
        return 0.0

    top_k_unique = _deduplicate_preserve_order(retrieved_ids, max_items=k)
    hits = sum(1 for doc_id in top_k_unique if doc_id in target_set)
    return float(hits) / float(k)


def compute_hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Hit@K (Binary): 1.0 if at least one relevant document is in top K unique results, else 0.0.
    
    Guaranteed invariant: compute_hit_at_k(r, rel, k) == 1.0 iff compute_recall_at_k(r, rel, k) > 0.0.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    if not target_set:
        return 0.0

    top_k_unique = _deduplicate_preserve_order(retrieved_ids, max_items=k)
    return 1.0 if any(doc_id in target_set for doc_id in top_k_unique) else 0.0


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str], k: Optional[int] = None) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of the first relevant document in retrieved_ids.
    
    Rank is 1-indexed over unique retrieved entities.
    If k is specified, evaluates MRR@K (returns 0.0 if first hit is beyond rank k).
    Returns 0.0 if no relevant document is found in retrieved_ids or if relevant_ids is empty.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    if k is not None and k <= 0:
        return 0.0

    target_set: Set[str] = set(relevant_ids)
    if not target_set:
        return 0.0

    unique_retrieved = _deduplicate_preserve_order(retrieved_ids, max_items=k)
    for rank, doc_id in enumerate(unique_retrieved, start=1):
        if doc_id in target_set:
            return 1.0 / float(rank)

    return 0.0


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
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    has_any_hit: bool = False
    empty_retrieval: bool = False
    is_false_positive: bool = False
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
            "precision_at_1": round(self.precision_at_1, 4),
            "precision_at_3": round(self.precision_at_3, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "mrr": round(self.mrr, 4),
            "has_any_hit": self.has_any_hit,
            "empty_retrieval": self.empty_retrieval,
            "is_false_positive": self.is_false_positive,
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
    mean_precision_at_1: float
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float
    mean_mrr: float
    zero_hit_query_count: int
    zero_hit_query_ids: List[str]
    unanswerable_empty_retrieval_count: int
    unanswerable_false_positive_count: int
    mean_latency_ms: float = 0.0
    unanswerable_handled_count: int = 0
    per_query_results: List[QueryEvalResult] = field(default_factory=list)

    def __post_init__(self):
        # Synchronize backward-compatible unanswerable_handled_count
        if self.unanswerable_handled_count == 0 and self.unanswerable_empty_retrieval_count > 0:
            self.unanswerable_handled_count = self.unanswerable_empty_retrieval_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "num_answerable": self.num_answerable,
            "num_unanswerable": self.num_unanswerable,
            "mean_recall_at_1": round(self.mean_recall_at_1, 4),
            "mean_recall_at_3": round(self.mean_recall_at_3, 4),
            "mean_recall_at_5": round(self.mean_recall_at_5, 4),
            "mean_precision_at_1": round(self.mean_precision_at_1, 4),
            "mean_precision_at_3": round(self.mean_precision_at_3, 4),
            "mean_precision_at_5": round(self.mean_precision_at_5, 4),
            "mean_hit_at_1": round(self.mean_hit_at_1, 4),
            "mean_hit_at_3": round(self.mean_hit_at_3, 4),
            "mean_hit_at_5": round(self.mean_hit_at_5, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "zero_hit_query_count": self.zero_hit_query_count,
            "zero_hit_query_ids": self.zero_hit_query_ids,
            "unanswerable_empty_retrieval_count": self.unanswerable_empty_retrieval_count,
            "unanswerable_false_positive_count": self.unanswerable_false_positive_count,
            "unanswerable_handled_count": self.unanswerable_handled_count,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "results": [r.to_dict() for r in self.per_query_results],
        }

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
            f"| **Mean Precision@1** | {self.mean_precision_at_1:.4f} |\n"
            f"| **Mean Precision@3** | {self.mean_precision_at_3:.4f} |\n"
            f"| **Mean Precision@5** | {self.mean_precision_at_5:.4f} |\n"
            f"| **Mean Hit@1** | {self.mean_hit_at_1:.4f} |\n"
            f"| **Mean Hit@3** | {self.mean_hit_at_3:.4f} |\n"
            f"| **Mean Hit@5** | {self.mean_hit_at_5:.4f} |\n"
            f"| **Mean Reciprocal Rank (MRR)** | {self.mean_mrr:.4f} |\n"
            f"| **Zero-Hit Queries (Miss Rate)** | {self.zero_hit_query_count}/{self.num_answerable} |\n"
            f"| **Unanswerable Clean Abstentions (Empty Retrieval)** | {self.unanswerable_empty_retrieval_count}/{self.num_unanswerable} |\n"
            f"| **Unanswerable Retrieval False Positives** | {self.unanswerable_false_positive_count}/{self.num_unanswerable} |\n"
            f"| **Mean Retrieval Latency** | {self.mean_latency_ms:.2f} ms |\n"
        )
