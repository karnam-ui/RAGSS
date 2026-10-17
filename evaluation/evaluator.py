"""
Retrieval Evaluation Harness Engine.

Responsible for:
- Loading and validating retrieval datasets
- Invoking retrievers (live or fixture) with latency tracking
- Extracting target IDs according to strict level contracts (chunk, parent, document)
- Computing per-query and aggregate metrics (Recall, Precision, Hit, MRR)
- Tracking unanswerable query retrieval false positives vs clean abstentions
- Producing reproducible evaluation reports
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.documents import Document

from evaluation.metrics import (
    AggregatedMetrics,
    QueryEvalResult,
    compute_hit_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)

VALID_TARGET_LEVELS = ("chunk", "parent", "document")


class RetrievalEvaluator:
    def __init__(
        self,
        dataset_path: Union[str, Path],
        target_id_level: str = "parent",
    ):
        if target_id_level not in VALID_TARGET_LEVELS:
            raise ValueError(
                f"Invalid target_id_level '{target_id_level}'. Must be one of {VALID_TARGET_LEVELS}."
            )
        self.dataset_path = Path(dataset_path)
        self.target_id_level = target_id_level
        self.dataset: List[Dict[str, Any]] = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {self.dataset_path}")

        records: List[Dict[str, Any]] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                try:
                    record = json.loads(clean_line)
                    self._validate_record(record, line_idx)
                    records.append(record)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_idx} in {self.dataset_path}: {e}")

        if not records:
            raise ValueError(f"Evaluation dataset at {self.dataset_path} contains no valid records.")
        return records

    @staticmethod
    def _validate_record(record: Dict[str, Any], line_idx: int) -> None:
        required_fields = ["query_id", "question", "is_answerable"]
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Record at line {line_idx} missing required field '{field}'")

        if record["is_answerable"]:
            has_targets = (
                bool(record.get("expected_chunk_ids"))
                or bool(record.get("expected_parent_ids"))
                or bool(record.get("expected_document_ids"))
            )
            if not has_targets:
                raise ValueError(
                    f"Answerable record at line {line_idx} ({record.get('query_id')}) has no target IDs specified."
                )

    def extract_document_ids(self, docs: List[Document]) -> List[str]:
        """
        Extracts identifier strings from retrieved Document list strictly based on target_id_level.
        
        If a retrieved Document lacks an ID at the requested target_id_level, a distinct
        placeholder is assigned so that:
        1. The document's rank position in the retrieved list is preserved.
        2. Downstream rank-based metrics (Precision@K, MRR) penalize the missing/invalid entity.
        3. No cross-level ID pollution occurs (e.g., comparing parent_id to expected_chunk_ids).
        """
        ids: List[str] = []
        for idx, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            val: Optional[str] = None

            if self.target_id_level == "chunk":
                val = meta.get("chunk_id")
            elif self.target_id_level == "parent":
                val = meta.get("parent_id")
            elif self.target_id_level == "document":
                val = meta.get("document_id") or meta.get("source")

            if val:
                ids.append(str(val))
            else:
                ids.append(f"__unidentified_{self.target_id_level}_{idx}__")

        return ids

    def evaluate(
        self,
        retriever_fn: Callable[[str, int], List[Document]],
        k_values: List[int] = (1, 3, 5),
        max_k: int = 5,
    ) -> AggregatedMetrics:
        """
        Runs evaluation across all dataset items.
        retriever_fn: Callable taking (query_string, k) -> List[Document].
        """
        per_query_results: List[QueryEvalResult] = []

        for record in self.dataset:
            query_id = record["query_id"]
            question = record["question"]
            is_answerable = record["is_answerable"]

            # Select ground-truth target IDs corresponding to target_id_level
            if self.target_id_level == "chunk":
                expected_ids = record.get("expected_chunk_ids") or []
            elif self.target_id_level == "parent":
                expected_ids = record.get("expected_parent_ids") or []
            else:
                expected_ids = record.get("expected_document_ids") or []

            start_t = time.perf_counter()
            retrieved_docs = retriever_fn(question, max_k)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            retrieved_ids = self.extract_document_ids(retrieved_docs)
            is_empty = (len(retrieved_ids) == 0)

            if is_answerable:
                r1 = compute_recall_at_k(retrieved_ids, expected_ids, k=1)
                r3 = compute_recall_at_k(retrieved_ids, expected_ids, k=3)
                r5 = compute_recall_at_k(retrieved_ids, expected_ids, k=5)
                p1 = compute_precision_at_k(retrieved_ids, expected_ids, k=1)
                p3 = compute_precision_at_k(retrieved_ids, expected_ids, k=3)
                p5 = compute_precision_at_k(retrieved_ids, expected_ids, k=5)
                h1 = compute_hit_at_k(retrieved_ids, expected_ids, k=1)
                h3 = compute_hit_at_k(retrieved_ids, expected_ids, k=3)
                h5 = compute_hit_at_k(retrieved_ids, expected_ids, k=5)
                mrr_val = compute_mrr(retrieved_ids, expected_ids, k=max_k)
                has_hit = (h5 == 1.0)
                is_fp = False
                unanswerable_flagged = False
            else:
                # Unanswerable query: no corpus document is relevant
                r1 = r3 = r5 = 0.0
                p1 = p3 = p5 = 0.0
                h1 = h3 = h5 = 0.0
                mrr_val = 0.0
                has_hit = False
                # If retriever returned documents for an out-of-corpus query, it's a retrieval false positive
                is_fp = not is_empty
                # Clean abstention at retrieval stage: retriever returned 0 documents
                unanswerable_flagged = is_empty

            result = QueryEvalResult(
                query_id=query_id,
                question=question,
                is_answerable=is_answerable,
                expected_ids=expected_ids,
                retrieved_ids=retrieved_ids,
                recall_at_1=r1,
                recall_at_3=r3,
                recall_at_5=r5,
                precision_at_1=p1,
                precision_at_3=p3,
                precision_at_5=p5,
                hit_at_1=h1,
                hit_at_3=h3,
                hit_at_5=h5,
                mrr=mrr_val,
                has_any_hit=has_hit,
                empty_retrieval=is_empty,
                is_false_positive=is_fp,
                unanswerable_flagged=unanswerable_flagged,
                latency_ms=elapsed_ms,
            )
            per_query_results.append(result)

        # Aggregate metrics over answerable queries only
        answerable_results = [r for r in per_query_results if r.is_answerable]
        unanswerable_results = [r for r in per_query_results if not r.is_answerable]

        num_ans = len(answerable_results)
        mean_r1 = sum(r.recall_at_1 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_r3 = sum(r.recall_at_3 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_r5 = sum(r.recall_at_5 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_p1 = sum(r.precision_at_1 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_p3 = sum(r.precision_at_3 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_p5 = sum(r.precision_at_5 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_h1 = sum(r.hit_at_1 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_h3 = sum(r.hit_at_3 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_h5 = sum(r.hit_at_5 for r in answerable_results) / num_ans if num_ans else 0.0
        mean_mrr_val = sum(r.mrr for r in answerable_results) / num_ans if num_ans else 0.0

        zero_hits = [r.query_id for r in answerable_results if not r.has_any_hit]
        unanswerable_empty = sum(1 for r in unanswerable_results if r.empty_retrieval)
        unanswerable_fp = sum(1 for r in unanswerable_results if r.is_false_positive)
        total_lat = sum(r.latency_ms for r in per_query_results)
        mean_latency = total_lat / len(per_query_results) if per_query_results else 0.0

        return AggregatedMetrics(
            total_queries=len(per_query_results),
            num_answerable=num_ans,
            num_unanswerable=len(unanswerable_results),
            mean_recall_at_1=mean_r1,
            mean_recall_at_3=mean_r3,
            mean_recall_at_5=mean_r5,
            mean_precision_at_1=mean_p1,
            mean_precision_at_3=mean_p3,
            mean_precision_at_5=mean_p5,
            mean_hit_at_1=mean_h1,
            mean_hit_at_3=mean_h3,
            mean_hit_at_5=mean_h5,
            mean_mrr=mean_mrr_val,
            zero_hit_query_count=len(zero_hits),
            zero_hit_query_ids=zero_hits,
            unanswerable_empty_retrieval_count=unanswerable_empty,
            unanswerable_false_positive_count=unanswerable_fp,
            unanswerable_handled_count=unanswerable_empty,
            mean_latency_ms=mean_latency,
            per_query_results=per_query_results,
        )
