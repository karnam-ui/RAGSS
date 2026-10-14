"""
Retrieval Evaluation Harness Engine.

Responsible for:
- Loading and validating retrieval datasets
- Invoking retrievers (live or fixture) with latency tracking
- Computing per-query and aggregate metrics
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
    compute_recall_at_k,
)


class RetrievalEvaluator:
    def __init__(
        self,
        dataset_path: Union[str, Path],
        target_id_level: str = "chunk",  # "chunk", "parent", or "document"
    ):
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
        """Extracts identifier strings from retrieved Document list based on target_id_level."""
        ids: List[str] = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            if self.target_id_level == "chunk":
                val = meta.get("chunk_id") or meta.get("parent_id") or meta.get("document_id")
            elif self.target_id_level == "parent":
                val = meta.get("parent_id") or meta.get("document_id")
            else:
                val = meta.get("document_id")

            if val:
                ids.append(str(val))
            else:
                ids.append(str(meta.get("source", "unknown")))
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

            # Select target IDs corresponding to target_id_level
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

            if is_answerable:
                r1 = compute_recall_at_k(retrieved_ids, expected_ids, k=1)
                r3 = compute_recall_at_k(retrieved_ids, expected_ids, k=3)
                r5 = compute_recall_at_k(retrieved_ids, expected_ids, k=5)
                mrr_val = compute_mrr(retrieved_ids, expected_ids)
                h1 = compute_hit_at_k(retrieved_ids, expected_ids, k=1)
                h3 = compute_hit_at_k(retrieved_ids, expected_ids, k=3)
                h5 = compute_hit_at_k(retrieved_ids, expected_ids, k=5)
                has_hit = bool(set(retrieved_ids).intersection(expected_ids))
                unanswerable_flagged = False
            else:
                # Unanswerable query: not expected to match any target
                r1 = 0.0
                r3 = 0.0
                r5 = 0.0
                mrr_val = 0.0
                h1 = 0.0
                h3 = 0.0
                h5 = 0.0
                has_hit = False
                # If retriever returned empty or ungrounded, unanswerable handled cleanly
                unanswerable_flagged = True

            result = QueryEvalResult(
                query_id=query_id,
                question=question,
                is_answerable=is_answerable,
                expected_ids=expected_ids,
                retrieved_ids=retrieved_ids,
                recall_at_1=r1,
                recall_at_3=r3,
                recall_at_5=r5,
                mrr=mrr_val,
                hit_at_1=h1,
                hit_at_3=h3,
                hit_at_5=h5,
                has_any_hit=has_hit,
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
        mean_mrr_val = sum(r.mrr for r in answerable_results) / num_ans if num_ans else 0.0

        zero_hits = [r.query_id for r in answerable_results if not r.has_any_hit]
        unanswerable_handled = sum(1 for r in unanswerable_results if r.unanswerable_flagged)

        return AggregatedMetrics(
            total_queries=len(per_query_results),
            num_answerable=num_ans,
            num_unanswerable=len(unanswerable_results),
            mean_recall_at_1=mean_r1,
            mean_recall_at_3=mean_r3,
            mean_recall_at_5=mean_r5,
            mean_mrr=mean_mrr_val,
            zero_hit_query_count=len(zero_hits),
            zero_hit_query_ids=zero_hits,
            unanswerable_handled_count=unanswerable_handled,
            per_query_results=per_query_results,
        )
