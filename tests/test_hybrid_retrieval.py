"""
Deterministic unit tests for Hybrid Retrieval (Dense + BM25), Reciprocal Rank Fusion (RRF),
and Cross-Encoder reranking using dependency injection.
Executes completely offline without downloading models.
"""

from pathlib import Path
from typing import List
import pytest
from langchain_core.documents import Document

from retrieval.retriver import Retriever


class MockVectorStore:
    def __init__(self, return_docs: List[Document]):
        self.return_docs = return_docs

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        return self.return_docs[:k]


class MockCrossEncoder:
    def __init__(self, score_mapping: dict):
        self.score_mapping = score_mapping

    def predict(self, pairs: List[tuple]) -> List[float]:
        # Returns mapped score for the document content in the pair
        scores = []
        for query, content in pairs:
            scores.append(self.score_mapping.get(content, 0.0))
        return scores


def test_rrf_and_cross_encoder_reranking_ordering():
    # Construct 3 child chunks
    child1 = Document(
        page_content="Content 1 about attention",
        metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "d1"},
    )
    child2 = Document(
        page_content="Content 2 about language model",
        metadata={"chunk_id": "c2", "parent_id": "p2", "document_id": "d2"},
    )
    child3 = Document(
        page_content="Content 3 about retrieval augmented generation",
        metadata={"chunk_id": "c3", "parent_id": "p3", "document_id": "d3"},
    )

    parent1 = Document(page_content="Full Page 1", metadata={"parent_id": "p1", "document_id": "d1"})
    parent2 = Document(page_content="Full Page 2", metadata={"parent_id": "p2", "document_id": "d2"})
    parent3 = Document(page_content="Full Page 3", metadata={"parent_id": "p3", "document_id": "d3"})

    mock_parents = [parent1, parent2, parent3]
    mock_children = [child1, child2, child3]

    # Vector search returns [child1, child2]
    mock_vs = MockVectorStore(return_docs=[child1, child2])

    # Cross encoder assigns highest score to child2
    mock_ce = MockCrossEncoder({
        child1.page_content: 0.2,
        child2.page_content: 0.95,
        child3.page_content: 0.1,
    })

    retriever = Retriever(
        embedder=object(),  # Dummy embedder
        vector_store=mock_vs,
        cross_encoder=mock_ce,
        parents=mock_parents,
        documents=mock_children,
    )

    results = retriever.retrieve("language model query", k=2, candidate_k=2)

    assert len(results) == 2
    # Because cross-encoder scored child2 highest (0.95 vs 0.2), parent2 should be ranked 1st!
    assert results[0].metadata["parent_id"] == "p2"
    assert results[1].metadata["parent_id"] == "p1"


def test_empty_retrieval_returns_empty_list():
    mock_vs = MockVectorStore(return_docs=[])
    retriever = Retriever(
        embedder=object(),
        vector_store=mock_vs,
        cross_encoder=False,
        parents=[],
        documents=[],
    )

    results = retriever.retrieve("query", k=3)
    assert results == []


def test_document_id_filter_in_retrieval():
    child1 = Document(page_content="Paper 1 text", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "doc1"})
    child2 = Document(page_content="Paper 2 text", metadata={"chunk_id": "c2", "parent_id": "p2", "document_id": "doc2"})

    parent1 = Document(page_content="Parent 1", metadata={"parent_id": "p1", "document_id": "doc1"})
    parent2 = Document(page_content="Parent 2", metadata={"parent_id": "p2", "document_id": "doc2"})

    mock_vs = MockVectorStore(return_docs=[child1, child2])

    retriever = Retriever(
        embedder=object(),
        vector_store=mock_vs,
        cross_encoder=False,
        parents=[parent1, parent2],
        documents=[child1, child2],
    )

    # Filter strictly for doc2
    results = retriever.retrieve("test query", k=2, document_id="doc2")
    assert all(d.metadata.get("document_id") == "doc2" for d in results)
