"""
Deterministic unit tests for hierarchical parent-child document expansion and fallback logic.
"""

from typing import List
import pytest
from langchain_core.documents import Document

from retrieval.retriver import Retriever


class MockVectorStore:
    def __init__(self, return_docs: List[Document]):
        self.return_docs = return_docs

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        return self.return_docs[:k]


def test_parent_expansion_resolves_correct_parent():
    child = Document(page_content="Child chunk", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "d1"})
    parent = Document(page_content="Full parent page context", metadata={"parent_id": "p1", "document_id": "d1"})

    retriever = Retriever(
        embedder=object(),
        vector_store=MockVectorStore([child]),
        cross_encoder=False,
        parents=[parent],
        documents=[child],
    )

    results = retriever.retrieve("query", k=1)
    assert len(results) == 1
    # Successfully expanded to parent
    assert results[0].page_content == "Full parent page context"
    assert results[0].metadata["parent_id"] == "p1"


def test_missing_parent_falls_back_to_child():
    # Child refers to parent_id 'p_missing' which is not in parents list
    child = Document(page_content="Child chunk fallback", metadata={"chunk_id": "c_orphan", "parent_id": "p_missing", "document_id": "d1"})

    retriever = Retriever(
        embedder=object(),
        vector_store=MockVectorStore([child]),
        cross_encoder=False,
        parents=[],  # No parents known
        documents=[child],
    )

    results = retriever.retrieve("query", k=1)
    assert len(results) == 1
    # Safe fallback to child content
    assert results[0].page_content == "Child chunk fallback"
    assert results[0].metadata["chunk_id"] == "c_orphan"


def test_multiple_children_from_same_parent_deduplicated():
    # Two children from the SAME parent page
    child1 = Document(page_content="Child chunk 1", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "d1"})
    child2 = Document(page_content="Child chunk 2", metadata={"chunk_id": "c2", "parent_id": "p1", "document_id": "d1"})
    parent = Document(page_content="Single Parent Page", metadata={"parent_id": "p1", "document_id": "d1"})

    retriever = Retriever(
        embedder=object(),
        vector_store=MockVectorStore([child1, child2]),
        cross_encoder=False,
        parents=[parent],
        documents=[child1, child2],
    )

    results = retriever.retrieve("query", k=5)
    # The parent should only appear ONCE in returned results, not twice!
    assert len(results) == 1
    assert results[0].metadata["parent_id"] == "p1"
