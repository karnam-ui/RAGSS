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


def test_multiple_distinct_orphan_chunks_not_dropped():
    # Two orphan chunks with no parent_id
    child1 = Document(page_content="Orphan 1", metadata={"chunk_id": "c1", "document_id": "d1"})
    child2 = Document(page_content="Orphan 2", metadata={"chunk_id": "c2", "document_id": "d1"})

    retriever = Retriever(
        embedder=object(),
        vector_store=MockVectorStore([child1, child2]),
        cross_encoder=False,
        parents=[],
        documents=[child1, child2],
    )

    results = retriever.retrieve("query", k=2)
    # Both distinct orphan chunks must be returned; none should be dropped
    assert len(results) == 2
    assert results[0].metadata["chunk_id"] == "c1"
    assert results[1].metadata["chunk_id"] == "c2"


def test_parent_expansion_enforces_document_id_filter():
    # Child from doc1 points to parent that has mismatched document_id 'doc2'
    child1 = Document(page_content="Child 1", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "doc1"})
    corrupt_parent = Document(page_content="Parent Corrupt", metadata={"parent_id": "p1", "document_id": "doc2"})

    retriever = Retriever(
        embedder=object(),
        vector_store=MockVectorStore([child1]),
        cross_encoder=False,
        parents=[corrupt_parent],
        documents=[child1],
    )

    # When filtering for doc1, the corrupt parent with doc2 must be rejected
    results = retriever.retrieve("query", k=1, document_id="doc1")
    assert len(results) == 0


def test_parent_expansion_respects_exact_k():
    children = [
        Document(page_content=f"Child {i}", metadata={"chunk_id": f"c{i}", "parent_id": f"p{i}", "document_id": f"d{i}"})
        for i in range(10)
    ]
    parents = [
        Document(page_content=f"Parent {i}", metadata={"parent_id": f"p{i}", "document_id": f"d{i}"})
        for i in range(10)
    ]

    retriever = Retriever(
        embedder=object(),
        vector_store=MockVectorStore(children),
        cross_encoder=False,
        parents=parents,
        documents=children,
    )

    results = retriever.retrieve("query", k=3)
    assert len(results) == 3
