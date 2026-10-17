"""
Deterministic unit tests for Hybrid Retrieval (Dense + BM25), Reciprocal Rank Fusion (RRF),
Cross-Encoder reranking, input boundaries, and filter invariants using dependency injection.
Executes completely offline without downloading models.
"""

from typing import Dict, List, Optional
import pytest
from langchain_core.documents import Document

from retrieval.retriver import Retriever


class MockVectorStore:
    def __init__(self, return_docs: List[Document], should_raise: bool = False):
        self.return_docs = return_docs
        self.should_raise = should_raise
        self.last_query: Optional[str] = None
        self.last_k: Optional[int] = None
        self.last_kwargs: Dict = {}

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        if self.should_raise:
            raise RuntimeError("Simulated vector store connection failure")
        self.last_query = query
        self.last_k = k
        self.last_kwargs = kwargs

        docs = self.return_docs
        filter_dict = kwargs.get("filter")
        if filter_dict:
            docs = [
                d for d in docs
                if all(d.metadata.get(f_k) == f_v for f_k, f_v in filter_dict.items())
            ]
        return docs[:k]


class MockCrossEncoder:
    def __init__(self, score_mapping: dict, should_raise: bool = False):
        self.score_mapping = score_mapping
        self.should_raise = should_raise

    def predict(self, pairs: List[tuple]) -> List[float]:
        if self.should_raise:
            raise RuntimeError("Simulated cross-encoder inference failure")
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
        embedder=object(),
        vector_store=mock_vs,
        cross_encoder=mock_ce,
        parents=mock_parents,
        documents=mock_children,
    )

    results = retriever.retrieve("language model query", k=2, candidate_k=2)

    assert len(results) == 2
    # Because cross-encoder scored child2 highest (0.95 vs 0.2), parent2 should be ranked 1st
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

    assert retriever.retrieve("query", k=3) == []


def test_input_validation_boundaries():
    child = Document(page_content="Text", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "d1"})
    parent = Document(page_content="Parent", metadata={"parent_id": "p1", "document_id": "d1"})
    mock_vs = MockVectorStore(return_docs=[child])

    retriever = Retriever(
        embedder=object(),
        vector_store=mock_vs,
        cross_encoder=False,
        parents=[parent],
        documents=[child],
    )

    # Empty / whitespace / None queries must return []
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve(None) == []  # type: ignore

    # Zero or negative k must return []
    assert retriever.retrieve("test", k=0) == []
    assert retriever.retrieve("test", k=-1) == []
    assert retriever.retrieve("test", k=-99) == []

    # Valid query returns results
    res = retriever.retrieve("test", k=1)
    assert len(res) == 1
    assert res[0].metadata["parent_id"] == "p1"


def test_candidate_k_auto_expansion():
    # If caller asks for k=5 but candidate_k was passed as 2,
    # candidate_k should be clamped to at least k so 5 items can be considered
    children = [
        Document(page_content=f"Text {i}", metadata={"chunk_id": f"c{i}", "parent_id": f"p{i}", "document_id": f"d{i}"})
        for i in range(5)
    ]
    parents = [
        Document(page_content=f"Parent {i}", metadata={"parent_id": f"p{i}", "document_id": f"d{i}"})
        for i in range(5)
    ]
    mock_vs = MockVectorStore(return_docs=children)

    retriever = Retriever(
        embedder=object(),
        vector_store=mock_vs,
        cross_encoder=False,
        parents=parents,
        documents=children,
    )

    results = retriever.retrieve("query", k=5, candidate_k=2)
    # mock_vs should have received k >= 5, not candidate_k=2
    assert mock_vs.last_k >= 5
    assert len(results) == 5


def test_document_id_filter_enforcement_in_vector_store_and_bm25():
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

    # Assert vector store was passed the filter parameter
    assert mock_vs.last_kwargs.get("filter") == {"document_id": "doc2"}

    # Assert all returned documents belong strictly to doc2
    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc2"
    assert results[0].metadata["parent_id"] == "p2"


def test_cross_encoder_failure_degrades_gracefully_to_rrf():
    child1 = Document(page_content="Child 1", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "d1"})
    parent1 = Document(page_content="Parent 1", metadata={"parent_id": "p1", "document_id": "d1"})
    mock_vs = MockVectorStore(return_docs=[child1])
    failing_ce = MockCrossEncoder(score_mapping={}, should_raise=True)

    retriever = Retriever(
        embedder=object(),
        vector_store=mock_vs,
        cross_encoder=failing_ce,
        parents=[parent1],
        documents=[child1],
    )

    # Should not raise; falls back gracefully to RRF order
    results = retriever.retrieve("query", k=1)
    assert len(results) == 1
    assert results[0].metadata["parent_id"] == "p1"


def test_vector_store_failure_degrades_to_lexical_bm25():
    child1 = Document(page_content="transformer language model", metadata={"chunk_id": "c1", "parent_id": "p1", "document_id": "d1"})
    parent1 = Document(page_content="Full Transformer Paper", metadata={"parent_id": "p1", "document_id": "d1"})
    failing_vs = MockVectorStore(return_docs=[], should_raise=True)

    retriever = Retriever(
        embedder=object(),
        vector_store=failing_vs,
        cross_encoder=False,
        parents=[parent1],
        documents=[child1],
    )

    # BM25 should still locate child1 despite vector store failure
    results = retriever.retrieve("transformer", k=1)
    assert len(results) == 1
    assert results[0].metadata["parent_id"] == "p1"


def test_rebuild_index_with_empty_documents_does_not_divide_by_zero():
    class DummyEmptyStore:
        def get(self, *args, **kwargs):
            return {"documents": [], "metadatas": [], "ids": []}
        def delete(self, *args, **kwargs):
            pass

    empty_store = DummyEmptyStore()
    retriever = Retriever(
        embedder=object(),
        vector_store=empty_store,
        cross_encoder=False,
        parents=[],
        documents=[],
    )

    # rebuild_index calls _load_documents -> empty list -> BM25 initialization
    # Must not raise ZeroDivisionError
    retriever.rebuild_index()
    assert retriever.bm25 is None
