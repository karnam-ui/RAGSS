import sys
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.embeddingss import Embed

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        file_paths=None,
        file_path=None,
        persist_directory=None,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name="research_papers",
        rebuild_index=False,
        embedder=None,
        vector_store=None,
        cross_encoder=None,
        parents=None,
        documents=None,
    ):
        target_paths = file_paths if file_paths is not None else file_path

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = Embed(target_paths)

        if parents is not None:
            self.parents = parents
        else:
            self.parents, _ = self.embedder.Embedchunks()

        self.parent_by_id = {
            document.metadata["parent_id"]: document
            for document in self.parents
            if document.metadata.get("parent_id")
        }

        self.persist_directory = Path(persist_directory or PROJECT_ROOT / "chroma_db")
        self.collection_name = collection_name

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
                embedding_function=getattr(self.embedder, "embedding_model", None),
            )

            if rebuild_index or self._needs_indexing():
                self.index_documents()

        if documents is not None:
            self.documents = documents
        else:
            self.documents = self._load_documents()

        if self.documents:
            tokenized_corpus = [self._tokenize(document.page_content) for document in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._doc_token_sets = [set(tokens) for tokens in tokenized_corpus]
        else:
            self.bm25 = None
            self._doc_token_sets = []

        if cross_encoder is not None:
            self.cross_encoder = None if cross_encoder is False else cross_encoder
        elif cross_encoder_model:
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        else:
            self.cross_encoder = None

    def index_documents(self):
        if not hasattr(self.embedder, "Embedchunks"):
            return
        _, children = self.embedder.Embedchunks()
        existing_ids = self.vector_store.get().get("ids", [])
        if existing_ids:
            self.vector_store.delete(ids=existing_ids)
        if children:
            self.vector_store = Chroma.from_documents(
                documents=children,
                embedding=self.embedder.embedding_model,
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
            )

    def rebuild_index(self):
        """Safely clears and re-indexes all documents into the shared Chroma collection."""
        self.index_documents()
        self.documents = self._load_documents()
        if self.documents:
            tokenized_corpus = [self._tokenize(document.page_content) for document in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._doc_token_sets = [set(tokens) for tokens in tokenized_corpus]
        else:
            self.bm25 = None
            self._doc_token_sets = []

    def _needs_indexing(self):
        stored = self.vector_store.get(include=["metadatas"])
        metadatas = stored.get("metadatas", [])
        return not metadatas or any(
            metadata is None
            or "parent_id" not in metadata
            or "document_id" not in metadata
            or "chunk_id" not in metadata
            for metadata in metadatas
        )

    def _load_documents(self):
        stored = self.vector_store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]

    @staticmethod
    def _tokenize(text):
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def get_indexed_papers(self):
        """Returns a list of unique paper metadata dictionaries [{'document_id': ..., 'document_name': ...}]."""
        seen_ids = set()
        papers = []
        for doc in self.documents:
            doc_id = doc.metadata.get("document_id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                papers.append({
                    "document_id": doc_id,
                    "document_name": doc.metadata.get("document_name", "Unknown"),
                })
        return papers

    def retrieve(
        self,
        query: str,
        k: int = 4,
        candidate_k: Optional[int] = None,
        document_id: Optional[str] = None,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Executes hybrid retrieval (dense vector similarity + lexical BM25), reciprocal rank fusion (RRF),
        optional cross-encoder reranking, and hierarchical parent-child expansion.

        Guarantees:
        - If query is empty, whitespace-only, or non-string: returns [] immediately.
        - If k <= 0: returns [] immediately.
        - candidate_k is clamped to max(candidate_k, k) so candidate pool is never smaller than requested k.
        - If document_id is provided, both dense and lexical branches filter strictly, and parent expansion
          preserves the filter invariant.
        - BM25 candidates strictly require lexical term overlap with query tokens (no zero-overlap hallucinations).
        - If cross-encoder fails during reranking, gracefully logs a warning and falls back to RRF rank order.
        - Orphan chunks without resolvable parents fall back safely without blocking subsequent orphans.
        """
        # 1. Input validation & contract boundaries
        if query is None or not isinstance(query, str) or not query.strip():
            return []

        if k is None:
            k = 4
        try:
            k = int(k)
        except (ValueError, TypeError):
            return []
        if k <= 0:
            return []

        if candidate_k is None or candidate_k <= 0:
            candidate_k = max(8, k * 2)
        else:
            try:
                candidate_k = int(candidate_k)
            except (ValueError, TypeError):
                candidate_k = max(8, k * 2)
        candidate_k = max(candidate_k, k)

        if document_id is not None:
            document_id = str(document_id).strip() or None

        # 2. Dense semantic retrieval
        search_kwargs = {}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        try:
            semantic_documents = self.vector_store.similarity_search(query, k=candidate_k, **search_kwargs)
        except Exception as e:
            logger.warning("Vector store similarity search failed (%s). Continuing with empty dense pool.", e)
            semantic_documents = []

        if document_id:
            semantic_documents = [
                doc for doc in semantic_documents
                if doc.metadata.get("document_id") == document_id
            ]

        # 3. Lexical BM25 retrieval (strictly requiring lexical term overlap)
        bm25_indexes = []
        if self.bm25 is not None and self.documents:
            tokenized_query = self._tokenize(query)
            query_token_set = set(tokenized_query)
            if query_token_set:
                scores = self.bm25.get_scores(tokenized_query)
                if document_id:
                    candidate_indexes = [
                        idx for idx, doc in enumerate(self.documents)
                        if doc.metadata.get("document_id") == document_id
                        and bool(query_token_set.intersection(self._doc_token_sets[idx]))
                    ]
                else:
                    candidate_indexes = [
                        idx for idx in range(len(self.documents))
                        if bool(query_token_set.intersection(self._doc_token_sets[idx]))
                    ]

                if candidate_indexes:
                    sorted_indexes = sorted(
                        candidate_indexes, key=lambda idx: scores[idx], reverse=True
                    )
                    bm25_indexes = sorted_indexes[:candidate_k]

        lexical_documents = [self.documents[index] for index in bm25_indexes]

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[Any, float] = {}
        document_by_key: Dict[Any, Document] = {}
        for ranked_documents in (semantic_documents, lexical_documents):
            for rank, document in enumerate(ranked_documents, start=1):
                key = self._document_key(document)
                if key not in document_by_key:
                    document_by_key[key] = document
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (float(rrf_k) + float(rank))

        # Stable sort descending by RRF score (Timsort preserves order of appearance for ties)
        fused_documents = [
            document_by_key[key]
            for key, _ in sorted(
                rrf_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        if not fused_documents:
            return []

        # 5. Optional Cross-Encoder reranking with fault tolerance
        if self.cross_encoder is not None:
            try:
                rerank_scores = self.cross_encoder.predict(
                    [(query, document.page_content) for document in fused_documents]
                )
                reranked_children = sorted(
                    zip(rerank_scores, fused_documents),
                    key=lambda item: item[0],
                    reverse=True,
                )
            except Exception as e:
                logger.warning("Cross-encoder reranking failed (%s). Falling back to RRF rank order.", e)
                reranked_children = [(1.0, doc) for doc in fused_documents]
        else:
            reranked_children = [(1.0, doc) for doc in fused_documents]

        # 6. Hierarchical parent expansion with strict deduplication & filter validation
        parent_documents: List[Document] = []
        seen_parent_ids: Set[str] = set()
        seen_chunk_keys: Set[Any] = set()

        for _, child in reranked_children:
            if len(parent_documents) >= k:
                break

            parent_id = child.metadata.get("parent_id")
            parent = self.parent_by_id.get(parent_id) if parent_id else None

            if parent is not None:
                # If document_id filter is active, enforce on parent document
                if document_id and parent.metadata.get("document_id") != document_id:
                    continue
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                parent_documents.append(parent)
            else:
                # Fallback: orphan chunk without resolvable parent
                if document_id and child.metadata.get("document_id") != document_id:
                    continue
                chunk_key = self._document_key(child)
                if chunk_key in seen_chunk_keys:
                    continue
                seen_chunk_keys.add(chunk_key)
                parent_documents.append(child)

        return parent_documents

    @staticmethod
    def _document_key(document):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return chunk_id
        return (
            document.metadata.get("document_id"),
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.page_content,
        )
