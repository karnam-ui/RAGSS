import sys
import re
from pathlib import Path

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
            self.bm25 = BM25Okapi([self._tokenize(document.page_content) for document in self.documents])
        else:
            self.bm25 = None

        if cross_encoder is not None:
            self.cross_encoder = None if cross_encoder is False else cross_encoder
        elif cross_encoder_model:
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        else:
            self.cross_encoder = None


    def index_documents(self):
        _, children = self.embedder.Embedchunks()
        existing_ids = self.vector_store.get().get("ids", [])
        if existing_ids:
            self.vector_store.delete(ids=existing_ids)
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
        self.bm25 = BM25Okapi([self._tokenize(document.page_content) for document in self.documents])

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

    def retrieve(self, query, k=4, candidate_k=8, document_id=None):
        search_kwargs = {}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        semantic_documents = self.vector_store.similarity_search(query, k=candidate_k, **search_kwargs)
        if document_id:
            semantic_documents = [
                doc for doc in semantic_documents
                if doc.metadata.get("document_id") == document_id
            ]

        bm25_indexes = []
        if self.bm25 is not None and self.documents:
            if document_id:
                matching_indexes = [
                    idx for idx, doc in enumerate(self.documents)
                    if doc.metadata.get("document_id") == document_id
                ]
                if matching_indexes:
                    scores = self.bm25.get_scores(self._tokenize(query))
                    sorted_matching = sorted(
                        matching_indexes, key=lambda idx: scores[idx], reverse=True
                    )
                    bm25_indexes = sorted_matching[:candidate_k]
            else:
                bm25_indexes = self.bm25.get_top_n(
                    self._tokenize(query),
                    list(range(len(self.documents))),
                    n=candidate_k
                )

        lexical_documents = [self.documents[index] for index in bm25_indexes]


        rrf_scores = {}
        document_by_key = {}
        for ranked_documents in (semantic_documents, lexical_documents):
            for rank, document in enumerate(ranked_documents, start=1):
                key = self._document_key(document)
                document_by_key[key] = document
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (60 + rank)

        fused_documents = [
            document_by_key[key]
            for key, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        if not fused_documents:
            return []

        if self.cross_encoder is not None:
            rerank_scores = self.cross_encoder.predict(
                [(query, document.page_content) for document in fused_documents]
            )
            reranked_children = sorted(
                zip(rerank_scores, fused_documents),
                key=lambda item: item[0],
                reverse=True,
            )
        else:
            reranked_children = [(1.0, doc) for doc in fused_documents]


        parent_documents = []
        seen_parent_ids = set()
        for _, child in reranked_children:
            parent_id = child.metadata.get("parent_id")
            if parent_id in seen_parent_ids:
                continue

            parent = self.parent_by_id.get(parent_id)
            if parent is not None:
                parent_documents.append(parent)
                seen_parent_ids.add(parent_id)
            else:
                parent_documents.append(child)
                seen_parent_ids.add(parent_id or child.metadata.get("chunk_id"))

            if len(parent_documents) == k:
                break

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
