import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.embeddingss import Embed


class Retriever:
    def __init__(
        self,
        file_path=None,
        persist_directory=None,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = Embed(file_path)
        self.parents, _ = self.embedder.Embedchunks()
        self.parent_by_id = {
            document.metadata["parent_id"]: document
            for document in self.parents
        }
        self.persist_directory = Path(persist_directory or PROJECT_ROOT / "chroma_db")
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedder.embedding_model,
        )

        if self._needs_indexing():
            self.index_documents()

        self.documents = self._load_documents()
        self.bm25 = BM25Okapi([self._tokenize(document.page_content) for document in self.documents])
        self.cross_encoder = CrossEncoder(cross_encoder_model)

    def index_documents(self):
        _, children = self.embedder.Embedchunks()
        existing_ids = self.vector_store.get().get("ids", [])
        if existing_ids:
            self.vector_store.delete(ids=existing_ids)
        self.vector_store = Chroma.from_documents(
            documents=children,
            embedding=self.embedder.embedding_model,
            persist_directory=str(self.persist_directory),
        )

    def _needs_indexing(self):
        stored = self.vector_store.get(include=["metadatas"])
        metadatas = stored.get("metadatas", [])
        return not metadatas or any(
            metadata is None or "parent_id" not in metadata
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

    def retrieve(self, query, k=4, candidate_k=8):
        semantic_documents = self.vector_store.similarity_search(query, k=candidate_k)
        bm25_indexes = self.bm25.get_top_n(
            self._tokenize(query), list(range(len(self.documents))), n=candidate_k
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
        rerank_scores = self.cross_encoder.predict(
            [(query, document.page_content) for document in fused_documents]
        )
        reranked_children = sorted(
            zip(rerank_scores, fused_documents),
            key=lambda item: item[0],
            reverse=True,
        )

        print(list(rerank_scores))

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

            if len(parent_documents) == k:
                break

        return parent_documents

    @staticmethod
    def _document_key(document):
        return (document.metadata.get("source"), document.metadata.get("page"), document.page_content)
