import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from ingestion.chunking import PDFIngestor
from ingestion.embeddingss import Embed
from retrieval.retriver import Retriever


@pytest.mark.integration
def test_multi_document_pipeline():
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    assert len(sample_files) >= 3, "Expected at least 3 sample PDF papers for testing."

    file_paths = [str(f) for f in sample_files]
    print(f"Testing with {len(file_paths)} PDF documents:")
    for fp in file_paths:
        print(f" - {Path(fp).name}")

    # 1. Ingestion & Metadata verification
    print("\n--- Step 1: Testing Multi-Document Ingestion ---")
    ingestor = PDFIngestor()
    parents, children = ingestor.ingest(file_paths)

    print(f"Total parent documents: {len(parents)}")
    print(f"Total child chunks: {len(children)}")

    chunk_ids = set()
    doc_ids_seen = set()
    doc_name_to_id = {}

    for chunk in children:
        meta = chunk.metadata
        assert "document_id" in meta, "Missing document_id in chunk metadata"
        assert "document_name" in meta, "Missing document_name in chunk metadata"
        assert "page" in meta, "Missing page in chunk metadata"
        assert "chunk_id" in meta, "Missing chunk_id in chunk metadata"

        c_id = meta["chunk_id"]
        assert c_id not in chunk_ids, f"Duplicate chunk_id found: {c_id}"
        chunk_ids.add(c_id)

        d_name = meta["document_name"]
        d_id = meta["document_id"]
        doc_ids_seen.add(d_id)

        if d_name in doc_name_to_id:
            assert doc_name_to_id[d_name] == d_id, f"Inconsistent document_id for {d_name}"
        else:
            doc_name_to_id[d_name] = d_id

    assert len(doc_ids_seen) == len(file_paths), (
        f"Expected {len(file_paths)} unique document_ids, got {len(doc_ids_seen)}"
    )
    print("✓ Metadata checks passed: All chunks have unique chunk_ids and consistent document_ids.")

    # 2. Embedding & Chroma Shared Indexing
    print("\n--- Step 2: Shared Chroma Indexing & BM25 Corpus ---")
    test_db_dir = PROJECT_ROOT / "tests" / "test_chroma_db"
    retriever = Retriever(
        file_paths=file_paths,
        persist_directory=str(test_db_dir),
        collection_name="multi_doc_test_papers",
        rebuild_index=True,
    )

    print(f"Chroma corpus size: {len(retriever.documents)} documents")
    print(f"BM25 corpus size: {len(retriever.bm25.doc_len)} documents")
    assert len(retriever.documents) == len(children)
    assert len(retriever.bm25.doc_len) == len(children)
    print("✓ Indexing check passed: Shared collection and BM25 encompass the full multi-document corpus.")

    # 3. Hybrid Retrieval across Multiple Documents
    print("\n--- Step 3: Retrieval Across Multiple Documents ---")
    query = "attention mechanism and bidirectional transformer representation"
    retrieved_parents = retriever.retrieve(query, k=4)

    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(retrieved_parents)} parent documents:")
    retrieved_doc_names = set()
    retrieved_doc_ids = set()

    for idx, doc in enumerate(retrieved_parents, 1):
        d_name = doc.metadata.get("document_name")
        d_id = doc.metadata.get("document_id")
        page = doc.metadata.get("page")
        retrieved_doc_names.add(d_name)
        retrieved_doc_ids.add(d_id)
        print(f" [{idx}] Paper: {d_name} | Document ID: {d_id} | Page: {page + 1 if page is not None else 'N/A'}")
        print(f"     Content excerpt: {doc.page_content[:120].strip()}...\n")

    print(f"Distinct document_names in retrieval: {retrieved_doc_names}")
    print(f"Distinct document_ids in retrieval: {retrieved_doc_ids}")
    assert len(retrieved_doc_names) > 1, "Expected results from multiple distinct papers."
    print("✓ Multi-document retrieval verified successfully!")


if __name__ == "__main__":
    test_multi_document_pipeline()
