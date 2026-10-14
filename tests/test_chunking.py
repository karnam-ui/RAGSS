import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunking import PDFIngestor

def test_pdf_ingestor_chunking():
    ingestor = PDFIngestor()
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    assert len(sample_files) >= 1, "Expected at least one sample paper in tests/sample_papers"
    test_paths = [str(f) for f in sample_files]

    parents, children = ingestor.ingest(test_paths)
    assert len(parents) > 0, "Ingestion must produce at least one parent page"
    assert len(children) >= len(parents), "Children chunks must be >= parent pages"

    for chunk in children:
        assert "document_id" in chunk.metadata
        assert "document_name" in chunk.metadata
        assert "page" in chunk.metadata
        assert "parent_id" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content.strip()) > 0


if __name__ == "__main__":
    test_pdf_ingestor_chunking()
    ingestor = PDFIngestor()
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    test_paths = [str(f) for f in sample_files]
    parents, children = ingestor.ingest(test_paths)
    print(f"Total parents: {len(parents)}, Total children: {len(children)}")
    for chunk in children[:5]:
        print("-----------------")
        print("Content:", chunk.page_content[:80], "...")
        print("Metadata:", {
            "document_id": chunk.metadata.get("document_id"),
            "document_name": chunk.metadata.get("document_name"),
            "page": chunk.metadata.get("page"),
            "chunk_id": chunk.metadata.get("chunk_id"),
            "parent_id": chunk.metadata.get("parent_id"),
        })