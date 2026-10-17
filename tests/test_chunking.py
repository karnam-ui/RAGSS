import sys
import tempfile
from pathlib import Path
import pytest

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
        assert "source" in chunk.metadata
        assert "file_path" in chunk.metadata
        assert len(chunk.page_content.strip()) > 0


def test_ingestion_order_invariance():
    ingestor = PDFIngestor()
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    p1 = str(sample_dir / "paper1_transformers.pdf")
    p2 = str(sample_dir / "paper2_bert.pdf")

    # Ingest in forward order
    parents_fwd, children_fwd = ingestor.ingest([p1, p2])
    fwd_doc_ids = [p.metadata["document_id"] for p in parents_fwd]
    fwd_chunk_ids = [c.metadata["chunk_id"] for c in children_fwd]

    # Ingest in reversed order
    parents_rev, children_rev = ingestor.ingest([p2, p1])
    rev_doc_ids = [p.metadata["document_id"] for p in parents_rev]
    rev_chunk_ids = [c.metadata["chunk_id"] for c in children_rev]

    # The canonical sorting ensures exact identity invariance regardless of input order
    assert fwd_doc_ids == rev_doc_ids
    assert fwd_chunk_ids == rev_chunk_ids


def test_single_file_preserves_canonical_id():
    ingestor = PDFIngestor()
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    p2 = str(sample_dir / "paper2_bert.pdf")

    # Even when ingested in isolation, paper2 must have doc_2_paper2_bert
    parents, children = ingestor.ingest(p2)
    assert len(parents) > 0
    assert parents[0].metadata["document_id"] == "doc_2_paper2_bert"
    assert children[0].metadata["document_id"] == "doc_2_paper2_bert"
    assert children[0].metadata["chunk_id"].startswith("doc_2_paper2_bert_p0_c")


def test_duplicate_file_paths_deduplicated():
    ingestor = PDFIngestor()
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    p1 = str(sample_dir / "paper1_transformers.pdf")

    # Pass duplicate paths in same batch
    parents_single, children_single = ingestor.ingest([p1])
    parents_dup, children_dup = ingestor.ingest([p1, p1])

    # Should not duplicate chunks
    assert len(parents_single) == len(parents_dup)
    assert len(children_single) == len(children_dup)


def test_missing_file_raises_filenotfound():
    ingestor = PDFIngestor()
    with pytest.raises(FileNotFoundError, match="PDF document not found"):
        ingestor.ingest("/tmp/nonexistent_paper_for_test_123.pdf")


def test_empty_file_raises_valueerror():
    ingestor = PDFIngestor()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        with pytest.raises(ValueError, match="Cannot ingest empty document"):
            ingestor.ingest(tmp.name)


def test_empty_input_returns_empty_lists():
    ingestor = PDFIngestor()
    parents, children = ingestor.ingest([])
    assert parents == []
    assert children == []