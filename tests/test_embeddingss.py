import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.embeddingss import Embed


class DummyEmbeddingModel:
    def embed_query(self, text: str):
        return [0.1] * 384


def test_embed_defaults_to_sample_papers_safely():
    mock_model = DummyEmbeddingModel()
    embedder = Embed(file_paths=None, embedding_model=mock_model)
    # Must not contain ~/Downloads/OS All Units Notes.pdf
    assert not any("Downloads" in p for p in embedder.file_paths)
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    if sample_dir.exists():
        assert len(embedder.file_paths) >= 1


def test_embed_dependency_injection_and_alias():
    mock_model = DummyEmbeddingModel()
    sample_pdf = PROJECT_ROOT / "tests" / "sample_papers" / "paper1_transformers.pdf"
    embedder = Embed(file_paths=[str(sample_pdf)], embedding_model=mock_model)

    assert embedder.embedding_model is mock_model
    parents, children = embedder.embed_chunks()
    assert len(parents) > 0
    assert len(children) > 0

    # Test alias consistency
    assert embedder.Embedchunks == embedder.embed_chunks


@pytest.mark.integration
def test_embedding_model_integration():
    sample_pdf = PROJECT_ROOT / "tests" / "sample_papers" / "paper1_transformers.pdf"
    embedder = Embed(file_paths=[str(sample_pdf)] if sample_pdf.exists() else None)
    parents, children = embedder.Embedchunks()
    assert len(parents) > 0
    assert len(children) > 0
    vector = embedder.embedding_model.embed_query("Transformer attention")
    assert len(vector) == 384  # bge-small-en-v1.5 dimension