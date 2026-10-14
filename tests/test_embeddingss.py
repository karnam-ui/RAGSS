import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from ingestion.embeddingss import Embed


@pytest.mark.integration
def test_embedding_model_integration():
    sample_pdf = PROJECT_ROOT / "tests" / "sample_papers" / "paper1_transformers.pdf"
    embedder = Embed(file_paths=[str(sample_pdf)] if sample_pdf.exists() else None)
    parents, children = embedder.Embedchunks()
    assert len(parents) > 0
    assert len(children) > 0
    vector = embedder.embedding_model.embed_query("Transformer attention")
    assert len(vector) == 384  # bge-small-en-v1.5 dimension


if __name__ == "__main__":
    sample_pdf = PROJECT_ROOT / "tests" / "sample_papers" / "paper1_transformers.pdf"
    embedder = Embed(file_paths=[str(sample_pdf)] if sample_pdf.exists() else None)
    parents, children = embedder.Embedchunks()
    print(f"Embedded {len(parents)} parents and {len(children)} children.")
    vector = embedder.embedding_model.embed_query("Transformer attention")
    print(f"Generated embedding vector of dimension: {len(vector)}")