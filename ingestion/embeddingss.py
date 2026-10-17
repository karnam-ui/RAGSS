import sys
from pathlib import Path
from typing import List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunking import PDFIngestor


class Embed:
    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]], None] = None,
        embedding_model=None,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        if file_paths is None:
            # Default to bundled sample papers if available, rather than an arbitrary personal path
            sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
            if sample_dir.exists():
                sample_files = sorted(list(sample_dir.glob("*.pdf")))
                self.file_paths = [str(f) for f in sample_files]
            else:
                self.file_paths = []
        elif isinstance(file_paths, (str, Path)):
            self.file_paths = [str(file_paths)]
        else:
            self.file_paths = [str(p) for p in file_paths]

        # Alias for backward compatibility if code references embed.file_path
        self.file_path = self.file_paths[0] if self.file_paths else None

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def Embedchunks(self):
        ingestor = PDFIngestor()
        return ingestor.ingest(self.file_paths)

    # Standard pythonic alias
    embed_chunks = Embedchunks
