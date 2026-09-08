import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Union
import torch

from ingestion.chunking import PDFIngestor
from langchain_huggingface import HuggingFaceEmbeddings


class Embed:

    def __init__(self, file_paths: Union[str, Path, List[Union[str, Path]], None] = None):
        if file_paths is None:
            default_path = str(Path.home() / "Downloads" / "OS All Units Notes.pdf")
            self.file_paths = [default_path]
        elif isinstance(file_paths, (str, Path)):
            self.file_paths = [str(file_paths)]
        else:
            self.file_paths = [str(p) for p in file_paths]

        # Alias for backward compatibility if code references embed.file_path
        self.file_path = self.file_paths[0] if self.file_paths else None

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def Embedchunks(self):
        ingestor = PDFIngestor()
        return ingestor.ingest(self.file_paths)

