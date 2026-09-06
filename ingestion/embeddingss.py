import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunking import PDFIngestor
from langchain_huggingface import HuggingFaceEmbeddings

class Embed:

    def __init__(self, file_path=None):
        self.file_path = file_path or str(Path.home() / "Downloads" / "OS All Units Notes.pdf")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def Embedchunks(self):
        ingestor = PDFIngestor()
        return ingestor.ingest(self.file_path)

