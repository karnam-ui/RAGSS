import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFIngestor:

    def ingest(self, file_path: str):
        loader = PyMuPDFLoader(file_path)
        parents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )

        children = []

        for parent_id, parent in enumerate(parents):
            parent.metadata["parent_id"] = parent_id

            child_chunks = splitter.split_documents([parent])

            for child_id, child in enumerate(child_chunks):
                child.metadata["parent_id"] = parent_id
                child.metadata["child_id"] = child_id

                children.append(child)

        return parents, children