import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple, Union

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestor:

    def ingest(
        self, file_paths: Union[str, Path, List[Union[str, Path]]]
    ) -> Tuple[List[Document], List[Document]]:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )

        all_parents: List[Document] = []
        all_children: List[Document] = []

        for doc_idx, file_path in enumerate(file_paths):
            path_obj = Path(file_path)
            # Deterministic document ID unique per document in the ingestion batch
            document_id = f"doc_{doc_idx + 1}_{path_obj.stem}"
            document_name = path_obj.name

            loader = PyMuPDFLoader(str(path_obj))
            parents = loader.load()

            for page_idx, parent in enumerate(parents):
                parent_id = f"{document_id}_p{page_idx}"
                parent.metadata["document_id"] = document_id
                parent.metadata["document_name"] = document_name
                parent.metadata["parent_id"] = parent_id

                child_chunks = splitter.split_documents([parent])

                for child_idx, child in enumerate(child_chunks):
                    chunk_id = f"{parent_id}_c{child_idx}"
                    child.metadata["document_id"] = document_id
                    child.metadata["document_name"] = document_name
                    child.metadata["parent_id"] = parent_id
                    child.metadata["chunk_id"] = chunk_id
                    child.metadata["child_id"] = child_idx

                    all_children.append(child)

                all_parents.append(parent)

        return all_parents, all_children