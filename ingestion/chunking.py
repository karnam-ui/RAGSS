import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestor:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @staticmethod
    def _generate_document_id(path_obj: Path, idx: int, seen_ids: Set[str]) -> str:
        """
        Generates a stable, deterministic document ID.
        If the filename already contains an explicit paper/doc number prefix (e.g. paper1_, paper2_),
        that canonical number is preserved even when ingested in isolation.
        Otherwise, canonical batch index is used.
        Disambiguates duplicate IDs if multiple files share identical stems.
        """
        stem = path_obj.stem
        match = re.match(r"^(?:paper|doc)[_]?(\d+)", stem)
        if match:
            candidate_id = f"doc_{match.group(1)}_{stem}"
        else:
            candidate_id = f"doc_{idx + 1}_{stem}"

        # Disambiguate if collision occurs in the same batch
        final_id = candidate_id
        counter = 2
        while final_id in seen_ids:
            final_id = f"{candidate_id}_{counter}"
            counter += 1

        seen_ids.add(final_id)
        return final_id

    def ingest(
        self, file_paths: Union[str, Path, List[Union[str, Path]]]
    ) -> Tuple[List[Document], List[Document]]:
        """
        Loads and chunks PDF documents into hierarchical parent pages and child retrieval chunks.

        Guarantees:
        - Deterministic ordering: canonical sorting ensures identical IDs regardless of input list ordering.
        - Strict file validation: raises FileNotFoundError for missing files and ValueError for 0-byte files.
        - Rich metadata propagation: document_id, document_name, file_path, source, parent_id, page, and chunk_id.
        - Whitespace filtering: excludes empty or whitespace-only child chunks.
        """
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        if not file_paths:
            return [], []

        # Validate existence and resolve paths
        resolved_paths: List[Path] = []
        seen_resolved = set()
        for p in file_paths:
            path_obj = Path(p).resolve()
            if not path_obj.exists() or not path_obj.is_file():
                raise FileNotFoundError(f"PDF document not found: {p}")
            if path_obj.stat().st_size == 0:
                raise ValueError(f"Cannot ingest empty document (0 bytes): {p}")
            path_str = str(path_obj)
            if path_str not in seen_resolved:
                seen_resolved.add(path_str)
                resolved_paths.append(path_obj)

        # Deterministic sorting so ingestion order doesn't alter document identity
        canonical_paths = sorted(resolved_paths, key=lambda p: (p.name.lower(), str(p)))

        all_parents: List[Document] = []
        all_children: List[Document] = []
        seen_doc_ids: Set[str] = set()

        for doc_idx, path_obj in enumerate(canonical_paths):
            document_id = self._generate_document_id(path_obj, doc_idx, seen_doc_ids)
            document_name = path_obj.name
            file_path_str = str(path_obj)

            loader = PyMuPDFLoader(file_path_str)
            parents = loader.load()
            if not parents:
                continue

            for page_idx, parent in enumerate(parents):
                parent_id = f"{document_id}_p{page_idx}"
                page_number = int(parent.metadata.get("page", page_idx))

                parent.metadata["document_id"] = document_id
                parent.metadata["document_name"] = document_name
                parent.metadata["file_path"] = file_path_str
                parent.metadata["source"] = document_name
                parent.metadata["parent_id"] = parent_id
                parent.metadata["page"] = page_number

                child_chunks = self.splitter.split_documents([parent])

                for child_idx, child in enumerate(child_chunks):
                    # Skip empty / whitespace-only chunks
                    if not child.page_content or not child.page_content.strip():
                        continue

                    chunk_id = f"{parent_id}_c{child_idx}"
                    child.metadata["document_id"] = document_id
                    child.metadata["document_name"] = document_name
                    child.metadata["file_path"] = file_path_str
                    child.metadata["source"] = document_name
                    child.metadata["parent_id"] = parent_id
                    child.metadata["chunk_id"] = chunk_id
                    child.metadata["child_id"] = child_idx
                    child.metadata["page"] = page_number

                    all_children.append(child)

                all_parents.append(parent)

        return all_parents, all_children