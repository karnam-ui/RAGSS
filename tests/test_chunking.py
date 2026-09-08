import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunking import PDFIngestor

ingestor = PDFIngestor()
sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
sample_files = sorted(list(sample_dir.glob("*.pdf")))
if sample_files:
    test_paths = [str(f) for f in sample_files]
else:
    test_paths = ["/Users/suhaas/Downloads/Morphex.pdf"]

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