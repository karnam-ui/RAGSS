import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriver import Retriever
from qb import bank

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "tests" / "sample_papers" / "paper1_transformers.pdf")
    retriever = Retriever(file_paths=[target_file])

    for q in bank:
        results = retriever.retrieve(q)
        print(f"\n==================================================")
        print(f"Query: {q}")
        print(f"Retrieved {len(results)} document(s):")
        for idx, doc in enumerate(results, 1):
            doc_name = doc.metadata.get("document_name", "Unknown")
            page = doc.metadata.get("page", "?")
            print(f"\n[{idx}] Document: {doc_name} | Page: {page}")
            print(f"Content:\n{doc.page_content.strip()}")
        print("==================================================")
