import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunking import PDFIngestor
from retrieval.retriver import Retriever
from retrieval.pipeline import BasicRetriever


def run_architecture(
    docs_dir: str = "tests/bank_docs",
    query: str = "what is the roi of savings acc of 22-23",
    db_dir: str = "tests/bank_chroma_db",
    k: int = 4,
):
    print("=" * 70)
    print(" RAG ARCHITECTURE: STEPWISE INGESTION, CHUNKING & RETRIEVAL")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: DOCUMENT INGESTION
    # ---------------------------------------------------------
    print("\n[STEP 1/5] INGESTION: Locating and Ingesting PDF Documents...")
    docs_path = Path(docs_dir).resolve()
    pdf_files = sorted(list(docs_path.glob("*.pdf")))

    if not pdf_files:
        print(f"❌ Error: No PDF files found in {docs_path}")
        return

    print(f"✓ Found {len(pdf_files)} PDF document(s) in: {docs_path}")
    for idx, pdf in enumerate(pdf_files, 1):
        print(f"  [{idx}] {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")

    ingestor = PDFIngestor()
    file_paths_str = [str(f) for f in pdf_files]

    # ---------------------------------------------------------
    # STEP 2: CHUNKING (PARENTS & CHILDREN)
    # ---------------------------------------------------------
    print("\n[STEP 2/5] CHUNKING: Generating Hierarchical Document Chunks...")
    parents, children = ingestor.ingest(file_paths_str)

    print(f"✓ Total Parent Documents (full pages): {len(parents)}")
    print(f"✓ Total Child Chunks (retrieval segments): {len(children)}")

    print("\nSample Chunk Metadata:")
    sample = children[0] if children else parents[0]
    print(f"  - Document ID:   {sample.metadata.get('document_id')}")
    print(f"  - Document Name: {sample.metadata.get('document_name')}")
    print(f"  - Page Number:   {sample.metadata.get('page')}")
    print(f"  - Chunk ID:      {sample.metadata.get('chunk_id')}")
    print(f"  - Content Snippet: {repr(sample.page_content[:120])}...")

    # ---------------------------------------------------------
    # STEP 3: EMBEDDINGS & VECTOR DATABASE INDEXING
    # ---------------------------------------------------------
    print("\n[STEP 3/5] INDEXING: Building Vector Store & BM25 Indices...")
    print(f"✓ Target Chroma DB: {db_dir}")
    print("✓ Embedding Model:  BAAI/bge-small-en-v1.5")
    print("✓ Cross-Encoder:    cross-encoder/ms-marco-MiniLM-L-6-v2")

    retriever = Retriever(
        file_paths=file_paths_str,
        persist_directory=str(PROJECT_ROOT / db_dir),
        collection_name="bank_multi_year_rates",
        rebuild_index=True,
    )
    print("✓ Indexed all document vectors and initialized BM25 lexical ranker.")

    # ---------------------------------------------------------
    # STEP 4: HYBRID RETRIEVAL & RERANKING
    # ---------------------------------------------------------
    print("\n[STEP 4/5] RETRIEVAL: Executing Hybrid Search & Cross-Encoder Reranking...")
    print(f"✓ Query: \"{query}\"")
    print(f"✓ Target Chunks to Retrieve (k): {k}")

    retrieved_docs = retriever.retrieve(query=query, k=k)
    print(f"✓ Retrieved {len(retrieved_docs)} relevant context document(s):")

    for idx, doc in enumerate(retrieved_docs, 1):
        doc_name = doc.metadata.get("document_name", "Unknown")
        page = doc.metadata.get("page", 0)
        display_page = int(page) + 1 if isinstance(page, int) else page
        print(f"\n  [Evidence {idx}] Document: {doc_name} | Page: {display_page}")
        lines = [line.strip() for line in doc.page_content.split("\n") if line.strip()]
        for line in lines[:6]:
            print(f"    | {line}")
        if len(lines) > 6:
            print(f"    | ... ({len(lines) - 6} more lines)")

    # ---------------------------------------------------------
    # STEP 5: RAG PIPELINE (LLM GENERATION)
    # ---------------------------------------------------------
    print("\n[STEP 5/5] GENERATION: Generating Grounded Answer with Citations...")
    pipeline = BasicRetriever(retriever)
    output = pipeline.generate(query)

    print("\n" + "=" * 70)
    print(f"QUESTION: {query}")
    print("=" * 70)
    print("ANSWER:")
    print(output.get("answer", "").strip())
    print("\nCITATIONS:")
    for cit in output.get("citations", []):
        print(f"  - [{cit.get('document_name')}] Page {cit.get('page')}: {cit.get('source')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG architecture on bank documents with stepwise confirmation."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="tests/bank_docs",
        help="Directory containing the bank PDF documents (default: tests/bank_docs)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="what is the roi of savings acc of 22-23",
        help="User query to evaluate (default: 'what is the roi of savings acc of 22-23')",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="tests/bank_chroma_db",
        help="Directory for the Chroma vector database (default: tests/bank_chroma_db)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of retrieved chunks (default: 3)",
    )
    args = parser.parse_args()

    run_architecture(
        docs_dir=args.docs_dir,
        query=args.query,
        db_dir=args.db_dir,
        k=args.k,
    )
