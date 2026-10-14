import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from retrieval.paper_analysis import PaperAnalysis, analyze_papers
from retrieval.retriver import Retriever


@pytest.mark.integration
def test_paper_analysis():
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    assert len(sample_files) >= 3, "Expected at least 3 sample PDF papers."

    file_paths = [str(f) for f in sample_files]
    print(f"Testing paper analysis on {len(file_paths)} papers:")
    for p in file_paths:
        print(f" - {Path(p).name}")

    # Use a separate test Chroma directory to avoid interfering with production/dev storage
    test_db_dir = PROJECT_ROOT / "tests" / "test_paper_analysis_db"
    retriever = Retriever(
        file_paths=file_paths,
        persist_directory=str(test_db_dir),
        collection_name="test_paper_analysis_collection",
        rebuild_index=True,
    )

    print("\nRunning analyze_papers()...")
    analyses = analyze_papers(file_paths=file_paths, retriever=retriever)

    # 1. Verify all 3 papers received an analysis
    assert len(analyses) == len(file_paths), (
        f"Expected {len(file_paths)} analyses, got {len(analyses)}"
    )
    print(f"\n✓ Generated {len(analyses)} paper analyses.")

    expected_doc_names = {Path(p).name for p in file_paths}
    analyzed_doc_names = set()

    for idx, analysis in enumerate(analyses, 1):
        assert isinstance(analysis, PaperAnalysis), "Output must be instance of PaperAnalysis"

        # 2. Verify correct document_name and document_id
        assert analysis.document_name in expected_doc_names, (
            f"Unexpected document_name: {analysis.document_name}"
        )
        assert analysis.document_id.startswith("doc_"), (
            f"Invalid document_id format: {analysis.document_id}"
        )
        analyzed_doc_names.add(analysis.document_name)

        # 3. Verify evidence and citations do not mix across papers
        for citation in analysis.citations:
            assert citation["document_id"] == analysis.document_id, (
                f"Citation document_id mismatch: {citation['document_id']} != {analysis.document_id}"
            )
            assert citation["document_name"] == analysis.document_name, (
                f"Citation document_name mismatch: {citation['document_name']} != {analysis.document_name}"
            )
            assert "page" in citation, "Citation must contain page number"
            assert "source" in citation, "Citation must contain source path"

        # 4. Verify all structured fields are present and non-empty
        assert analysis.title, "Title field cannot be empty"
        assert analysis.research_problem, "Research problem field cannot be empty"
        assert analysis.methodology, "Methodology field cannot be empty"
        assert analysis.major_claims, "Major claims field cannot be empty"
        assert analysis.evidence, "Evidence field cannot be empty"
        assert analysis.limitations, "Limitations field cannot be empty"

        print(f"\n========================================================")
        print(f"Paper Analysis [{idx}/{len(analyses)}]: {analysis.document_name}")
        print(f"Document ID:      {analysis.document_id}")
        print(f"Title:            {analysis.title}")
        print(f"Research Problem: {analysis.research_problem}")
        print(f"Methodology:      {analysis.methodology}")
        print(f"Major Claims:     {analysis.major_claims}")
        print(f"Evidence:         {analysis.evidence}")
        print(f"Limitations:      {analysis.limitations}")
        print(f"Citations:        {analysis.citations}")
        print(f"========================================================")

    assert analyzed_doc_names == expected_doc_names, (
        f"Missing analyses for: {expected_doc_names - analyzed_doc_names}"
    )
    print("\n✓ All assertions passed: Isolated paper analyses generated successfully with complete metadata.")


if __name__ == "__main__":
    test_paper_analysis()
