import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.cross_paper_analysis import CrossPaperAnalysis, CrossPaperAnalyzer, analyze_cross_paper
from retrieval.paper_analysis import PaperAnalysis, analyze_papers
from retrieval.retriver import Retriever


def test_cross_paper_pipeline():
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    assert len(sample_files) >= 3, "Expected at least 3 sample PDF papers."

    file_paths = [str(f) for f in sample_files]
    print(f"Testing Cross-Paper Analysis across {len(file_paths)} papers:")
    for p in file_paths:
        print(f" - {Path(p).name}")

    # Step 1: Generate individual PaperAnalysis objects
    print("\n--- Step 1: Paper-Level Analysis ---")
    test_db_dir = PROJECT_ROOT / "tests" / "test_cross_paper_db"
    retriever = Retriever(
        file_paths=file_paths,
        persist_directory=str(test_db_dir),
        collection_name="test_cross_paper_collection",
        rebuild_index=True,
    )
    paper_analyses = analyze_papers(file_paths=file_paths, retriever=retriever)
    assert len(paper_analyses) == 3, f"Expected 3 paper analyses, got {len(paper_analyses)}"
    print(f"✓ Generated {len(paper_analyses)} paper analyses.")

    # Step 2: Cross-Paper Analysis across all 3 papers
    print("\n--- Step 2: Multi-Paper Synthesis ---")
    analyzer = CrossPaperAnalyzer()
    cross_result = analyzer.analyze(paper_analyses)

    assert isinstance(cross_result, CrossPaperAnalysis), "Output must be an instance of CrossPaperAnalysis"
    assert len(cross_result.analyzed_papers) == 3, "All 3 papers must be listed in analyzed_papers"

    expected_doc_names = {Path(p).name for p in file_paths}
    actual_doc_names = {p["document_name"] for p in cross_result.analyzed_papers}
    assert actual_doc_names == expected_doc_names, (
        f"Analyzed papers mismatch: {actual_doc_names} != {expected_doc_names}"
    )

    # Check that all sections are populated
    assert cross_result.similarities, "Similarities section cannot be empty"
    assert cross_result.differences, "Differences section cannot be empty"
    assert cross_result.corroborating_findings, "Corroborating findings section cannot be empty"
    assert cross_result.conflicting_findings, "Conflicting findings section cannot be empty"
    assert cross_result.research_gaps, "Research gaps section cannot be empty"
    assert cross_result.overall_synthesis, "Overall synthesis section cannot be empty"

    print("\n========================================================")
    print("CROSS-PAPER RESEARCH SYNTHESIS")
    print("========================================================")
    print(f"Analyzed Papers ({len(cross_result.analyzed_papers)}):")
    for p in cross_result.analyzed_papers:
        print(f" - {p['document_name']} ({p['document_id']}): {p['title']}")
    print(f"\n[1] SIMILARITIES:\n{cross_result.similarities}\n")
    print(f"[2] DIFFERENCES:\n{cross_result.differences}\n")
    print(f"[3] CORROBORATING FINDINGS:\n{cross_result.corroborating_findings}\n")
    print(f"[4] CONFLICTING FINDINGS:\n{cross_result.conflicting_findings}\n")
    print(f"[5] RESEARCH GAPS:\n{cross_result.research_gaps}\n")
    print(f"[6] OVERALL SYNTHESIS:\n{cross_result.overall_synthesis}\n")
    print("========================================================")
    print("✓ Full 3-paper cross-paper analysis verified successfully!")

    # Step 3: Verify pipeline adaptability with different number of papers (e.g. 2 papers)
    print("\n--- Step 3: Verifying Adaptability with 2 Papers ---")
    two_papers = paper_analyses[:2]
    subset_result = analyze_cross_paper(two_papers)
    assert len(subset_result.analyzed_papers) == 2
    assert subset_result.similarities
    assert subset_result.differences
    print("✓ 2-paper subset synthesis succeeded!")

    # Step 4: Verify single-paper edge case
    print("\n--- Step 4: Verifying 1-Paper Edge Case ---")
    one_paper = [paper_analyses[0]]
    single_result = analyze_cross_paper(one_paper)
    assert len(single_result.analyzed_papers) == 1
    assert "Only one paper" in single_result.similarities
    print("✓ Single-paper handling verified successfully!")


if __name__ == "__main__":
    test_cross_paper_pipeline()
