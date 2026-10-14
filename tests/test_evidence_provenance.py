"""
Deterministic unit tests for Evidence provenance, citations, and Claim structures.
Tests structural provenance separately from semantic support.
"""

import pytest
from langchain_core.documents import Document

from retrieval.evidence import (
    Claim,
    ConflictingFinding,
    CorroboratingFinding,
    Evidence,
    EvidenceAggregator,
    ResearchGap,
    SupportStatus,
    build_citation,
    build_citations,
)


def test_evidence_from_document_preserves_metadata():
    doc = Document(
        page_content="The Transformer relies entirely on an attention mechanism.",
        metadata={
            "document_id": "doc_1_paper1_transformers",
            "document_name": "paper1_transformers.pdf",
            "page": 0,  # 0-indexed in raw metadata
            "chunk_id": "doc_1_paper1_transformers_p0_c0",
            "parent_id": "doc_1_paper1_transformers_p0",
        },
    )

    ev = Evidence.from_document(doc, support_status=SupportStatus.SUPPORTED)

    assert ev.document_id == "doc_1_paper1_transformers"
    assert ev.document_name == "paper1_transformers.pdf"
    assert ev.page == 1  # Converted to 1-indexed for display
    assert ev.chunk_id == "doc_1_paper1_transformers_p0_c0"
    assert ev.parent_id == "doc_1_paper1_transformers_p0"
    assert ev.support_status == SupportStatus.SUPPORTED
    assert "Transformer relies" in ev.evidence_text


def test_build_citation_formatting_and_rejection():
    ev = Evidence(
        evidence_text="Sample text",
        document_id="doc_bert",
        document_name="paper2_bert.pdf",
        page=2,
    )

    # Bracket style
    assert build_citation(ev, style="bracket") == "[paper2_bert.pdf, p.2]"
    # Inline style
    assert build_citation(ev, style="inline") == "paper2_bert.pdf · p.2"

    # Must reject non-Evidence inputs (prevent hallucinated citations)
    with pytest.raises(TypeError):
        build_citation("paper2_bert.pdf: p.2")  # type: ignore


def test_build_citations_deduplication():
    ev1 = Evidence(evidence_text="Text 1", document_id="doc1", document_name="paper1.pdf", page=1)
    ev2 = Evidence(evidence_text="Text 2", document_id="doc1", document_name="paper1.pdf", page=1)  # Duplicate source/page
    ev3 = Evidence(evidence_text="Text 3", document_id="doc2", document_name="paper2.pdf", page=3)

    citations = build_citations([ev1, ev2, ev3])
    assert len(citations) == 2
    assert citations[0] == "[paper1.pdf, p.1]"
    assert citations[1] == "[paper2.pdf, p.3]"


def test_claim_structure_and_citations():
    ev = Evidence(evidence_text="Attention maps queries to keys", document_id="doc1", document_name="paper1.pdf", page=2)
    claim = Claim(
        text="Attention computes compatibility scores.",
        evidence=[ev],
        support_status=SupportStatus.SUPPORTED,
    )

    assert claim.text == "Attention computes compatibility scores."
    assert len(claim.get_citations()) == 1
    assert claim.get_citations()[0] == "[paper1.pdf, p.2]"


def test_evidence_aggregator_deduplication_and_grouping():
    ev1 = Evidence(evidence_text="A", document_id="doc1", document_name="p1.pdf", page=1, chunk_id="c1")
    ev2 = Evidence(evidence_text="A", document_id="doc1", document_name="p1.pdf", page=1, chunk_id="c1")  # Duplicate
    ev3 = Evidence(evidence_text="B", document_id="doc2", document_name="p2.pdf", page=2, chunk_id="c2")

    aggregator = EvidenceAggregator([ev1, ev2, ev3])
    assert len(aggregator.get_all_evidence()) == 3

    deduped = aggregator.deduplicate()
    assert len(deduped) == 2
    assert [e.chunk_id for e in deduped] == ["c1", "c2"]

    grouped = aggregator.group_by_document(by="document_name")
    assert "p1.pdf" in grouped
    assert "p2.pdf" in grouped
    assert len(grouped["p1.pdf"]) == 1


def test_cross_paper_finding_structures():
    ev1 = Evidence(evidence_text="T1", document_id="d1", document_name="p1.pdf", page=1)
    ev2 = Evidence(evidence_text="T2", document_id="d2", document_name="p2.pdf", page=1)

    corr = CorroboratingFinding(
        finding="Both architectures avoid recurrence.",
        supporting_papers=["p1.pdf", "p2.pdf"],
        evidence=[ev1, ev2],
    )
    assert len(corr.get_citations()) == 2
    assert corr.supporting_papers == ["p1.pdf", "p2.pdf"]

    conf = ConflictingFinding(
        topic="Pre-training speed",
        papers=["p1.pdf", "p2.pdf"],
        explanation="P1 reports 2x speedup while P2 reports 1.2x.",
        evidence=[ev1, ev2],
    )
    assert conf.topic == "Pre-training speed"

    gap = ResearchGap(
        finding="Neither paper explores long-context retrieval > 32k tokens.",
        relevant_papers=["p1.pdf", "p2.pdf"],
        evidence=[ev1, ev2],
    )
    assert len(gap.relevant_papers) == 2
