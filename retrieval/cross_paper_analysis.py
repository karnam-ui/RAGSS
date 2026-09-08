import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from retrieval.evidence import (
    ConflictingFinding,
    CorroboratingFinding,
    Evidence,
    EvidenceAggregator,
    ResearchGap,
)
from retrieval.paper_analysis import PaperAnalysis


@dataclass
class CrossPaperAnalysis:
    similarities: str
    differences: str
    corroborating_findings: str
    conflicting_findings: str
    research_gaps: str
    overall_synthesis: str
    analyzed_papers: List[Dict[str, str]] = field(default_factory=list)
    source_evidence: List[Evidence] = field(default_factory=list)
    corroborating_findings_structured: List[CorroboratingFinding] = field(default_factory=list)
    conflicting_findings_structured: List[ConflictingFinding] = field(default_factory=list)
    research_gaps_structured: List[ResearchGap] = field(default_factory=list)


CROSS_PAPER_PROMPT = ChatPromptTemplate.from_template("""You are an expert meta-scientific research analyst.

You are given structured analyses of {num_papers} research papers.
Your task is to produce a concise cross-paper comparison and research synthesis based ONLY on the provided paper analyses.

Strict Grounding Rules:
- Base your analysis ONLY on the provided paper summaries.
- Do not introduce facts, benchmarks, numbers, or conclusions not stated in these summaries.
- Do not assume two papers disagree merely because they examine different tasks or employ different architectures.
- Do not manufacture consensus if the papers address separate topics.
- Distinguish between a topic that is "not mentioned" versus one that is "contradicted".
- If there is insufficient evidence to compare or identify conflicts, explicitly state: "None identified from the provided evidence."
- Explicitly reference the relevant paper names (e.g. paper1, paper2) when attributing findings or comparisons.
- Keep the synthesis concise, objective, and research-oriented.

Format your output EXACTLY with the following section headers:
SIMILARITIES:
<Common research objectives, shared methods, or common findings across papers>

DIFFERENCES:
<Differences in problem formulation, methodologies, datasets/benchmarks, or assumptions>

CORROBORATING FINDINGS:
<Claims or findings supported by multiple papers, explicitly naming the supporting papers>

CONFLICTING FINDINGS:
<Points where papers present contradictory claims/results, or "None identified from the provided evidence.">

RESEARCH GAPS:
<Recurring limitations, missing empirical evidence, or unresolved areas across the papers>

OVERALL SYNTHESIS:
<Concise overarching summary of what the collection collectively demonstrates, distinguishing consensus from divergence>

Paper Analyses to Compare:
{papers_context}

Structured Cross-Paper Synthesis:
""")


class CrossPaperAnalyzer:
    def __init__(self, llm: Optional[Any] = None):
        if llm is None:
            from retrieval.retrieve_llm_pipeline import llm as default_llm
            self.llm = default_llm
        else:
            self.llm = llm

        self.chain = CROSS_PAPER_PROMPT | self.llm

    def format_paper_analyses(self, paper_analyses: List[PaperAnalysis]) -> str:
        """Formats structured PaperAnalysis objects into cleanly bounded text blocks for LLM reasoning."""
        formatted_blocks = []
        for idx, paper in enumerate(paper_analyses, 1):
            block = (
                f"--- PAPER [{idx}]: {paper.document_name} (ID: {paper.document_id}) ---\n"
                f"Title: {paper.title}\n"
                f"Research Problem: {paper.research_problem}\n"
                f"Methodology: {paper.methodology}\n"
                f"Major Claims: {paper.major_claims}\n"
                f"Evidence: {paper.evidence}\n"
                f"Limitations: {paper.limitations}\n"
            )
            formatted_blocks.append(block)
        return "\n".join(formatted_blocks)

    def analyze(self, paper_analyses: List[PaperAnalysis]) -> CrossPaperAnalysis:
        if not paper_analyses:
            return CrossPaperAnalysis(
                similarities="No papers provided for analysis.",
                differences="No papers provided for analysis.",
                corroborating_findings="No papers provided for analysis.",
                conflicting_findings="None identified from the provided evidence.",
                research_gaps="No papers provided for analysis.",
                overall_synthesis="No papers provided for analysis.",
                analyzed_papers=[],
            )

        paper_meta_list = [
            {
                "document_id": p.document_id,
                "document_name": p.document_name,
                "title": p.title,
            }
            for p in paper_analyses
        ]

        # Aggregate all source evidence across papers
        aggregator = EvidenceAggregator()
        for p in paper_analyses:
            aggregator.add_evidence(p.source_evidence)
        all_evidence = aggregator.deduplicate()

        if len(paper_analyses) == 1:
            single = paper_analyses[0]
            single_gaps = (
                [ResearchGap(finding=single.limitations, relevant_papers=[single.document_name], evidence=list(single.source_evidence))]
                if single.limitations and "not available" not in single.limitations.lower()
                else []
            )
            return CrossPaperAnalysis(
                similarities=f"Only one paper provided ({single.document_name}). Similarities require at least two papers.",
                differences=f"Only one paper provided ({single.document_name}). Differences require at least two papers.",
                corroborating_findings="None (single paper).",
                conflicting_findings="None identified from the provided evidence.",
                research_gaps=single.limitations,
                overall_synthesis=f"Analysis of single paper '{single.title}': addresses {single.research_problem} using {single.methodology}.",
                analyzed_papers=paper_meta_list,
                source_evidence=all_evidence,
                corroborating_findings_structured=[],
                conflicting_findings_structured=[],
                research_gaps_structured=single_gaps,
            )

        papers_context = self.format_paper_analyses(paper_analyses)
        llm_response = self.chain.invoke({
            "num_papers": len(paper_analyses),
            "papers_context": papers_context,
        })

        parsed = self._parse_cross_analysis_output(llm_response)
        corroborating, conflicting, gaps = self._build_structured_findings(parsed, paper_analyses)

        return CrossPaperAnalysis(
            similarities=parsed["similarities"],
            differences=parsed["differences"],
            corroborating_findings=parsed["corroborating_findings"],
            conflicting_findings=parsed["conflicting_findings"],
            research_gaps=parsed["research_gaps"],
            overall_synthesis=parsed["overall_synthesis"],
            analyzed_papers=paper_meta_list,
            source_evidence=all_evidence,
            corroborating_findings_structured=corroborating,
            conflicting_findings_structured=conflicting,
            research_gaps_structured=gaps,
        )

    @staticmethod
    def _detect_papers_for_text(text: str, paper_analyses: List[PaperAnalysis]) -> List[PaperAnalysis]:
        """Identifies which papers are referenced in a finding text without trusting hallucinated citations."""
        matched = []
        text_lower = text.lower()
        for idx, paper in enumerate(paper_analyses, 1):
            stem = Path(paper.document_name).stem.lower()
            name = paper.document_name.lower()
            doc_id = paper.document_id.lower()
            patterns = [
                rf"\bpaper\s*\[?{idx}\]?",
                rf"\bdoc_{idx}\b",
                re.escape(stem),
                re.escape(name),
                re.escape(doc_id),
            ]
            if any(re.search(pat, text_lower) for pat in patterns):
                matched.append(paper)

        if not matched:
            matched = list(paper_analyses)
        return matched

    def _build_structured_findings(
        self,
        parsed: Dict[str, str],
        paper_analyses: List[PaperAnalysis],
    ) -> Tuple[List[CorroboratingFinding], List[ConflictingFinding], List[ResearchGap]]:
        from typing import Tuple
        corroborating: List[CorroboratingFinding] = []
        conflicting: List[ConflictingFinding] = []
        gaps: List[ResearchGap] = []

        # Corroborating findings
        corr_text = parsed.get("corroborating_findings", "")
        if corr_text and "none identified" not in corr_text.lower():
            lines = [re.sub(r"^[\s*\-\d\.\)]+", "", l).strip() for l in corr_text.split("\n") if l.strip()]
            for line in lines:
                if line:
                    papers = self._detect_papers_for_text(line, paper_analyses)
                    paper_evidence = [ev for p in papers for ev in p.source_evidence]
                    corroborating.append(
                        CorroboratingFinding(
                            finding=line,
                            supporting_papers=[p.document_name for p in papers],
                            evidence=EvidenceAggregator(paper_evidence).deduplicate(),
                        )
                    )

        # Conflicting findings
        conf_text = parsed.get("conflicting_findings", "")
        if conf_text and "none identified" not in conf_text.lower():
            lines = [re.sub(r"^[\s*\-\d\.\)]+", "", l).strip() for l in conf_text.split("\n") if l.strip()]
            for line in lines:
                if line:
                    papers = self._detect_papers_for_text(line, paper_analyses)
                    paper_evidence = [ev for p in papers for ev in p.source_evidence]
                    conflicting.append(
                        ConflictingFinding(
                            topic=line,
                            papers=[p.document_name for p in papers],
                            explanation=line,
                            evidence=EvidenceAggregator(paper_evidence).deduplicate(),
                        )
                    )

        # Research gaps
        gap_text = parsed.get("research_gaps", "")
        if gap_text and "none identified" not in gap_text.lower() and "not available" not in gap_text.lower():
            lines = [re.sub(r"^[\s*\-\d\.\)]+", "", l).strip() for l in gap_text.split("\n") if l.strip()]
            for line in lines:
                if line:
                    papers = self._detect_papers_for_text(line, paper_analyses)
                    paper_evidence = [ev for p in papers for ev in p.source_evidence]
                    gaps.append(
                        ResearchGap(
                            finding=line,
                            relevant_papers=[p.document_name for p in papers],
                            evidence=EvidenceAggregator(paper_evidence).deduplicate(),
                        )
                    )

        return corroborating, conflicting, gaps

    @staticmethod
    def _parse_cross_analysis_output(raw_text: str) -> Dict[str, str]:
        """Robust parser extracting the 6 comparative sections across varied markdown/heading styles."""
        fields = {
            "similarities": "Not available in the provided evidence.",
            "differences": "Not available in the provided evidence.",
            "corroborating_findings": "Not available in the provided evidence.",
            "conflicting_findings": "None identified from the provided evidence.",
            "research_gaps": "Not available in the provided evidence.",
            "overall_synthesis": "Not available in the provided evidence.",
        }

        header_patterns = [
            ("similarities", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:SIMILARITIES?)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:DIFFERENCES?|CORROBORAT|CONFLICT|RESEARCH\s*GAPS?|OVERALL)|$)"),
            ("differences", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:DIFFERENCES?)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:CORROBORAT|CONFLICT|RESEARCH\s*GAPS?|OVERALL)|$)"),
            ("corroborating_findings", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:CORROBORATING\s*FINDINGS?|CORROBORATION)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:CONFLICT|RESEARCH\s*GAPS?|OVERALL)|$)"),
            ("conflicting_findings", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:CONFLICTING\s*FINDINGS?|CONFLICTS?)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:RESEARCH\s*GAPS?|OVERALL)|$)"),
            ("research_gaps", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:RESEARCH\s*GAPS?|GAPS?\s*(?:AND|&)?\s*LIMITATIONS?)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:OVERALL|SYNTHESIS)|$)"),
            ("overall_synthesis", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:OVERALL\s*SYNTHESIS|SYNTHESIS|CONCLUSION)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:Assistant:|###)|$)"),
        ]

        for key, pattern in header_patterns:
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                val = match.group(1).strip().strip("*#-_ \t\n")
                if val:
                    fields[key] = val

        return fields


def analyze_cross_paper(
    paper_analyses: List[PaperAnalysis],
    llm: Optional[Any] = None,
) -> CrossPaperAnalysis:
    analyzer = CrossPaperAnalyzer(llm=llm)
    return analyzer.analyze(paper_analyses)
