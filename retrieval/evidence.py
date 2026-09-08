import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


@dataclass
class Evidence:
    evidence_text: str
    document_id: str
    document_name: str
    page: int
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    support_status: SupportStatus = SupportStatus.NOT_ENOUGH_EVIDENCE

    @classmethod
    def from_document(
        cls,
        doc: Document,
        support_status: SupportStatus = SupportStatus.NOT_ENOUGH_EVIDENCE,
    ) -> "Evidence":
        if not isinstance(doc, Document):
            raise TypeError(f"Expected langchain_core.documents.Document, got {type(doc)}")

        raw_page = doc.metadata.get("page", 0)
        # Consistent with project convention: raw PDF pages are 0-indexed, display citations are 1-indexed
        try:
            page_num = int(raw_page) + 1
        except (ValueError, TypeError):
            page_num = 1

        return cls(
            evidence_text=doc.page_content,
            document_id=doc.metadata.get("document_id", "unknown_doc"),
            document_name=doc.metadata.get("document_name", "unknown_document.pdf"),
            page=page_num,
            chunk_id=doc.metadata.get("chunk_id"),
            parent_id=doc.metadata.get("parent_id"),
            support_status=support_status,
        )


def build_citation(evidence: Evidence, style: str = "bracket") -> str:
    """
    Deterministically formats a citation string from an Evidence object.
    Strictly uses Evidence.document_name and Evidence.page.
    Rejects raw strings or ungrounded LLM inputs.
    """
    if not isinstance(evidence, Evidence):
        raise TypeError(
            f"build_citation strictly requires an Evidence instance, got {type(evidence)}. "
            "Arbitrary LLM citation strings are not permitted."
        )

    if style == "inline":
        return f"{evidence.document_name} · p.{evidence.page}"
    return f"[{evidence.document_name}, p.{evidence.page}]"


def build_citations(evidence_list: List[Evidence], style: str = "bracket") -> List[str]:
    """
    Deterministically formats and deduplicates citation strings across a list of Evidence objects,
    preserving order of appearance.
    """
    seen: Set[str] = set()
    citations: List[str] = []
    for ev in evidence_list:
        cit = build_citation(ev, style=style)
        if cit not in seen:
            seen.add(cit)
            citations.append(cit)
    return citations


@dataclass
class Claim:
    text: str
    evidence: List[Evidence] = field(default_factory=list)
    support_status: SupportStatus = SupportStatus.SUPPORTED

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


@dataclass
class CorroboratingFinding:
    finding: str
    supporting_papers: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


@dataclass
class ConflictingFinding:
    topic: str
    papers: List[str] = field(default_factory=list)
    explanation: str = ""
    evidence: List[Evidence] = field(default_factory=list)

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


@dataclass
class ResearchGap:
    finding: str
    relevant_papers: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


class EvidenceAggregator:
    """
    In-memory service responsible for collecting, deduplicating,
    grouping, and linking Evidence objects across papers and claims.
    """

    def __init__(self, evidence_list: Optional[List[Evidence]] = None):
        self.evidence: List[Evidence] = []
        if evidence_list:
            self.add_evidence(evidence_list)

    def add_evidence(self, evidence: Union[Evidence, List[Evidence]]) -> None:
        if isinstance(evidence, Evidence):
            self.evidence.append(evidence)
        elif isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, Evidence):
                    raise TypeError(f"Expected Evidence object, got {type(item)}")
                self.evidence.append(item)
        else:
            raise TypeError(f"Expected Evidence or List[Evidence], got {type(evidence)}")

    def get_all_evidence(self) -> List[Evidence]:
        return list(self.evidence)

    def deduplicate(self) -> List[Evidence]:
        """
        Deduplicates identical evidence instances based on chunk_id or (document_id, page, content).
        Preserves original order.
        """
        seen: Set[Any] = set()
        deduped: List[Evidence] = []
        for ev in self.evidence:
            if ev.chunk_id:
                key = ("chunk_id", ev.chunk_id)
            elif ev.parent_id:
                key = ("parent_id", ev.parent_id, ev.evidence_text.strip())
            else:
                key = ("content", ev.document_id, ev.page, ev.evidence_text.strip())

            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        self.evidence = deduped
        return deduped

    def group_by_document(self, by: str = "document_name") -> Dict[str, List[Evidence]]:
        """
        Groups aggregated evidence by document name or document id.
        """
        grouped: Dict[str, List[Evidence]] = {}
        for ev in self.evidence:
            key = ev.document_id if by == "document_id" else ev.document_name
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(ev)
        return grouped

    def get_sources_for_paper(self, paper_identifier: str) -> List[Evidence]:
        """
        Retrieves all evidence belonging to a given document_id or document_name.
        """
        return [
            ev for ev in self.evidence
            if ev.document_id == paper_identifier or ev.document_name == paper_identifier
        ]

    def get_sources_for_claim(self, claim: Claim) -> List[Evidence]:
        """
        Retrieves attached evidence for a specific claim.
        """
        if not isinstance(claim, Claim):
            raise TypeError(f"Expected Claim instance, got {type(claim)}")
        return list(claim.evidence)

    def build_source_references(self, style: str = "bracket") -> List[str]:
        """
        Produces sorted, deduplicated citation strings for all aggregated evidence.
        """
        deduped = self.deduplicate()
        return build_citations(deduped, style=style)
