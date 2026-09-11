from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceBlock:
    """A paragraph-like unit extracted from the source document."""

    block_id: str
    order: int
    kind: str
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str | None = None

    def __post_init__(self) -> None:
        # Keep the source wording separately from normalized text.  This is
        # essential when a later review needs to quote or locate the original.
        if self.raw_text is None:
            self.raw_text = self.text


@dataclass(slots=True)
class SourceSpan:
    """Trace a prepared block back to its original document location."""

    block_id: str
    source_block_id: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(slots=True)
class BoundaryDecision:
    """Explain how two adjacent source blocks are structurally related."""

    boundary_id: str
    left_block_id: str
    right_block_id: str
    relation: str
    confidence: float
    decision_source: str
    evidence: list[str] = field(default_factory=list)
    review_required: bool = False
    warnings: list[str] = field(default_factory=list)
    rule_relation: str | None = None
    rule_confidence: float | None = None
    llm_model: str | None = None
    llm_reason: str | None = None
    right_block_role: str | None = None
    inferred_title: str | None = None


@dataclass(slots=True)
class Clause:
    """A contract clause assembled from one or more source blocks."""

    clause_id: str
    label: str
    title: str
    level: int
    heading: str
    body: str
    text: str
    parent_id: str | None
    block_ids: list[str]
    page_start: int | None = None
    page_end: int | None = None
    source_spans: list[SourceSpan] = field(default_factory=list)


@dataclass(slots=True)
class DocumentParseResult:
    filename: str
    file_type: str
    full_text: str
    blocks: list[SourceBlock]
    clauses: list[Clause]
    boundaries: list[BoundaryDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""

    @property
    def stats(self) -> dict[str, int]:
        return {
            "characters": len(self.full_text),
            "raw_characters": len(self.raw_text),
            "blocks": len(self.blocks),
            "clauses": len(self.clauses),
            "review_boundaries": sum(
                boundary.review_required for boundary in self.boundaries
            ),
            "llm_reviewed_boundaries": sum(
                boundary.llm_model is not None for boundary in self.boundaries
            ),
            "pages": max((block.page or 0 for block in self.blocks), default=0),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stats"] = self.stats
        return payload


@dataclass(slots=True)
class Element:
    """A contract element slot, filled by M3 and traceable to the source text.

    Instead of guessing party-specific fields before M3 exists, every element
    keeps a ``kind`` discriminator plus the location it came from, so later
    modules can always quote the original wording.
    """

    element_id: str
    kind: str
    value: str
    normalized: str = ""
    role: str = ""
    clause_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = ""


@dataclass(slots=True)
class ContractElements:
    """The four element slots; empty until M3 extracts their contents."""

    parties: list[Element] = field(default_factory=list)
    amounts: list[Element] = field(default_factory=list)
    dates: list[Element] = field(default_factory=list)
    subjects: list[Element] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Contract:
    """The unified contract object every module from M3 onwards consumes.

    It keeps the clause tree and source mapping produced by the parsing stage
    untouched under ``document``, and adds the element slots M3 will fill in.
    """

    contract_id: str
    document: DocumentParseResult
    elements: ContractElements = field(default_factory=ContractElements)
    schema_version: str = "m2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "filename": self.document.filename,
            "file_type": self.document.file_type,
            "stats": self.document.stats,
            "elements": self.elements.to_dict(),
            "document": self.document.to_dict(),
        }
