"""Contract upload, parsing, and clause segmentation for the demo."""

from .models import (
    BoundaryDecision,
    Clause,
    Contract,
    ContractElements,
    DocumentParseResult,
    Element,
    SourceBlock,
    SourceSpan,
)
from .service import build_contract, derive_contract_id, parse_contract

__all__ = [
    "BoundaryDecision",
    "Clause",
    "Contract",
    "ContractElements",
    "DocumentParseResult",
    "Element",
    "SourceBlock",
    "SourceSpan",
    "build_contract",
    "derive_contract_id",
    "parse_contract",
]
