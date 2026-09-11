from __future__ import annotations

from hashlib import sha256

from .models import Contract, DocumentParseResult
from .parsers import parse_blocks
from .splitter import analyze_clause_structure
from .validation import validate_structure


def parse_contract(filename: str, content: bytes) -> DocumentParseResult:
    file_type, source_blocks, warnings = parse_blocks(filename, content)
    split_result = analyze_clause_structure(source_blocks)
    blocks = split_result.blocks
    clauses = split_result.clauses
    boundaries = split_result.boundaries
    full_text = "\n\n".join(block.text for block in source_blocks)
    raw_text = "\n\n".join(block.raw_text or block.text for block in source_blocks)

    if len(clauses) == 1 and clauses[0].level == 0:
        warnings.append("未识别到明确的条款编号，已将全文作为一个正文区块。")
    warnings.extend(
        f"[结构校验] {warning}"
        for warning in validate_structure(blocks, clauses, boundaries)
    )

    return DocumentParseResult(
        filename=filename,
        file_type=file_type,
        full_text=full_text,
        blocks=blocks,
        clauses=clauses,
        boundaries=boundaries,
        warnings=warnings,
        raw_text=raw_text,
    )


def derive_contract_id(result: DocumentParseResult) -> str:
    """Stable identifier so the same contract always maps to the same id."""

    digest = sha256(
        (result.filename + "\x00" + result.full_text).encode("utf-8")
    ).hexdigest()[:12]
    return f"c-{digest}"


def build_contract(result: DocumentParseResult, contract_id: str = "") -> Contract:
    """Wrap a parse result into the unified contract object used from M3 on.

    The clause tree and source mapping are kept as-is under ``document``; the
    element slots stay empty until M3 fills them.
    """

    return Contract(
        contract_id=contract_id or derive_contract_id(result),
        document=result,
    )
