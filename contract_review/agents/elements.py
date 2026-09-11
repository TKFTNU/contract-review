"""要素抽取节点（统一图中的 M3）: structured elements + mathematical signals.

The node wraps the existing M3 pipeline (``extractor.extract_elements``) so
the unified graph has exactly one place that produces parties/amounts/dates/
subjects with clause and block traceability.

On top of the extraction it emits ``ElementSignal``s — pure mathematical
facts (non-positive amounts, extreme values, reversed date ranges, missing
key elements). Signals are deliberately NOT risk candidates: risk detection
stays with the models. They only (a) cross-validate model findings on the
same clause and (b) flag element-level doubts for the human queue, which is
done by the fuse node.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..extractor import M3Config, extract_elements
from ..llm_client import Transport
from ..models import Contract
from .state import AgentReviewConfig, ElementSignal


_DATE_TEXT = re.compile(
    r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日"
)
_TERM_RANGE = re.compile(
    r"自\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*起?\s*[至到]\s*"
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*止"
)
_EXTREME_AMOUNT = 10_000_000

_MISSING_NOTES = {
    "parties": "未提取到当事人信息",
    "amounts": "未提取到任何金额",
    "dates": "未提取到任何日期",
    "subjects": "未提取到标的描述",
}


def _amount_signals(contract: Contract) -> list[ElementSignal]:
    signals: list[ElementSignal] = []
    for element in contract.elements.amounts:
        try:
            value = float(element.normalized or element.value)
        except ValueError:
            continue
        if value <= 0:
            signals.append(
                ElementSignal(
                    kind="amount",
                    note=f"{element.role or '金额'}为 {element.value} 元（非正数）",
                    clause_id=element.clause_id or "",
                    value=element.value,
                )
            )
        elif value >= _EXTREME_AMOUNT:
            signals.append(
                ElementSignal(
                    kind="amount",
                    note=f"{element.role or '金额'}高达 {element.value} 元",
                    clause_id=element.clause_id or "",
                    value=element.value,
                )
            )
    return signals


def _date_signals(contract: Contract) -> list[ElementSignal]:
    signals: list[ElementSignal] = []
    for clause in contract.document.clauses:
        for match in _TERM_RANGE.finditer(clause.text):
            found = [
                date(int(m["y"]), int(m["m"]), int(m["d"]))
                for m in _DATE_TEXT.finditer(match.group(0))
            ]
            if len(found) == 2 and found[0] > found[1]:
                signals.append(
                    ElementSignal(
                        kind="date",
                        note=(
                            f"期限倒置：起始 {found[0].isoformat()} 晚于"
                            f"结束 {found[1].isoformat()}"
                        ),
                        clause_id=clause.clause_id,
                        value=match.group(0),
                    )
                )
    return signals


def _completeness_signals(contract: Contract) -> list[ElementSignal]:
    elements = contract.elements
    counts = {
        "parties": len(elements.parties),
        "amounts": len(elements.amounts),
        "dates": len(elements.dates),
        "subjects": len(elements.subjects),
    }
    return [
        ElementSignal(kind="completeness", note=note)
        for key, note in _MISSING_NOTES.items()
        if counts[key] == 0
    ]


def extract_contract_elements(
    contract: Contract,
    config: AgentReviewConfig,
    transport: Transport | None = None,
) -> tuple[
    Contract, dict[str, Any], list[ElementSignal], dict[str, Any], list[str]
]:
    """Run M3 inside the unified graph.

    Returns (contract with filled elements, elements payload, signals,
    M3 stats, errors).
    """

    m3_config = M3Config(
        endpoint=config.endpoint,
        model=config.model,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
        disable_thinking=config.disable_thinking,
        max_characters_per_batch=config.max_characters_per_batch,
        enable_rules=True,
        enable_llm=config.enable_llm,
    )
    errors: list[str] = []
    try:
        filled, summary = extract_elements(contract, m3_config, transport)
    except Exception as exc:  # noqa: BLE001 - never block the graph
        errors.append(f"要素抽取失败：{exc}")
        return contract, contract.elements.to_dict(), [], {}, errors
    errors.extend(summary.errors)

    signals = [
        *_amount_signals(filled),
        *_date_signals(filled),
        *_completeness_signals(filled),
    ]
    stats = {
        "clauses": summary.clauses,
        "batches": summary.batches,
        "succeeded_batches": summary.succeeded_batches,
        "rule_elements": summary.rule_elements,
        "llm_elements": summary.llm_elements,
        "merged_elements": summary.merged_elements,
        "by_kind": summary.by_kind,
    }
    return filled, filled.elements.to_dict(), signals, stats, errors
