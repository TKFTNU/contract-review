"""F 统一融合节点（新）: one deduplicated, cross-validated risk view.

Deterministic code, no model calls:

- dedupe by (risk type + overlapping clauses) — the same problem flagged via
  different wording collapses into one entry keeping the strongest fields;
- cross-validation — a risk sharing a clause with an element signal gets
  ``cross_validated=True`` and a small confidence bump (a mathematical fact
  independently confirms it);
- element signals with no matching risk still go to the human queue as
  "element-level doubts" — never silently ignored;
- stable ordering: severity → confidence → number of sources;
- every entry keeps the full evidence chain (clauses + citations + sources).
"""

from __future__ import annotations

from typing import Any

from .state import (
    AgentReviewConfig,
    Conflict,
    ElementSignal,
    FinalRisk,
)


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _overlaps(left: FinalRisk, right: FinalRisk) -> bool:
    if left.code != right.code:
        return False
    if not left.clause_ids or not right.clause_ids:
        return True
    return bool(set(left.clause_ids) & set(right.clause_ids))


def fuse(
    understanding: dict[str, Any],
    elements: dict[str, Any],
    element_signals: list[ElementSignal],
    final_risks: list[FinalRisk],
    conflicts: list[Conflict],
    config: AgentReviewConfig,
) -> tuple[list[FinalRisk], list[dict[str, str]], list[str]]:
    """Fuse everything into (sorted risks, augmented review queue, errors)."""

    errors: list[str] = []

    # ---- 1. cross-validation from element signals (position-based) ----
    signal_clauses = {
        signal.clause_id for signal in element_signals if signal.clause_id
    }
    # Dates/reversed ranges are clause-level facts; amounts come with their
    # clause id as well, so a shared clause id is a meaningful confirmation.
    for risk in final_risks:
        if risk.code in _SIGNAL_CODE_HINTS and set(risk.clause_ids) & signal_clauses:
            risk.cross_validated = True
            risk.confidence = min(1.0, round(risk.confidence + 0.1, 2))

    # ---- 2. dedupe same-code risks on overlapping clauses ----
    merged: list[FinalRisk] = []
    for risk in final_risks:
        target = next((m for m in merged if _overlaps(m, risk)), None)
        if target is None:
            merged.append(risk)
            continue
        if _SEVERITY_RANK.get(risk.severity, 1) < _SEVERITY_RANK.get(
            target.severity, 1
        ):
            target.severity = risk.severity
        target.confidence = max(target.confidence, risk.confidence)
        target.cross_validated = target.cross_validated or risk.cross_validated
        target.review_required = target.review_required or risk.review_required
        for citation in risk.citations:
            if citation.article_no not in {
                c.article_no for c in target.citations
            }:
                target.citations.append(citation)
        for clause_id in risk.clause_ids:
            if clause_id not in target.clause_ids:
                target.clause_ids.append(clause_id)
        for block_id in risk.block_ids:
            if block_id not in target.block_ids:
                target.block_ids.append(block_id)
        for source in risk.agent_sources:
            if source not in target.agent_sources:
                target.agent_sources.append(source)
        for snippet in risk.evidence:
            if snippet not in target.evidence:
                target.evidence.append(snippet)
        target.message = target.message or risk.message
        target.rationale = target.rationale or risk.rationale

    # ---- 3. stable ordering ----
    merged.sort(
        key=lambda risk: (
            _SEVERITY_RANK.get(risk.severity, 1),
            -risk.confidence,
            -len(risk.agent_sources),
            risk.code,
        )
    )
    for index, risk in enumerate(merged, start=1):
        risk.risk_id = f"R{index:04d}"

    # ---- 4. review queue: risks + unmatched element doubts ----
    queue: list[dict[str, str]] = [
        {
            "risk_id": risk.risk_id,
            "code": risk.code,
            "reason": risk.rationale or "需人工确认。",
        }
        for risk in merged
        if risk.review_required
    ]
    risk_clauses = {cid for risk in merged for cid in risk.clause_ids}
    for signal in element_signals:
        if signal.clause_id and signal.clause_id in risk_clauses:
            continue  # already reflected in a reported risk
        queue.append(
            {
                "risk_id": "—",
                "code": f"element_signal:{signal.kind}",
                "clause_id": signal.clause_id,
                "reason": signal.note
                + (f"（条款 {signal.clause_id}）" if signal.clause_id else ""),
            }
        )

    return merged, queue, errors


# Codes a mathematical signal can meaningfully confirm (amount/date facts
# against amount/date risk families). Anything else still merges by clause
# overlap only — this list gates the confidence bump, not inclusion.
_SIGNAL_CODE_HINTS = {
    "zero_rent",
    "negative_fee",
    "extreme_rent",
    "deposit_over_20pct",
    "date_conflict",
    "date_reversed",
    "term_over_20y",
    "missing_element",
    "invalid_date_range",
}
