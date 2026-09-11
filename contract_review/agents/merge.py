"""风险候选合并: merge the two review branches into one candidate list.

Duplicates (same code on overlapping clauses, e.g. the rule detector and the
model both flagging zero rent in 第三条) collapse into one candidate that
keeps the highest confidence, the union of evidence, and every source tag.
"""

from __future__ import annotations

from .state import RiskCandidate


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _overlaps(left: RiskCandidate, right: RiskCandidate) -> bool:
    if left.code != right.code:
        return False
    if not left.clause_ids or not right.clause_ids:
        return left.code == right.code
    return bool(set(left.clause_ids) & set(right.clause_ids))


def merge_candidates(
    clause_candidates: list[RiskCandidate],
    cross_candidates: list[RiskCandidate],
) -> list[RiskCandidate]:
    """Fuse overlapping candidates, keeping the stronger copy's fields."""

    merged: list[RiskCandidate] = []
    for candidate in [*clause_candidates, *cross_candidates]:
        target = next(
            (m for m in merged if _overlaps(m, candidate)), None
        )
        if target is None:
            merged.append(candidate)
            continue
        if candidate.confidence > target.confidence:
            target.confidence = candidate.confidence
            target.severity = candidate.severity
            target.message = candidate.message or target.message
        if candidate.source not in target.source:
            target.source = f"{target.source}+{candidate.source}"
        for evidence in candidate.evidence:
            if evidence and evidence not in target.evidence:
                target.evidence.append(evidence)
        for clause_id in candidate.clause_ids:
            if clause_id not in target.clause_ids:
                target.clause_ids.append(clause_id)
        for block_id in candidate.block_ids:
            if block_id not in target.block_ids:
                target.block_ids.append(block_id)

    merged.sort(
        key=lambda c: (
            _SEVERITY_ORDER.get(c.severity, 1),
            c.code,
            c.candidate_id,
        )
    )
    return merged
