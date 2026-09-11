from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .models import BoundaryDecision, Clause, SourceBlock
from .splitter import AUTO_ACCEPT_CONFIDENCE


DECIMAL_PATH = re.compile(r"^\d+(?:\.\d+)+$")

# Labels that carry no real clause number, so repeats are not duplicates.
NON_LABELS = {"", "前言", "标题", "未编号"}


@dataclass(slots=True)
class StructureIssue:
    """A machine-readable consistency problem found after clause assembly.

    ``block_ids`` and ``clause_ids`` locate the damage so the C to L reflow loop
    can re-annotate only the affected window instead of the whole contract, and
    ``refixable`` marks the issues a model annotation can realistically repair.
    """

    code: str
    message: str
    severity: str = "warning"
    block_ids: list[str] = field(default_factory=list)
    clause_ids: list[str] = field(default_factory=list)
    refixable: bool = False


def validate_structure_issues(
    blocks: list[SourceBlock],
    clauses: list[Clause],
    boundaries: list[BoundaryDecision],
    confidence_threshold: float = AUTO_ACCEPT_CONFIDENCE,
) -> list[StructureIssue]:
    """Check traceability and global consistency after local boundary decisions."""

    issues: list[StructureIssue] = []
    block_ids = [block.block_id for block in blocks]
    block_id_set = set(block_ids)

    if len(block_ids) != len(block_id_set):
        repeats = sorted(
            block_id
            for block_id, count in Counter(block_ids).items()
            if count > 1
        )
        issues.append(
            StructureIssue(
                code="duplicate_block_id",
                message="检测到重复的原文区块ID，来源定位可能不可靠。",
                severity="error",
                block_ids=repeats,
            )
        )

    referenced_ids = [block_id for clause in clauses for block_id in clause.block_ids]
    missing_ids = sorted(set(referenced_ids) - block_id_set)
    if missing_ids:
        preview = "、".join(missing_ids[:5])
        offenders = [
            clause.clause_id
            for clause in clauses
            if any(block_id in set(missing_ids) for block_id in clause.block_ids)
        ]
        issues.append(
            StructureIssue(
                code="orphan_block_reference",
                message=f"条款引用了不存在的原文区块：{preview}。",
                severity="error",
                block_ids=missing_ids,
                clause_ids=offenders,
            )
        )

    reference_counts = Counter(referenced_ids)
    uncovered = [block_id for block_id in block_ids if reference_counts[block_id] == 0]
    duplicated = sorted(
        block_id for block_id, count in reference_counts.items() if count > 1
    )
    if uncovered:
        issues.append(
            StructureIssue(
                code="uncovered_block",
                message=f"有{len(uncovered)}个原文区块未进入任何条款。",
                severity="error",
                block_ids=uncovered,
                refixable=True,
            )
        )
    if duplicated:
        duplicated_set = set(duplicated)
        offenders = [
            clause.clause_id
            for clause in clauses
            if any(block_id in duplicated_set for block_id in clause.block_ids)
        ]
        issues.append(
            StructureIssue(
                code="duplicated_block",
                message=f"有{len(duplicated)}个原文区块被重复归入条款。",
                severity="error",
                block_ids=duplicated,
                clause_ids=offenders,
                refixable=True,
            )
        )

    if len(boundaries) != max(0, len(blocks) - 1):
        issues.append(
            StructureIssue(
                code="boundary_mismatch",
                message="边界决策数量与原文区块数量不匹配。",
                severity="error",
            )
        )
    for index, boundary in enumerate(boundaries[: max(0, len(blocks) - 1)], start=1):
        expected_left = blocks[index - 1].block_id
        expected_right = blocks[index].block_id
        if (
            boundary.left_block_id != expected_left
            or boundary.right_block_id != expected_right
        ):
            issues.append(
                StructureIssue(
                    code="boundary_order_mismatch",
                    message=f"边界{boundary.boundary_id}与原文顺序不一致。",
                    severity="error",
                    block_ids=[expected_left, expected_right],
                )
            )
            break

    clauses_by_id = {clause.clause_id: clause for clause in clauses}
    for clause in clauses:
        if clause.parent_id is None:
            continue
        parent = clauses_by_id.get(clause.parent_id)
        if parent is None:
            issues.append(
                StructureIssue(
                    code="missing_parent",
                    message=f"{clause.clause_id}引用了不存在的父条款。",
                    severity="error",
                    block_ids=list(clause.block_ids),
                    clause_ids=[clause.clause_id],
                    refixable=True,
                )
            )
        elif parent.level >= clause.level:
            issues.append(
                StructureIssue(
                    code="parent_level_invalid",
                    message=f"{clause.clause_id}的父子层级关系异常。",
                    severity="error",
                    block_ids=list(clause.block_ids),
                    clause_ids=[clause.clause_id, parent.clause_id],
                    refixable=True,
                )
            )

    seen_decimal_labels: set[str] = set()
    seen_labels: Counter[str] = Counter()
    for clause in clauses:
        normalized_label = clause.label.rstrip(".、").strip()
        if normalized_label not in NON_LABELS:
            seen_labels[normalized_label] += 1
        if DECIMAL_PATH.fullmatch(normalized_label):
            parent_label = normalized_label.rsplit(".", 1)[0]
            # A clause already linked in the tree (for example 1.1 under
            # "第一条") needs no separate "1." sibling to be valid.
            if parent_label not in seen_decimal_labels and clause.parent_id is None:
                issues.append(
                    StructureIssue(
                        code="decimal_parent_missing",
                        message=(
                            f"{clause.label}未找到编号父级{parent_label}，"
                            "请检查层级或原文缺失。"
                        ),
                        severity="warning",
                        block_ids=list(clause.block_ids),
                        clause_ids=[clause.clause_id],
                        refixable=True,
                    )
                )
            seen_decimal_labels.add(normalized_label)
        elif normalized_label.isdigit():
            seen_decimal_labels.add(normalized_label)

    duplicate_labels = [label for label, count in seen_labels.items() if count > 1]
    if duplicate_labels:
        preview = "、".join(duplicate_labels[:5])
        duplicated_label_set = set(duplicate_labels)
        offenders = [
            clause
            for clause in clauses
            if clause.label.rstrip(".、").strip() in duplicated_label_set
        ]
        issues.append(
            StructureIssue(
                code="duplicate_label",
                message=f"检测到重复条款编号：{preview}。",
                severity="warning",
                block_ids=[
                    block_id for clause in offenders for block_id in clause.block_ids
                ],
                clause_ids=[clause.clause_id for clause in offenders],
                refixable=True,
            )
        )

    low_confidence = [
        boundary
        for boundary in boundaries
        if boundary.confidence < confidence_threshold
    ]
    if low_confidence:
        affected = sorted(
            {
                block_id
                for boundary in low_confidence
                for block_id in (boundary.left_block_id, boundary.right_block_id)
            }
        )
        issues.append(
            StructureIssue(
                code="low_confidence_boundary",
                message=(
                    f"有{len(low_confidence)}个低置信度边界等待语义模型或人工复核。"
                ),
                severity="warning",
                block_ids=affected,
                refixable=True,
            )
        )
    return issues


def validate_structure(
    blocks: list[SourceBlock],
    clauses: list[Clause],
    boundaries: list[BoundaryDecision],
) -> list[str]:
    """Backwards-compatible wrapper returning human-readable messages only."""

    return [
        issue.message
        for issue in validate_structure_issues(blocks, clauses, boundaries)
    ]
