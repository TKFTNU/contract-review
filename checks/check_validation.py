from __future__ import annotations

import unittest

from contract_review.models import BoundaryDecision, Clause, SourceBlock
from contract_review.validation import (
    validate_structure,
    validate_structure_issues,
)


def make_block(block_id: str, text: str = "正文内容。") -> SourceBlock:
    return SourceBlock(
        block_id=block_id,
        order=int(block_id[1:]),
        kind="paragraph",
        text=text,
    )


def make_clause(
    clause_id: str,
    block_ids: list[str],
    label: str = "第一条",
    level: int = 2,
    parent_id: str | None = None,
) -> Clause:
    return Clause(
        clause_id=clause_id,
        label=label,
        title="",
        level=level,
        heading="",
        body="",
        text="",
        parent_id=parent_id,
        block_ids=block_ids,
    )


def make_boundary(
    index: int,
    left: str,
    right: str,
    confidence: float = 0.95,
) -> BoundaryDecision:
    return BoundaryDecision(
        boundary_id=f"D{index:04d}",
        left_block_id=left,
        right_block_id=right,
        relation="NEW_CLAUSE",
        confidence=confidence,
        decision_source="llm_split",
    )


class StructureIssueChecks(unittest.TestCase):
    def test_clean_structure_has_no_issues(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001"), make_block("B0002")],
            [make_clause("C0001", ["B0001", "B0002"])],
            [make_boundary(1, "B0001", "B0002")],
        )
        self.assertEqual(issues, [])

    def test_uncovered_block_is_located_and_refixable(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001"), make_block("B0002"), make_block("B0003")],
            [make_clause("C0001", ["B0001"]), make_clause("C0002", ["B0002"])],
            [make_boundary(1, "B0001", "B0002"), make_boundary(2, "B0002", "B0003")],
        )
        uncovered = [issue for issue in issues if issue.code == "uncovered_block"]
        self.assertEqual(len(uncovered), 1)
        self.assertEqual(uncovered[0].severity, "error")
        self.assertTrue(uncovered[0].refixable)
        self.assertEqual(uncovered[0].block_ids, ["B0003"])

    def test_duplicated_block_is_located(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001"), make_block("B0002")],
            [
                make_clause("C0001", ["B0001", "B0002"]),
                make_clause("C0002", ["B0002"], label="第二条"),
            ],
            [make_boundary(1, "B0001", "B0002")],
        )
        duplicated = [issue for issue in issues if issue.code == "duplicated_block"]
        self.assertEqual(len(duplicated), 1)
        self.assertTrue(duplicated[0].refixable)
        self.assertEqual(duplicated[0].block_ids, ["B0002"])
        self.assertIn("C0002", duplicated[0].clause_ids)

    def test_orphan_reference_is_not_refixable(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001")],
            [make_clause("C0001", ["B0001", "B9999"])],
            [],
        )
        orphan = [issue for issue in issues if issue.code == "orphan_block_reference"]
        self.assertEqual(len(orphan), 1)
        self.assertFalse(orphan[0].refixable)
        self.assertEqual(orphan[0].block_ids, ["B9999"])

    def test_missing_parent_is_refixable(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001")],
            [make_clause("C0001", ["B0001"], level=3, parent_id="C9999")],
            [],
        )
        missing = [issue for issue in issues if issue.code == "missing_parent"]
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].refixable)
        self.assertEqual(missing[0].clause_ids, ["C0001"])

    def test_duplicate_label_is_warning_and_refixable(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001"), make_block("B0002")],
            [
                make_clause("C0001", ["B0001"], label="第一条"),
                make_clause("C0002", ["B0002"], label="第一条"),
            ],
            [make_boundary(1, "B0001", "B0002")],
        )
        duplicates = [issue for issue in issues if issue.code == "duplicate_label"]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0].severity, "warning")
        self.assertTrue(duplicates[0].refixable)
        self.assertEqual(duplicates[0].clause_ids, ["C0001", "C0002"])

    def test_placeholder_labels_are_not_duplicates(self) -> None:
        issues = validate_structure_issues(
            [make_block("B0001"), make_block("B0002")],
            [
                make_clause("C0001", ["B0001"], label="未编号"),
                make_clause("C0002", ["B0002"], label="未编号"),
            ],
            [make_boundary(1, "B0001", "B0002")],
        )
        self.assertFalse(
            [issue for issue in issues if issue.code == "duplicate_label"]
        )

    def test_confidence_threshold_controls_low_confidence_issue(self) -> None:
        blocks = [make_block("B0001"), make_block("B0002")]
        clauses = [make_clause("C0001", ["B0001", "B0002"])]
        boundaries = [make_boundary(1, "B0001", "B0002", confidence=0.85)]

        default = validate_structure_issues(blocks, clauses, boundaries)
        relaxed = validate_structure_issues(blocks, clauses, boundaries, 0.50)

        self.assertTrue(
            [issue for issue in default if issue.code == "low_confidence_boundary"]
        )
        self.assertFalse(
            [issue for issue in relaxed if issue.code == "low_confidence_boundary"]
        )

    def test_wrapper_keeps_returning_messages(self) -> None:
        blocks = [make_block("B0001"), make_block("B0002")]
        clauses = [make_clause("C0001", ["B0001"])]
        boundaries = [make_boundary(1, "B0001", "B0002")]

        messages = validate_structure(blocks, clauses, boundaries)
        issues = validate_structure_issues(blocks, clauses, boundaries)

        self.assertTrue(all(isinstance(message, str) for message in messages))
        self.assertEqual(messages, [issue.message for issue in issues])


if __name__ == "__main__":
    unittest.main()
