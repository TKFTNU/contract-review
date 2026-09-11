from __future__ import annotations

import unittest

from contract_review.llm_client import LLMClientError
from contract_review.llm_splitter import (
    LLMSplitConfig,
    _build_batches,
    split_contract_with_llm,
)
from contract_review.models import BoundaryDecision, SourceBlock
from contract_review.service import parse_contract


class LLMSplitterChecks(unittest.TestCase):
    def test_splits_unnumbered_headings_into_clauses(self) -> None:
        original = parse_contract(
            "unnumbered-headings.txt",
            (
                "合同名称\n"
                "第一条 服务内容\n"
                "乙方提供服务。\n"
                "通知与送达\n"
                "地址变更应当书面通知。"
            ).encode(),
        )
        # The rule engine cannot see a new clause in the unnumbered heading.
        self.assertEqual(len(original.clauses), 2)

        def fake_transport(url, payload, headers, timeout):
            self.assertTrue(url.endswith("/v1/chat/completions"))
            self.assertEqual(payload.get("reasoning_effort"), "none")
            self.assertEqual(
                payload["response_format"]["json_schema"]["name"],
                "contract_clause_split",
            )
            return {
                "clauses": [
                    {
                        "label": "",
                        "title": "合同首部",
                        "level": 1,
                        "block_ids": ["B0001"],
                        "confidence": 0.92,
                    },
                    {
                        "label": "第一条",
                        "title": "服务内容",
                        "level": 2,
                        "block_ids": ["B0002", "B0003"],
                        "confidence": 0.96,
                    },
                    {
                        "label": "",
                        "title": "通知与送达",
                        "level": 2,
                        "block_ids": ["B0004", "B0005"],
                        "confidence": 0.90,
                    },
                ]
            }

        reviewed, summary = split_contract_with_llm(
            original,
            LLMSplitConfig(model="fake-qwen"),
            transport=fake_transport,
        )

        self.assertEqual(summary.succeeded_batches, 1)
        self.assertEqual(summary.rule_clauses, 2)
        self.assertEqual(summary.llm_clauses, 3)
        self.assertTrue(summary.changed)
        self.assertEqual([clause.title for clause in reviewed.clauses][1:], ["服务内容", "通知与送达"])
        self.assertEqual(reviewed.clauses[1].block_ids, ["B0002", "B0003"])
        self.assertEqual(reviewed.clauses[1].parent_id, "C0001")
        self.assertEqual(reviewed.clauses[2].parent_id, "C0001")
        self.assertEqual(reviewed.clauses[1].body.strip(), "乙方提供服务。")
        self.assertFalse(
            any(warning.startswith("[结构校验] 有") for warning in reviewed.warnings)
        )

    def test_repairs_missing_and_duplicated_block_ids(self) -> None:
        original = parse_contract(
            "repair.txt",
            (
                "第一条 甲内容\n"
                "乙方应当提供。\n"
                "第二条 乙内容\n"
                "丙方应当支付。"
            ).encode(),
        )

        def fake_transport(url, payload, headers, timeout):
            return {
                "clauses": [
                    {
                        "label": "第一条",
                        "title": "甲内容",
                        "level": 2,
                        "block_ids": ["B0001", "B0002", "B0002"],
                        "confidence": 0.9,
                    },
                    {
                        "label": "第二条",
                        "title": "乙内容",
                        "level": 2,
                        "block_ids": ["B0004"],
                        "confidence": 0.9,
                    },
                ]
            }

        reviewed, summary = split_contract_with_llm(
            original,
            LLMSplitConfig(model="fake-qwen"),
            transport=fake_transport,
        )

        self.assertEqual(summary.llm_clauses, 2)
        referenced = [
            block_id for clause in reviewed.clauses for block_id in clause.block_ids
        ]
        self.assertEqual(sorted(referenced), ["B0001", "B0002", "B0003", "B0004"])
        # The dropped B0003 stays with the clause it belongs to.
        self.assertEqual(reviewed.clauses[0].block_ids, ["B0001", "B0002", "B0003"])
        self.assertEqual(reviewed.clauses[1].block_ids, ["B0004"])

    def test_keeps_rule_result_when_all_batches_fail(self) -> None:
        original = parse_contract(
            "offline.txt",
            "第一条 服务内容\n乙方提供服务。".encode(),
        )

        def failing_transport(url, payload, headers, timeout):
            raise LLMClientError("无法连接模型服务：connection refused")

        reviewed, summary = split_contract_with_llm(
            original,
            LLMSplitConfig(model="fake-qwen"),
            transport=failing_transport,
        )

        self.assertEqual(summary.succeeded_batches, 0)
        self.assertTrue(summary.errors)
        self.assertEqual(
            [clause.label for clause in reviewed.clauses],
            [clause.label for clause in original.clauses],
        )
        self.assertEqual(reviewed.clauses[0].block_ids, original.clauses[0].block_ids)

    def test_batches_cut_at_confident_clause_starts(self) -> None:
        blocks = [
            SourceBlock(f"B{index:04d}", index, "paragraph", f"第{index}条 标题{index}")
            for index in range(1, 8)
        ]
        boundaries = [
            BoundaryDecision(
                boundary_id=f"D{index:04d}",
                left_block_id=f"B{index:04d}",
                right_block_id=f"B{index + 1:04d}",
                relation="NEW_CLAUSE",
                confidence=0.98,
                decision_source="textual_numbering",
            )
            for index in range(1, 7)
        ]

        batches = _build_batches(
            blocks, boundaries, LLMSplitConfig(max_blocks_per_batch=3)
        )

        self.assertEqual([len(batch.targets) for batch in batches], [3, 3, 1])
        self.assertEqual(batches[1].targets[0].block_id, "B0004")
        # The second batch keeps the tail of the first one as context only.
        self.assertEqual(
            [block.block_id for block in batches[1].context],
            ["B0001", "B0002", "B0003"],
        )


class ReflowLoopChecks(unittest.TestCase):
    """The C to L reflow loop: repair, roll back, and stay off when disabled."""

    def test_reflow_repairs_duplicate_labels(self) -> None:
        original = parse_contract(
            "duplicate-labels.txt",
            "第一条 甲内容\n乙方提供。\n第一条 乙内容\n丙方支付。".encode(),
        )

        def fake_transport(url, payload, headers, timeout):
            user = payload["messages"][1]["content"]
            if "存在下列结构问题" in user:
                return {
                    "clauses": [
                        {
                            "label": "第一条",
                            "title": "甲内容",
                            "level": 2,
                            "block_ids": ["B0001", "B0002"],
                            "confidence": 0.95,
                        },
                        {
                            "label": "第二条",
                            "title": "乙内容",
                            "level": 2,
                            "block_ids": ["B0003", "B0004"],
                            "confidence": 0.95,
                        },
                    ]
                }
            return {
                "clauses": [
                    {
                        "label": "第一条",
                        "title": "甲内容",
                        "level": 2,
                        "block_ids": ["B0001", "B0002"],
                        "confidence": 0.95,
                    },
                    {
                        "label": "第一条",
                        "title": "乙内容",
                        "level": 2,
                        "block_ids": ["B0003", "B0004"],
                        "confidence": 0.95,
                    },
                ]
            }

        reviewed, summary = split_contract_with_llm(
            original, LLMSplitConfig(model="fake-qwen"), transport=fake_transport
        )

        self.assertEqual(summary.issues_before, 1)
        self.assertEqual(summary.issues_after, 0)
        self.assertEqual(summary.reflow_rounds, 1)
        self.assertGreater(summary.reflowed_blocks, 0)
        self.assertEqual(
            [clause.label for clause in reviewed.clauses], ["第一条", "第二条"]
        )
        self.assertTrue(any("已采纳" in note for note in summary.reflow_notes))

    def test_reflow_rolls_back_when_not_improved(self) -> None:
        original = parse_contract(
            "rollback.txt", "第一条 甲内容\n乙方提供。".encode()
        )

        def fake_transport(url, payload, headers, timeout):
            return {
                "clauses": [
                    {
                        "label": "第一条",
                        "title": "甲内容",
                        "level": 2,
                        "block_ids": ["B0001"],
                        "confidence": 0.50,
                    },
                    {
                        "label": "未编号",
                        "title": "",
                        "level": 2,
                        "block_ids": ["B0002"],
                        "confidence": 0.50,
                    },
                ]
            }

        reviewed, summary = split_contract_with_llm(
            original, LLMSplitConfig(model="fake-qwen"), transport=fake_transport
        )

        self.assertEqual(summary.issues_before, 1)
        self.assertEqual(summary.issues_after, 1)
        # The round ran but was rejected, so the result must stay unchanged.
        self.assertEqual(summary.reflow_rounds, 1)
        self.assertGreater(summary.reflowed_blocks, 0)
        self.assertTrue(any("未减少" in note for note in summary.reflow_notes))
        self.assertEqual(len(reviewed.clauses), 2)

    def test_reflow_disabled_stays_single_pass(self) -> None:
        original = parse_contract(
            "no-reflow.txt", "第一条 甲内容\n乙方提供。".encode()
        )
        calls: list[dict] = []

        def fake_transport(url, payload, headers, timeout):
            calls.append(payload)
            return {
                "clauses": [
                    {
                        "label": "第一条",
                        "title": "甲内容",
                        "level": 2,
                        "block_ids": ["B0001"],
                        "confidence": 0.60,
                    },
                    {
                        "label": "第二条",
                        "title": "乙内容",
                        "level": 2,
                        "block_ids": ["B0002"],
                        "confidence": 0.60,
                    },
                ]
            }

        reviewed, summary = split_contract_with_llm(
            original,
            LLMSplitConfig(model="fake-qwen", max_reflow_rounds=0),
            transport=fake_transport,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(summary.reflow_rounds, 0)
        self.assertEqual(summary.reflow_notes, [])
        self.assertEqual(summary.issues_before, 1)
        self.assertEqual(len(reviewed.clauses), 2)


if __name__ == "__main__":
    unittest.main()
