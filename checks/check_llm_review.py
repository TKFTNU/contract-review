from __future__ import annotations

import unittest

from contract_review.llm_review import LLMReviewConfig, review_contract_boundaries
from contract_review.service import parse_contract


class LLMReviewChecks(unittest.TestCase):
    def test_reviews_uncertain_boundaries_and_rebuilds_clauses(self) -> None:
        original = parse_contract(
            "unnumbered.txt",
            "合同名称\n服务内容\n乙方提供技术服务。".encode(),
        )

        def fake_transport(url, payload, headers, timeout):
            self.assertTrue(url.endswith("/v1/chat/completions"))
            self.assertEqual(payload.get("reasoning_effort"), "none")
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["seed"], 42)
            return {
                "boundaries": [
                    {
                        "boundary_id": "D0001",
                        "relation": "NEW_CLAUSE",
                        "confidence": 0.93,
                        "right_block_role": "TITLE",
                        "inferred_title": "服务内容",
                        "reason": "右侧是新的主题标题",
                    },
                    {
                        "boundary_id": "D0002",
                        "relation": "SAME_CLAUSE",
                        "confidence": 0.92,
                        "right_block_role": "BODY",
                        "inferred_title": "",
                        "reason": "右侧是服务内容的正文",
                    },
                ]
            }

        reviewed, summary = review_contract_boundaries(
            original,
            LLMReviewConfig(model="local-qwen"),
            transport=fake_transport,
        )

        self.assertEqual(summary.requested, 2)
        self.assertEqual(summary.accepted, 2)
        self.assertEqual(summary.unresolved, 0)
        self.assertEqual(len(reviewed.clauses), 2)
        self.assertEqual(reviewed.clauses[1].title, "服务内容")
        self.assertIn("乙方提供技术服务", reviewed.clauses[1].body)
        self.assertFalse(
            any(warning.startswith("未识别到明确") for warning in reviewed.warnings)
        )
        self.assertTrue(all(boundary.llm_model == "local-qwen" for boundary in reviewed.boundaries))
        self.assertTrue(all(boundary.review_required for boundary in original.boundaries))

    def test_missing_model_result_stays_unresolved(self) -> None:
        original = parse_contract("unnumbered.txt", "合同名称\n服务内容".encode())

        def empty_transport(url, payload, headers, timeout):
            return {"boundaries": []}

        reviewed, summary = review_contract_boundaries(
            original,
            LLMReviewConfig(model="local-qwen"),
            transport=empty_transport,
        )
        self.assertEqual(summary.accepted, 0)
        self.assertEqual(summary.unresolved, 1)
        self.assertTrue(summary.errors)
        self.assertTrue(reviewed.boundaries[0].review_required)


if __name__ == "__main__":
    unittest.main()
