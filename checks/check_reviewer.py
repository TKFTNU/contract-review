from __future__ import annotations

import json
import unittest

from contract_review.llm_client import LLMClientError
from contract_review.reviewer import M4Config, review_contract
from contract_review.service import build_contract, parse_contract


def make_contract(text: str):
    return build_contract(parse_contract("sample.txt", text.encode("utf-8")))


AMOUNT_CLEAN = (
    "第一条 价款\n"
    "总价款为人民币（小写）1,200,000.00元。\n"
    "第二条 付款\n"
    "首付款人民币（小写）360,000.00元，余款人民币（小写）840,000.00元。\n"
)
AMOUNT_CONFLICT = AMOUNT_CLEAN.replace("840,000.00", "860,000.00")

DATE_CLEAN = (
    "第一条 交付\n"
    "甲方应于2026年11月28日前交付房屋。\n"
    "约定交付日：2026年11月28日\n"
)
DATE_CONFLICT = DATE_CLEAN.replace("11月28日前", "12月13日前")

PARTY_CLEAN = (
    "甲方（出租人）：星河数据公司；乙方（承租人）：云图科技公司。\n"
    "甲方（签章）：星河数据公司    乙方（签章）：云图科技公司\n"
)
PARTY_CONFLICT = PARTY_CLEAN.replace("乙方（签章）：云图科技公司", "乙方（签章）：林云图")

REFERENCE_CLEAN = (
    "第九条 税费与登记\n"
    "相关配套约定见附件四。\n"
    "附件一 房屋平面图\n"
    "房屋坐落信息。\n"
    "附件四 付款约定\n"
    "付款细节。\n"
)
REFERENCE_CONFLICT = REFERENCE_CLEAN.replace("见附件四。", "见附件九。")


class RuleCheckerChecks(unittest.TestCase):
    def test_amount_conflict_detected(self) -> None:
        report, summary = review_contract(make_contract(AMOUNT_CONFLICT), M4Config(enable_llm=False))
        codes = [finding.code for finding in report.findings]
        self.assertIn("amount_conflict", codes)
        finding = next(f for f in report.findings if f.code == "amount_conflict")
        self.assertIn("1,220,000.00", finding.message)
        self.assertIn("1,200,000.00", finding.message)
        self.assertTrue(finding.clause_ids)

    def test_amount_consistent_passes(self) -> None:
        report, _ = review_contract(make_contract(AMOUNT_CLEAN), M4Config(enable_llm=False))
        self.assertFalse([f for f in report.findings if f.code == "amount_conflict"])

    def test_date_conflict_detected(self) -> None:
        report, _ = review_contract(make_contract(DATE_CONFLICT), M4Config(enable_llm=False))
        conflicts = [f for f in report.findings if f.code == "date_conflict"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("2026-11-28", conflicts[0].message)
        self.assertIn("2026-12-13", conflicts[0].message)

    def test_date_consistent_passes(self) -> None:
        report, _ = review_contract(make_contract(DATE_CLEAN), M4Config(enable_llm=False))
        self.assertFalse([f for f in report.findings if f.code == "date_conflict"])

    def test_term_reversal_detected(self) -> None:
        text = (
            "第二条 租赁期限\n"
            "租赁期限自2026年12月1日起至2026年11月30日止，乙方同意期限倒置仍然有效。\n"
        )
        report, _ = review_contract(make_contract(text), M4Config(enable_llm=False))
        conflicts = [f for f in report.findings if f.code == "date_conflict"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("2026-12-01", conflicts[0].message)
        self.assertIn("2026-11-30", conflicts[0].message)
        self.assertIn("自2026年12月1日起至2026年11月30日止", conflicts[0].evidence[0])
        self.assertTrue(conflicts[0].clause_ids)

    def test_term_reversal_summary_line_detected(self) -> None:
        text = "期限 | 2026年12月1日—2026年11月30日\n"
        report, _ = review_contract(make_contract(text), M4Config(enable_llm=False))
        conflicts = [f for f in report.findings if f.code == "date_conflict"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("2026-12-01", conflicts[0].message)

    def test_term_range_normal_order_passes(self) -> None:
        text = "第二条 租赁期限\n租赁期限自2026年1月1日起至2026年12月31日止。\n"
        report, _ = review_contract(make_contract(text), M4Config(enable_llm=False))
        self.assertFalse([f for f in report.findings if f.code == "date_conflict"])

    def test_reversed_dates_without_term_context_passes(self) -> None:
        # Two plain dates out of order but no term wording must stay silent.
        text = "第一条 说明\n备忘日期为2026年12月1日，参考日期为2026年11月30日。\n"
        report, _ = review_contract(make_contract(text), M4Config(enable_llm=False))
        self.assertFalse([f for f in report.findings if f.code == "date_conflict"])

    def test_party_conflict_detected(self) -> None:
        report, _ = review_contract(make_contract(PARTY_CONFLICT), M4Config(enable_llm=False))
        conflicts = [f for f in report.findings if f.code == "party_conflict"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("乙方", conflicts[0].message)
        self.assertEqual(conflicts[0].severity, "high")

    def test_party_consistent_passes(self) -> None:
        report, _ = review_contract(make_contract(PARTY_CLEAN), M4Config(enable_llm=False))
        self.assertFalse([f for f in report.findings if f.code == "party_conflict"])

    def test_reference_conflict_detected(self) -> None:
        report, _ = review_contract(make_contract(REFERENCE_CONFLICT), M4Config(enable_llm=False))
        conflicts = [f for f in report.findings if f.code == "reference_conflict"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn("附件九", conflicts[0].message)
        self.assertIn("[1, 4]", conflicts[0].message)

    def test_reference_defined_passes(self) -> None:
        report, _ = review_contract(make_contract(REFERENCE_CLEAN), M4Config(enable_llm=False))
        self.assertFalse([f for f in report.findings if f.code == "reference_conflict"])

    def test_clean_contracts_produce_no_findings(self) -> None:
        for text in (AMOUNT_CLEAN, DATE_CLEAN, PARTY_CLEAN, REFERENCE_CLEAN):
            report, _ = review_contract(make_contract(text), M4Config(enable_llm=False))
            self.assertEqual(report.findings, [])


class LLMReviewChecks(unittest.TestCase):
    def test_llm_findings_merged_with_rules(self) -> None:
        contract = make_contract(AMOUNT_CONFLICT)

        def fake_transport(url, payload, headers, timeout):
            self.assertEqual(
                payload["response_format"]["json_schema"]["name"],
                "contract_consistency_review",
            )
            return {
                "findings": [
                    {
                        "code": "semantic_conflict",
                        "severity": "medium",
                        "message": "第五条逾期违约金与第八条责任上限互相矛盾。",
                        "evidence": "第五条…第八条…",
                        "clause_ids": ["C0001"],
                        "confidence": 0.81,
                    }
                ]
            }

        report, summary = review_contract(
            contract, M4Config(model="fake-qwen"), transport=fake_transport
        )
        codes = [finding.code for finding in report.findings]
        self.assertIn("amount_conflict", codes)
        self.assertIn("semantic_conflict", codes)
        self.assertGreater(summary.llm_findings, 0)
        self.assertEqual(summary.findings, len(report.findings))
        self.assertTrue(all(f.finding_id for f in report.findings))

    def test_llm_failure_keeps_rule_findings(self) -> None:
        contract = make_contract(AMOUNT_CONFLICT)

        def failing_transport(url, payload, headers, timeout):
            raise LLMClientError("模型服务返回HTTP 500：boom")

        report, summary = review_contract(
            contract, M4Config(model="fake-qwen"), transport=failing_transport
        )
        self.assertTrue(summary.errors)
        self.assertEqual(summary.succeeded_batches, 0)
        self.assertTrue(
            [finding for finding in report.findings if finding.source == "m4_rule"]
        )

    def test_report_serializes(self) -> None:
        report, _ = review_contract(make_contract(REFERENCE_CONFLICT), M4Config(enable_llm=False))
        payload = report.to_dict()
        self.assertEqual(payload["schema_version"], "m4")
        self.assertEqual(payload["findings"][0]["code"], "reference_conflict")
        json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
