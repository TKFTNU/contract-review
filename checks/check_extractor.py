from __future__ import annotations

import unittest

from contract_review.extractor import (
    M3Config,
    MONEY_PATTERN,
    extract_by_rules,
    extract_elements,
    merge_elements,
)
from contract_review.llm_client import LLMClientError
from contract_review.models import Element
from contract_review.service import build_contract, parse_contract


CONTRACT_TEXT = (
    "山东省新建商品房买卖合同\n"
    "出卖人（甲方）：张明明；通讯地址：南京市建邺区某某路1号。\n"
    "买受人（乙方）：李芳芳；户籍所在地：中国。\n"
    "第一条 计价方式与价款\n"
    "总价款为人民币（小写）1,200,000.00元。\n"
    "第二条 付款方式及期限\n"
    "首付款人民币（小写）360,000.00元，余款人民币（小写）840,000.00元。\n"
    "第三条 交付时间\n"
    "甲方应于2026年10月11日前交付房屋。\n"
    "第四条 标的\n"
    "坐落于南京市建邺区某某路88号的商品房，建筑面积为141.78平方米。\n"
    "签约日期：2026年9月11日\n"
)


def make_contract():
    return build_contract(
        parse_contract("sample.txt", CONTRACT_TEXT.encode("utf-8"))
    )


class RuleExtractionChecks(unittest.TestCase):
    def test_extracts_parties(self) -> None:
        elements = extract_by_rules(make_contract())
        parties = {element.value: element.role for element in elements if element.kind == "party"}
        self.assertEqual(parties.get("张明明"), "出卖人（甲方）")
        self.assertEqual(parties.get("李芳芳"), "买受人（乙方）")

    def test_extracts_amounts_with_roles_and_normalization(self) -> None:
        elements = extract_by_rules(make_contract())
        amounts = {
            element.normalized: element.role
            for element in elements
            if element.kind == "amount"
        }
        self.assertEqual(amounts.get("1200000"), "合同总价")
        self.assertEqual(amounts.get("360000"), "首付款")
        self.assertEqual(amounts.get("840000"), "余款")

    def test_extracts_zero_amount(self) -> None:
        contract = build_contract(
            parse_contract(
                "zero.txt",
                "城镇房屋租赁合同\n第三条 租金与押金\n月租金为人民币0元，押金为人民币999999元。\n".encode("utf-8"),
            )
        )
        amounts = {
            element.normalized: element.role
            for element in extract_by_rules(contract)
            if element.kind == "amount"
        }
        # Zero rent is a real value (adversarial samples inject it); the
        # 租金 role now comes from the MONEY_ROLES table.
        self.assertEqual(amounts.get("0"), "租金")
        self.assertEqual(amounts.get("999999"), "押金")

    def test_bare_zero_only_matches_before_yuan(self) -> None:
        # A bare "0" must stay inert unless it is immediately followed by 元.
        self.assertIsNone(MONEY_PATTERN.search("编号为20260911号"))
        self.assertIsNone(MONEY_PATTERN.search("共0份文件"))
        self.assertIsNotNone(MONEY_PATTERN.search("月租金为人民币0元"))
        self.assertIsNotNone(MONEY_PATTERN.search("押金为人民币0.00元"))

    def test_extracts_dates_with_roles(self) -> None:
        elements = extract_by_rules(make_contract())
        dates = {
            element.normalized: element.role
            for element in elements
            if element.kind == "date"
        }
        self.assertEqual(dates.get("2026-10-11"), "交付日")
        self.assertEqual(dates.get("2026-09-11"), "签署日")

    def test_extracts_subject_address_and_area(self) -> None:
        elements = extract_by_rules(make_contract())
        subjects = {element.role: element.value for element in elements if element.kind == "subject"}
        self.assertEqual(subjects.get("坐落地址"), "南京市建邺区某某路88号")
        self.assertEqual(subjects.get("建筑面积"), "141.78平方米")

    def test_elements_carry_clause_and_block_traceability(self) -> None:
        contract = make_contract()
        elements = extract_by_rules(contract)
        self.assertTrue(elements)
        block_ids = {block.block_id for block in contract.document.blocks}
        for element in elements:
            self.assertTrue(element.element_id.startswith("E"))
            self.assertEqual(element.source, "m3_rule")
            self.assertIsNotNone(element.clause_id)
            self.assertTrue(set(element.block_ids) <= block_ids)


class MergeChecks(unittest.TestCase):
    def test_keeps_higher_confidence_copy(self) -> None:
        rule = [
            Element(
                element_id="",
                kind="amount",
                value="1,200,000.00",
                normalized="1200000",
                role="合同总价",
                confidence=0.9,
                source="m3_rule",
            )
        ]
        model = [
            Element(
                element_id="",
                kind="amount",
                value="1200000",
                normalized="1200000",
                role="总价",
                confidence=0.8,
                source="m3_llm",
            )
        ]
        merged = merge_elements(rule, model)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].role, "合同总价")
        self.assertEqual(merged[0].source, "m3_rule")

    def test_keeps_model_only_elements(self) -> None:
        rule = [
            Element(
                element_id="",
                kind="party",
                value="张明明",
                confidence=0.9,
                source="m3_rule",
            )
        ]
        model = [
            Element(
                element_id="",
                kind="subject",
                value="商品房（住宅）",
                role="交易标的",
                confidence=0.7,
                source="m3_llm",
            )
        ]
        merged = merge_elements(rule, model)
        self.assertEqual(len(merged), 2)
        self.assertEqual([element.element_id for element in merged], ["E0001", "E0002"])


class ExtractElementsChecks(unittest.TestCase):
    def test_rules_only_when_llm_disabled(self) -> None:
        contract = make_contract()
        result, summary = extract_elements(
            contract, M3Config(enable_llm=False)
        )
        self.assertEqual(summary.batches, 0)
        self.assertGreater(summary.rule_elements, 0)
        self.assertEqual(summary.llm_elements, 0)
        self.assertEqual(
            summary.merged_elements,
            len(result.elements.parties)
            + len(result.elements.amounts)
            + len(result.elements.dates)
            + len(result.elements.subjects),
        )
        self.assertTrue(result.elements.parties)

    def test_llm_adds_missing_subject(self) -> None:
        contract = make_contract()

        def fake_transport(url, payload, headers, timeout):
            self.assertEqual(
                payload["response_format"]["json_schema"]["name"],
                "contract_element_extraction",
            )
            user = payload["messages"][1]["content"]
            self.assertIn("clauses=", user)
            return {
                "parties": [],
                "amounts": [],
                "dates": [],
                "subjects": [
                    {
                        "value": "商品房（住宅）",
                        "normalized": "",
                        "role": "交易标的",
                        "clause_id": "C0001",
                        "source_text": "山东省新建商品房买卖合同",
                        "confidence": 0.78,
                    }
                ],
            }

        result, summary = extract_elements(
            contract, M3Config(model="fake-qwen"), transport=fake_transport
        )
        self.assertEqual(summary.succeeded_batches, 1)
        subjects = {element.value: element.source for element in result.elements.subjects}
        self.assertEqual(subjects.get("商品房（住宅）"), "m3_llm")
        # Rule-made subjects survive alongside the model-only one.
        self.assertGreater(len(result.elements.subjects), 1)

    def test_llm_failure_keeps_rule_elements(self) -> None:
        contract = make_contract()

        def failing_transport(url, payload, headers, timeout):
            raise LLMClientError("模型服务返回HTTP 500：boom")

        result, summary = extract_elements(
            contract, M3Config(model="fake-qwen"), transport=failing_transport
        )
        self.assertTrue(summary.errors)
        self.assertEqual(summary.succeeded_batches, 0)
        self.assertGreater(len(result.elements.parties), 0)
        self.assertGreater(len(result.elements.amounts), 0)


if __name__ == "__main__":
    unittest.main()
