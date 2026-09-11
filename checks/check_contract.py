from __future__ import annotations

import json
import unittest

from contract_review.models import Contract, ContractElements, Element
from contract_review.service import build_contract, parse_contract


SAMPLE_TEXT = "第一条 服务内容\n乙方提供技术服务。\n第二条 合同价款\n价款为100元。"


class ContractModelChecks(unittest.TestCase):
    def test_build_contract_wraps_parse_result(self) -> None:
        result = parse_contract("sample.txt", SAMPLE_TEXT.encode())
        contract = build_contract(result)

        self.assertEqual(contract.schema_version, "m2")
        self.assertTrue(contract.contract_id.startswith("c-"))
        self.assertIs(contract.document, result)

        payload = contract.to_dict()
        self.assertEqual(payload["schema_version"], "m2")
        self.assertEqual(payload["filename"], "sample.txt")
        self.assertEqual(payload["file_type"], "txt")
        self.assertEqual(
            set(payload["elements"]),
            {"parties", "amounts", "dates", "subjects"},
        )
        self.assertEqual(payload["stats"]["clauses"], len(result.clauses))
        self.assertEqual(
            len(payload["document"]["clauses"]), len(result.clauses)
        )
        json.dumps(payload, ensure_ascii=False)

    def test_element_slots_start_empty(self) -> None:
        contract = build_contract(parse_contract("sample.txt", SAMPLE_TEXT.encode()))
        for slot in ("parties", "amounts", "dates", "subjects"):
            self.assertEqual(getattr(contract.elements, slot), [])
        self.assertEqual(contract.elements.to_dict()["parties"], [])

    def test_contract_id_is_stable_and_overridable(self) -> None:
        first = build_contract(parse_contract("a.txt", SAMPLE_TEXT.encode()))
        second = build_contract(parse_contract("a.txt", SAMPLE_TEXT.encode()))
        self.assertEqual(first.contract_id, second.contract_id)

        custom = build_contract(
            parse_contract("a.txt", SAMPLE_TEXT.encode()), contract_id="c-custom"
        )
        self.assertEqual(custom.contract_id, "c-custom")

    def test_document_payload_is_unchanged_by_m2(self) -> None:
        result = parse_contract("sample.txt", SAMPLE_TEXT.encode())
        payload = result.to_dict()
        self.assertIn("stats", payload)
        self.assertNotIn("elements", payload)
        self.assertNotIn("schema_version", payload)

    def test_elements_keep_traceability_fields(self) -> None:
        element = Element(
            element_id="E0001",
            kind="party",
            value="星海科技产业园管理有限公司",
            role="甲方",
            clause_id="C0001",
            block_ids=["B0003"],
            confidence=0.91,
            source="m3_llm",
        )
        contract = Contract(
            contract_id="c-test",
            document=parse_contract("sample.txt", SAMPLE_TEXT.encode()),
            elements=ContractElements(parties=[element]),
        )

        payload = contract.to_dict()
        stored = payload["elements"]["parties"][0]
        self.assertEqual(stored["kind"], "party")
        self.assertEqual(stored["clause_id"], "C0001")
        self.assertEqual(stored["block_ids"], ["B0003"])
        self.assertEqual(stored["confidence"], 0.91)
        self.assertEqual(stored["source"], "m3_llm")


if __name__ == "__main__":
    unittest.main()
