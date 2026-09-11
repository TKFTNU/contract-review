from __future__ import annotations

import json
import unittest

from contract_review.agents import AgentReviewConfig, run_unified_review
from contract_review.agents.state import ASSESSMENT_LABELS
from contract_review.legal_kb import RetrievedArticle
from contract_review.llm_client import LLMClientError
from contract_review.service import build_contract, parse_contract


CONTRACT_TEXT = (
    "商品房买卖合同\n"
    "出卖人（甲方）：华信置业公司；买受人（乙方）：启明服务公司。\n"
    "第一条 价款\n"
    "总价款为人民币（小写）1,200,000.00元。\n"
    "第二条 付款\n"
    "首付款人民币（小写）360,000.00元，余款人民币（小写）860,000.00元。\n"
)

# Zero rent lets the M3 rule channel extract a non-positive amount, which
# becomes an element signal on clause C0002.
ZERO_RENT_TEXT = (
    "城镇房屋租赁合同\n"
    "甲方（出租人）：星河数据公司；乙方（承租人）：云图科技公司。\n"
    "第三条 租金\n"
    "月租金为人民币0元。\n"
)

FAKE_ARTICLE = RetrievedArticle(
    article_uid="civil_code_2021_article_705",
    law_name="中华人民共和国民法典",
    article_no="第七百零五条",
    content="租赁期限不得超过二十年。超过二十年的，超过部分无效。",
    risk_tags=["term_over_20y"],
)


def make_contract(text: str = CONTRACT_TEXT):
    return build_contract(
        parse_contract("sample.txt", text.encode("utf-8"))
    )


class FakeKB:
    def search(self, query, contract_types=None, top_k=6):
        return [FAKE_ARTICLE]


def _json_arrays(text: str):
    """Yield every JSON array embedded in a prompt string."""

    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    yield json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    pass
                start = -1


def _candidate_ids(prompt: str) -> list[str]:
    ids: list[str] = []
    for array in _json_arrays(prompt):
        for item in array:
            if isinstance(item, dict) and "candidate_id" in item:
                ids.append(str(item["candidate_id"]))
    return ids


class FakeLLM:
    """Calls by JSON-schema name with per-name failure switches."""

    def __init__(
        self,
        understanding: dict | None = None,
        clause_risks: list[dict] | None = None,
        cross_conflicts: list[dict] | None = None,
        legal_verdict: str = "legal_risk",
        legal_rounds: list[list[dict]] | None = None,
        judge_rounds: list[dict] | None = None,
        arbitrate_rounds: list[dict] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self.understanding = understanding or {
            "contract_type": "商品房买卖",
            "subject_summary": "商品房",
            "payment_summary": "分期付款",
            "key_obligations": [],
            "clause_roles": [],
        }
        self.clause_risks = clause_risks or []
        self.cross_conflicts = cross_conflicts or []
        self.legal_verdict = legal_verdict
        self.legal_rounds = legal_rounds
        self.judge_rounds = judge_rounds
        self.arbitrate_rounds = arbitrate_rounds
        self.fail = fail or set()
        self.calls: dict[str, int] = {}

    def __call__(self, url, payload, headers, timeout):
        name = payload["response_format"]["json_schema"]["name"]
        self.calls[name] = self.calls.get(name, 0) + 1
        if name in self.fail:
            raise LLMClientError(f"{name} 不可用")
        if name == "contract_understanding":
            return dict(self.understanding)
        if name == "contract_element_extraction":
            return {"parties": [], "amounts": [], "dates": [], "subjects": []}
        if name == "clause_risk_review":
            return {"risks": list(self.clause_risks)}
        if name == "cross_clause_conflict_review":
            return {"conflicts": list(self.cross_conflicts)}
        if name == "legal_application_verdict":
            user = payload["messages"][1]["content"]
            ids = _candidate_ids(user)
            if self.legal_rounds is not None:
                index = min(
                    self.calls[name] - 1, len(self.legal_rounds) - 1
                )
                return {"verdicts": list(self.legal_rounds[index])}
            return {
                "verdicts": [
                    {
                        "candidate_id": candidate_id,
                        "assessment": self.legal_verdict,
                        "severity": "high",
                        "reason": "结论依据。",
                        "article_uid": "civil_code_2021_article_705"
                        if self.legal_verdict == "legal_risk"
                        else "",
                        "quote": "租赁期限不得超过二十年。",
                    }
                    for candidate_id in ids
                ]
            }
        if name == "final_adjudication":
            user = payload["messages"][1]["content"]
            if self.judge_rounds:
                index = min(
                    self.calls[name] - 1, len(self.judge_rounds) - 1
                )
                response = dict(self.judge_rounds[index])
                provided = {
                    str(item["candidate_id"])
                    for item in response.get("verdicts", [])
                    if "candidate_id" in item
                }
                for candidate_id in _candidate_ids(user):
                    if candidate_id not in provided:
                        response.setdefault("verdicts", []).append(
                            {
                                "candidate_id": candidate_id,
                                "keep": True,
                                "severity": "medium",
                                "review_required": False,
                                "rationale": "保留。",
                            }
                        )
                return response
            return {
                "verdicts": [
                    {
                        "candidate_id": candidate_id,
                        "keep": True,
                        "severity": "high",
                        "review_required": False,
                        "rationale": "风险成立，保留。",
                    }
                    for candidate_id in _candidate_ids(user)
                ],
                "overall_assessment": "medium_risk",
                "overall_score": 60,
                "summary": "合同整体存在若干风险。",
            }
        if name == "cross_module_arbitration":
            if self.arbitrate_rounds:
                index = min(
                    self.calls[name] - 1, len(self.arbitrate_rounds) - 1
                )
                return dict(self.arbitrate_rounds[index])
            return {"conflicts": [], "has_material_conflict": False}
        raise AssertionError(f"未知 schema：{name}")


def clause_risk(**overrides) -> dict:
    payload = {
        "code": "unfair_term",
        "severity": "high",
        "clause_id": "C0002",
        "reason": "付款安排不平衡。",
        "evidence": "首付款高于余款。",
    }
    payload.update(overrides)
    return payload


class PureLLMChecks(unittest.TestCase):
    def test_understanding_unavailable_without_fallback(self) -> None:
        llm = FakeLLM(fail={"contract_understanding"})
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertEqual(report["understanding"]["source"], "unavailable")
        self.assertTrue(
            any("合同理解" in error for error in report["errors"])
        )

    def test_candidates_are_model_only(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            cross_conflicts=[
                {
                    "code": "amount_conflict",
                    "severity": "high",
                    "clause_ids": ["C0001", "C0002"],
                    "reason": "分项合计与总价不一致。",
                    "evidence": "1,220,000 vs 1,200,000",
                }
            ],
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        sources = {
            source
            for risk in report["risks"]
            for source in risk["agent_sources"]
        }
        self.assertTrue(sources)
        self.assertNotIn("cross_rule", sources)
        self.assertNotIn("clause_rule", sources)


class ElementSignalChecks(unittest.TestCase):
    def test_zero_amount_cross_validates_model_risk(self) -> None:
        llm = FakeLLM(
            clause_risks=[
                clause_risk(
                    code="zero_rent",
                    clause_id="C0002",
                    reason="月租金为零。",
                    evidence="月租金为人民币0元。",
                )
            ],
            legal_verdict="rights_imbalance",
        )
        report = run_unified_review(
            make_contract(ZERO_RENT_TEXT),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertTrue(report["element_signals"])
        risk = next(
            (r for r in report["risks"] if r["code"] == "zero_rent"), None
        )
        self.assertIsNotNone(risk)
        self.assertTrue(risk["cross_validated"])
        self.assertGreaterEqual(risk["confidence"], 0.85)
        self.assertEqual(report["stats"]["cross_validated"], 1)

    def test_unmatched_signal_enters_review_queue(self) -> None:
        # No model risk reported at all: the zero-rent signal still surfaces.
        llm = FakeLLM()
        report = run_unified_review(
            make_contract(ZERO_RENT_TEXT),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        codes = {item["code"] for item in report["review_queue"]}
        self.assertTrue(
            any(code.startswith("element_signal") for code in codes)
        )

    def test_elements_payload_present(self) -> None:
        llm = FakeLLM()
        report = run_unified_review(
            make_contract(ZERO_RENT_TEXT),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertIn("amounts", report["elements"])
        self.assertTrue(report["elements"]["amounts"])
        self.assertIn("merged_elements", report["m3_stats"])


class GuardrailChecks(unittest.TestCase):
    def test_non_violation_assessment_survives_judge_drop(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            legal_verdict="commercial_unreasonable",
            judge_rounds=[
                {
                    "verdicts": [
                        {
                            "candidate_id": "KC0001",
                            "keep": False,
                            "severity": "medium",
                            "review_required": False,
                            "rationale": "不违法，可忽略。",
                        }
                    ],
                    "overall_assessment": "medium_risk",
                    "overall_score": 55,
                    "summary": "存在商业不合理条款。",
                }
            ],
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        risk = next(
            (r for r in report["risks"] if r["code"] == "unfair_term"), None
        )
        self.assertIsNotNone(risk)
        self.assertEqual(risk["category"], "commercial_unreasonable")
        self.assertEqual(
            risk["category_label"],
            ASSESSMENT_LABELS["commercial_unreasonable"],
        )
        self.assertTrue(risk["review_required"])
        self.assertIn("护栏", risk["rationale"])

    def test_no_substantive_risk_may_be_dropped(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            legal_verdict="no_substantive_risk",
            judge_rounds=[
                {
                    "verdicts": [
                        {
                            "candidate_id": "KC0001",
                            "keep": False,
                            "severity": "low",
                            "review_required": False,
                            "rationale": "无实质问题。",
                        }
                    ],
                    "overall_assessment": "low_risk",
                    "overall_score": 88,
                    "summary": "合同无明显问题。",
                }
            ],
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertEqual(report["risks"], [])
        self.assertEqual(report["overall"]["overall_score"], 88)


class JudgeChecks(unittest.TestCase):
    def test_overall_conclusion_present(self) -> None:
        llm = FakeLLM(clause_risks=[clause_risk()])
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        overall = report["overall"]
        self.assertEqual(overall["overall_assessment"], "medium_risk")
        self.assertEqual(overall["overall_score"], 60)
        self.assertTrue(overall["summary"])

    def test_judge_failure_keeps_everything_for_humans(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()], fail={"final_adjudication"}
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertTrue(report["risks"])
        # Every risk is queued; element signals may add extra queue entries.
        queue_risk_ids = {
            item["risk_id"] for item in report["review_queue"]
        }
        risk_ids = {risk["risk_id"] for risk in report["risks"]}
        self.assertTrue(risk_ids <= queue_risk_ids)
        self.assertEqual(
            report["overall"]["overall_assessment"], "unknown"
        )
        self.assertEqual(report["stats"]["loops"]["judge_retries"], 2)


class ArbitrationChecks(unittest.TestCase):
    def test_material_conflict_triggers_one_rerun(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            arbitrate_rounds=[
                {
                    "conflicts": [
                        {
                            "type": "severity_mismatch",
                            "description": "同一问题严重度不一致。",
                            "involved_ids": ["KC0001"],
                            "resolution": "以条款审查的高严重度为准。",
                        }
                    ],
                    "has_material_conflict": True,
                },
                {
                    "conflicts": [],
                    "has_material_conflict": False,
                },
            ],
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertEqual(report["stats"]["loops"]["arbitration_rounds"], 1)
        self.assertEqual(llm.calls["final_adjudication"], 2)
        self.assertEqual(llm.calls["cross_module_arbitration"], 2)
        self.assertTrue(report["risks"])

    def test_arbitration_unavailable_does_not_block(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            fail={"cross_module_arbitration"},
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertTrue(report["risks"])
        self.assertEqual(report["stats"]["loops"]["arbitration_rounds"], 0)
        self.assertEqual(report["conflicts"], [])


class LoopChecks(unittest.TestCase):
    def test_legal_retry_with_wide_query(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            legal_rounds=[
                [],
                [
                    {
                        "candidate_id": "KC0001",
                        "assessment": "legal_risk",
                        "severity": "high",
                        "reason": "违反规定。",
                        "article_uid": "civil_code_2021_article_705",
                        "quote": "。",
                    }
                ],
            ],
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertEqual(report["stats"]["loops"]["legal_retries"], 1)
        risk = next(
            (r for r in report["risks"] if r["code"] == "unfair_term"), None
        )
        self.assertIsNotNone(risk)
        self.assertTrue(risk["citations"])

    def test_evidence_round_trip_reviews_clauses_again(self) -> None:
        llm = FakeLLM(
            clause_risks=[clause_risk()],
            legal_verdict="insufficient_facts",
            judge_rounds=[
                {
                    "verdicts": [],
                    "overall_assessment": "medium_risk",
                    "overall_score": 50,
                    "summary": "证据不足。",
                },
                {
                    "verdicts": [],
                    "overall_assessment": "medium_risk",
                    "overall_score": 55,
                    "summary": "补充后重判。",
                },
            ],
        )
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertEqual(llm.calls.get("clause_risk_review"), 2)
        self.assertEqual(report["stats"]["loops"]["evidence_rounds"], 1)
        self.assertEqual(report["overall"]["overall_score"], 55)

    def test_full_degradation_keeps_queue_and_reports_errors(self) -> None:
        def failing_transport(url, payload, headers, timeout):
            raise LLMClientError("模型不可达")

        report = run_unified_review(
            make_contract(ZERO_RENT_TEXT),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=failing_transport,
        )
        self.assertTrue(report["errors"])
        self.assertEqual(report["understanding"]["source"], "unavailable")
        # The element signal still reaches the human queue.
        self.assertTrue(
            any(
                item["code"].startswith("element_signal")
                for item in report["review_queue"]
            )
        )


class ReportShapeChecks(unittest.TestCase):
    def test_unified_report_shape(self) -> None:
        llm = FakeLLM()
        report = run_unified_review(
            make_contract(),
            AgentReviewConfig(kb_index_dir=None),
            kb=FakeKB(),
            transport=llm,
        )
        self.assertEqual(report["schema_version"], "m6")
        for key in (
            "schema_version",
            "contract_id",
            "understanding",
            "elements",
            "element_signals",
            "m3_stats",
            "risks",
            "conflicts",
            "overall",
            "review_queue",
            "stats",
            "errors",
        ):
            self.assertIn(key, report)
        loops = report["stats"]["loops"]
        for key in (
            "legal_retries",
            "judge_retries",
            "evidence_rounds",
            "arbitration_rounds",
        ):
            self.assertIn(key, loops)


if __name__ == "__main__":
    unittest.main()
