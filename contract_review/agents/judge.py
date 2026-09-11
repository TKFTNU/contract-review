"""⑤ 最终裁判与复核Agent: the model adjudicates; code only guards.

The model decides, per candidate, whether to keep it, its final severity and
whether a human must review — and it produces the overall reasonableness
conclusion (assessment / score / summary) the pipeline was missing.

Code-side guardrails (the model cannot override these):

- a candidate whose legal assessment is anything other than
  ``no_substantive_risk`` is NEVER dropped — "not illegal" does not mean
  "reasonable";
- when the adjudicator is unreachable every candidate survives as
  review_required, so nothing is lost silently;
- candidates the model responds without a verdict for are kept and queued.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClientError, Transport, request_json
from ..models import Contract
from .state import (
    AgentReviewConfig,
    ASSESSMENT_LABELS,
    Citation,
    FinalRisk,
    PROTECTED_ASSESSMENTS,
    RiskCandidate,
)


SCHEMA_NAME = "final_adjudication"

_SYSTEM = (
    "你是合同审查的最终裁判。根据合同理解、风险候选和法条评估，逐条裁决"
    "并给出整份合同的合理性结论。裁决准则：\n"
    "1. 不违法不等于合理：法条评估为商业不合理、权利义务失衡、履约风险的"
    "候选必须保留；只有评估为 no_substantive_risk 且你确认无实质问题的"
    "才可以删除。\n"
    "2. 事实不足、需要补充材料的候选保留并标记 review_required=true。\n"
    "3. 严重程度由你根据对当事人利益的影响重新裁定。\n"
    "4. 同一问题的多个候选可能来自不同审查员，裁决时注意一致性。\n"
    "最后输出整体结论：overall_assessment（low_risk/medium_risk/high_risk）、"
    "overall_score（0-100 合理性评分，越高越合理）、summary（一句话概括"
    "合同整体是否合理、偏向哪一方、最突出的问题）。"
    "严格按 JSON Schema 输出。"
)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "keep": {"type": "boolean"},
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "review_required": {"type": "boolean"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "candidate_id",
                        "keep",
                        "severity",
                        "review_required",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
            "overall_assessment": {
                "type": "string",
                "enum": ["low_risk", "medium_risk", "high_risk"],
            },
            "overall_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
            "summary": {"type": "string"},
        },
        "required": [
            "verdicts",
            "overall_assessment",
            "overall_score",
            "summary",
        ],
        "additionalProperties": False,
    }


def _prompt(
    understanding: dict[str, Any],
    candidates: list[RiskCandidate],
    applications: list[dict[str, Any]],
    feedback: list[str],
) -> str:
    apps = {str(app.get("candidate_id")): app for app in applications}
    payload = []
    for candidate in candidates:
        verdict = apps.get(candidate.candidate_id, {}).get("verdict") or {}
        payload.append(
            {
                "candidate_id": candidate.candidate_id,
                "code": candidate.code,
                "clause_ids": candidate.clause_ids,
                "severity": candidate.severity,
                "issue": candidate.message[:200],
                "evidence": candidate.evidence[:2],
                "assessment": verdict.get("assessment", ""),
                "assessment_label": ASSESSMENT_LABELS.get(
                    str(verdict.get("assessment", "")), ""
                ),
                "legal_reason": str(verdict.get("reason", ""))[:200],
                "citation": (
                    verdict.get("citation") or {}
                ).get("article_no", ""),
            }
        )
    retry_note = ""
    if feedback:
        retry_note = (
            "上一轮裁决被认为不完整，请特别处理："
            + "；".join(feedback[:5])
            + "\n\n"
        )
    return (
        f"{retry_note}合同理解摘要：{json.dumps(understanding, ensure_ascii=False)[:800]}\n\n"
        f"候选清单：{json.dumps(payload, ensure_ascii=False)}"
    )


def _fallback(
    candidates: list[RiskCandidate], reason: str
) -> tuple[list[FinalRisk], list[dict[str, str]], dict[str, Any], list[dict[str, str]]]:
    """Adjudicator unavailable: keep everything for the human queue."""

    final: list[FinalRisk] = []
    queue: list[dict[str, str]] = []
    for candidate in candidates:
        risk = FinalRisk(
            risk_id=f"R{len(final) + 1:04d}",
            code=candidate.code,
            severity=candidate.severity,
            message=candidate.message,
            clause_ids=list(candidate.clause_ids),
            block_ids=list(candidate.block_ids),
            confidence=0.5,
            review_required=True,
            rationale=f"最终裁判不可用（{reason}），保守保留，转人工复核。",
            agent_sources=sorted(set(candidate.source.split("+"))),
            evidence=[e for e in candidate.evidence if e],
        )
        final.append(risk)
        queue.append(
            {
                "risk_id": risk.risk_id,
                "code": risk.code,
                "reason": risk.rationale,
            }
        )
    overall = {
        "overall_assessment": "unknown",
        "overall_score": None,
        "summary": "最终裁判不可用，未能给出整体结论，请人工复核全部风险。",
    }
    return final, queue, overall, []


def adjudicate(
    contract: Contract,
    understanding: dict[str, Any],
    candidates: list[RiskCandidate],
    applications: list[dict[str, Any]],
    config: AgentReviewConfig,
    transport: Transport | None = None,
    feedback: list[str] | None = None,
) -> tuple[
    list[FinalRisk],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, str]],
    bool,
]:
    """Run ⑤: (final_risks, review_queue, overall, evidence_feedback, ok)."""

    if not candidates:
        if not config.enable_llm:
            # No model, no candidates: this is "not reviewed", which must
            # never masquerade as "no problems found".
            return (
                [],
                [],
                {
                    "overall_assessment": "unknown",
                    "overall_score": None,
                    "summary": "未启用模型，未执行实质审查，结果不可作为合同结论。",
                },
                [],
                True,
            )
        return (
            [],
            [],
            {
                "overall_assessment": "low_risk",
                "overall_score": 90,
                "summary": "未发现风险候选，合同无明显异常。",
            },
            [],
            True,
        )
    if not config.enable_llm:
        result = _fallback(candidates, "未启用模型")
        return (*result, False)

    try:
        parsed = request_json(
            config.endpoint,
            config.model,
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": _prompt(
                        understanding, candidates, applications, feedback or []
                    ),
                },
            ],
            schema_name=SCHEMA_NAME,
            schema=_schema(),
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            disable_thinking=config.disable_thinking,
            transport=transport,
        )
    except LLMClientError as exc:
        result = _fallback(candidates, str(exc))
        return (*result, False)

    decisions = {
        str(item.get("candidate_id")): item
        for item in parsed.get("verdicts", [])
        if isinstance(item, dict)
    }
    apps = {str(app.get("candidate_id")): app for app in applications}
    citation_lookup: dict[str, Citation | None] = {}
    for candidate_id, app in apps.items():
        payload = ((app.get("verdict") or {}).get("citation")) or None
        citation_lookup[candidate_id] = (
            Citation(
                law_name=str(payload.get("law_name", "")),
                article_no=str(payload.get("article_no", "")),
                quote=str(payload.get("quote", "")),
                source_url=str(payload.get("source_url", "")),
            )
            if payload
            else None
        )

    final: list[FinalRisk] = []
    queue: list[dict[str, str]] = []
    evidence_feedback: list[dict[str, str]] = []
    for candidate in candidates:
        decision = decisions.get(candidate.candidate_id)
        verdict = apps.get(candidate.candidate_id, {}).get("verdict") or {}
        assessment = str(verdict.get("assessment", ""))

        # ---- guardrails: the model cannot drop reasonable risks ----
        if decision is None:
            keep = True
            review_required = True
            severity = candidate.severity
            rationale = "裁判未返回该项裁决，保守保留，转人工复核。"
        else:
            keep = bool(decision.get("keep", True))
            review_required = bool(decision.get("review_required", False))
            severity = str(decision.get("severity") or candidate.severity)
            rationale = str(decision.get("rationale") or "")
            if not keep and assessment != "no_substantive_risk":
                keep = True
                review_required = True
                rationale += "（护栏：评估为" + ASSESSMENT_LABELS.get(
                    assessment, assessment
                ) + "，不因未违法而删除。）"
            elif not keep:
                continue  # genuinely no substantive risk: allowed to drop

        if assessment in ("insufficient_facts", "need_more_material"):
            review_required = True
            for clause_id in candidate.clause_ids or [""]:
                evidence_feedback.append(
                    {
                        "clause_id": clause_id,
                        "reason": ASSESSMENT_LABELS.get(
                            assessment, assessment
                        )
                        + "："
                        + str(verdict.get("reason", ""))[:120],
                    }
                )

        citation = citation_lookup.get(candidate.candidate_id)
        message = str(verdict.get("reason") or candidate.message)[:200]
        risk = FinalRisk(
            risk_id=f"R{len(final) + 1:04d}",
            code=candidate.code,
            severity=severity,
            message=message,
            citations=[citation] if citation else [],
            clause_ids=list(candidate.clause_ids),
            block_ids=list(candidate.block_ids),
            confidence=0.9 if decision else 0.5,
            review_required=review_required,
            rationale=rationale,
            agent_sources=sorted(set(candidate.source.split("+"))),
            category=assessment or "unassessed",
            evidence=[e for e in candidate.evidence if e],
        )
        final.append(risk)

    for index, risk in enumerate(final, start=1):
        risk.risk_id = f"R{index:04d}"
        if risk.review_required:
            queue.append(
                {
                    "risk_id": risk.risk_id,
                    "code": risk.code,
                    "reason": risk.rationale or "需人工确认。",
                }
            )

    overall = {
        "overall_assessment": str(
            parsed.get("overall_assessment") or "unknown"
        ),
        "overall_score": parsed.get("overall_score"),
        "summary": str(parsed.get("summary") or ""),
    }
    return final, queue, overall, evidence_feedback, True
