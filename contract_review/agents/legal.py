"""④ 法律检索与适用Agent: ground every candidate in the statute base.

The verdict is a multi-class assessment, not a violation flag: "not illegal"
does not mean "reasonable", so the model may return commercial
unreasonableness, rights imbalance, performance risk, insufficient facts or
need-more-material — all of which survive to the final report. Citations are
restricted to the retrieved pool; a retry round widens the query for
candidates that got no verdict the first time.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClientError, Transport, request_json
from ..models import Contract
from .state import (
    AgentReviewConfig,
    ASSESSMENT_TYPES,
    RiskCandidate,
    RISK_LABELS,
)


SCHEMA_NAME = "legal_application_verdict"

_SYSTEM = (
    "你是法律适用与合同评估专家。下面每个疑点附有按相似度召回的候选法条，"
    "请综合条款内容与法条给出评估结论。assessment 只能取以下值：\n"
    "legal_risk=违反法律强制性规定；commercial_unreasonable=不违法但商业上"
    "明显不合理；rights_imbalance=权利义务严重失衡；performance_risk=存在"
    "履约风险；insufficient_facts=现有文本不足以判断；need_more_material="
    "需要补充材料才能定论；no_substantive_risk=条款无实质问题。\n"
    "重要：不违法不等于合理——商业不合理、权利义务失衡等即使不违反法条"
    "也必须如实报告。引用必须来自该疑点的 candidate_articles（article_uid "
    "原样返回，quote 摘录法条关键句）；如果拟判 legal_risk 但没有相关法条，"
    "改判 commercial_unreasonable 或 rights_imbalance。"
    "reason 是一句话（不超过80字）说明结论与依据，不要复述分析过程。"
    "严格按 JSON Schema 输出。"
)

_EXTRA_LABELS = {
    "unfair_term": "权利义务显失公平",
    "missing_element": "缺少必要条款要素",
    "ambiguous_wording": "条款表述含歧义",
}


def _label(code: str) -> str:
    return RISK_LABELS.get(code, _EXTRA_LABELS.get(code, code))


_MAX_CANDIDATES_PER_CALL = 4


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
                        "assessment": {
                            "type": "string",
                            "enum": ASSESSMENT_TYPES,
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reason": {"type": "string"},
                        "article_uid": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": [
                        "candidate_id",
                        "assessment",
                        "severity",
                        "reason",
                        "article_uid",
                        "quote",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _prompt(payload: list[dict[str, Any]]) -> str:
    return (
        "逐条评估下面的疑点。每条都要返回 assessment（七类之一）。\n\n"
        f"candidates={json.dumps(payload, ensure_ascii=False)}"
    )


def _query_for(
    candidate: RiskCandidate, contract: Contract, retry: bool
) -> str:
    label = _label(candidate.code)
    if retry:
        # Wider query for the retry round: risk label + the model's own words,
        # without pinning to one clause text.
        return f"{label} {candidate.message[:150]}"
    clause_text = ""
    for clause in contract.document.clauses:
        if clause.clause_id in candidate.clause_ids:
            clause_text = clause.text[:300]
            break
    evidence = candidate.evidence[0] if candidate.evidence else candidate.message
    return f"{label} {evidence[:200]} {clause_text}"


def apply_law(
    candidates: list[RiskCandidate],
    contract: Contract,
    config: AgentReviewConfig,
    kb: Any = None,
    transport: Transport | None = None,
    retry: bool = False,
    only_for: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run ④ and return (applications, errors).

    ``retry`` widens the retrieval query; ``only_for`` restricts the round to
    candidates that still lack a verdict.
    """

    errors: list[str] = []
    selected = [
        c
        for c in candidates
        if only_for is None or c.candidate_id in only_for
    ]
    if not selected:
        return [], errors

    applications: list[dict[str, Any]] = []
    article_index: dict[str, Any] = {}
    for candidate in selected:
        articles: list[Any] = []
        if kb is not None:
            try:
                articles = kb.search(
                    _query_for(candidate, contract, retry),
                    top_k=config.top_k,
                )
            except Exception as exc:  # noqa: BLE001 - degrade to rule-only
                errors.append(f"法条检索失败（{candidate.candidate_id}）：{exc}")
        for article in articles:
            article_index[article.article_uid] = article
        applications.append(
            {
                "candidate_id": candidate.candidate_id,
                "articles": [
                    {
                        "article_uid": a.article_uid,
                        "citation": a.citation(),
                        "content": a.content,
                    }
                    for a in articles
                ],
                "verdict": None,
            }
        )
    if kb is None:
        errors.append("法律索引不可用，跳过法条检索。")

    if not config.enable_llm:
        return applications, errors

    by_id = {a["candidate_id"]: a for a in applications}
    for start in range(0, len(selected), _MAX_CANDIDATES_PER_CALL):
        chunk = selected[start : start + _MAX_CANDIDATES_PER_CALL]
        payload = [
            {
                "candidate_id": candidate.candidate_id,
                "risk": _label(candidate.code),
                "issue": candidate.message[:200],
                "clause_text": (
                    candidate.evidence[0][:200] if candidate.evidence else ""
                ),
                "candidate_articles": by_id[candidate.candidate_id]["articles"],
            }
            for candidate in chunk
        ]
        try:
            parsed = request_json(
                config.endpoint,
                config.model,
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _prompt(payload)},
                ],
                schema_name=SCHEMA_NAME,
                schema=_schema(),
                api_key=config.api_key,
                timeout_seconds=config.timeout_seconds,
                disable_thinking=config.disable_thinking,
                transport=transport,
            )
        except LLMClientError as exc:
            errors.append(f"法律适用评估失败：{exc}")
            continue
        for item in parsed.get("verdicts", []):
            if not isinstance(item, dict):
                continue
            application = by_id.get(str(item.get("candidate_id", "")))
            if application is None:
                continue
            verdict = dict(item)
            article = article_index.get(str(item.get("article_uid", "")))
            verdict["citation"] = (
                {
                    "law_name": article.law_name,
                    "article_no": article.article_no,
                    "quote": str(item.get("quote", ""))[:200],
                    "source_url": "https://flk.npc.gov.cn",
                }
                if article is not None
                else None
            )
            application["verdict"] = verdict

    return applications, errors


def unresolved_candidate_ids(
    applications: list[dict[str, Any]],
) -> list[str]:
    """Candidates that still have no verdict (retry candidates)."""

    return [
        str(app.get("candidate_id"))
        for app in applications
        if not app.get("verdict")
    ]
