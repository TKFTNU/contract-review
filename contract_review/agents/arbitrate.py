"""LOOP 跨模块冲突仲裁节点（新）: consistency check across modules.

The five-agent pipeline already resolves conflicts *within* one review round
(retrieval retry, judge retry, evidence round-trip). M6 adds the missing
piece: conflicts *between modules* —

- the same clause flagged with different risk types / severities by the
  clause reviewer and the cross-clause reviewer;
- element signals (mathematical facts) contradicting a model conclusion;
- mutually exclusive conclusions (a risk kept while its basis is denied).

A material conflict sends the adjudicator back for one bounded re-run with
the arbitration feedback; otherwise the graph proceeds to fusion. When the
model is unavailable the node degrades to "no conflicts", which only means
no extra round — it never blocks or drops anything.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClientError, Transport, request_json
from ..models import Contract
from .state import (
    AgentReviewConfig,
    ASSESSMENT_LABELS,
    Conflict,
    ElementSignal,
    FinalRisk,
)


SCHEMA_NAME = "cross_module_arbitration"

_SYSTEM = (
    "你是审查质量仲裁员，负责检查多个审查模块的结论之间是否互相矛盾。"
    "检查以下情形：\n"
    "1. 同一问题被不同模块判成不同风险类型或严重度明显不一致；\n"
    "2. 要素事实（金额、日期等数学核对结果）与风险结论互相矛盾"
    "（例如要素显示金额为 0，但结论称金额正常）；\n"
    "3. 结论互斥（同一事项一处认定有风险、另一处认定无风险）；\n"
    "4. 同一条款的同类风险严重度差异超过一个级别。\n"
    "对每个冲突给出类型、描述、涉及的编号和仲裁意见（应以哪边为准、"
    "或需要重新裁决）。has_material_conflict 只在冲突会实质影响结论时"
    "为 true；轻微表述差异不算。没有冲突就返回空数组。"
    "严格按 JSON Schema 输出。"
)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "severity_mismatch",
                                "type_mismatch",
                                "element_contradiction",
                                "mutual_exclusion",
                            ],
                        },
                        "description": {"type": "string"},
                        "involved_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "resolution": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "description",
                        "involved_ids",
                        "resolution",
                    ],
                    "additionalProperties": False,
                },
            },
            "has_material_conflict": {"type": "boolean"},
        },
        "required": ["conflicts", "has_material_conflict"],
        "additionalProperties": False,
    }


def _prompt(
    understanding: dict[str, Any],
    signals: list[ElementSignal],
    risks: list[FinalRisk],
    feedback: list[str],
) -> str:
    risk_payload = [
        {
            "risk_id": risk.risk_id,
            "code": risk.code,
            "category": risk.category,
            "category_label": ASSESSMENT_LABELS.get(
                risk.category, risk.category
            ),
            "severity": risk.severity,
            "clause_ids": risk.clause_ids,
            "message": risk.message[:150],
            "sources": risk.agent_sources,
            "review_required": risk.review_required,
        }
        for risk in risks
    ]
    signal_payload = [signal.to_dict() for signal in signals]
    retry_note = ""
    if feedback:
        retry_note = (
            "上一轮仲裁已要求重新裁决，以下是当时的仲裁反馈，"
            "请确认问题是否已解决：\n- "
            + "\n- ".join(feedback[:5])
            + "\n\n"
        )
    return (
        f"{retry_note}合同类型：{understanding.get('contract_type', '未识别')}\n\n"
        f"最终风险清单：{json.dumps(risk_payload, ensure_ascii=False)}\n\n"
        f"要素事实信号：{json.dumps(signal_payload, ensure_ascii=False)}"
    )


def arbitrate(
    contract: Contract,
    understanding: dict[str, Any],
    element_signals: list[ElementSignal],
    final_risks: list[FinalRisk],
    config: AgentReviewConfig,
    transport: Transport | None = None,
    feedback: list[str] | None = None,
) -> tuple[list[Conflict], bool, list[str]]:
    """Run the arbitration node: (conflicts, has_material_conflict, errors)."""

    if not final_risks:
        return [], False, []
    if not config.enable_llm:
        return [], False, ["冲突仲裁：未启用模型，跳过仲裁。"]

    try:
        parsed = request_json(
            config.endpoint,
            config.model,
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": _prompt(
                        understanding, element_signals, final_risks, feedback or []
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
        return [], False, [f"冲突仲裁失败（不影响结果）：{exc}"]

    conflicts: list[Conflict] = []
    for item in parsed.get("conflicts", []):
        if not isinstance(item, dict):
            continue
        conflicts.append(
            Conflict(
                type=str(item.get("type") or "unknown"),
                description=str(item.get("description") or ""),
                involved_ids=[
                    str(value) for value in item.get("involved_ids", [])
                ],
                resolution=str(item.get("resolution") or ""),
            )
        )
    return conflicts, bool(parsed.get("has_material_conflict")), []
