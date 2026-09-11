"""② 逐条审查Agent: inspect each clause with the model, one clause at a time.

Risk codes are open-ended on purpose: real contracts carry risks nobody can
enumerate in a keyword table (hidden liability shifts, one-sided renewals,
missing remedies, unusual payment mechanics...). The model reports whatever
it can ground in the clause text; ``SUGGESTED_CLAUSE_CODES`` is only a hint
list, not a closed enum. The shared ``RISK_LABELS`` table adds Chinese labels
for well-known codes when building retrieval queries downstream.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClientError, Transport, request_json
from ..models import Contract
from .state import (
    AgentReviewConfig,
    RiskCandidate,
    SUGGESTED_CLAUSE_CODES,
)


SCHEMA_NAME = "clause_risk_review"

_SYSTEM = (
    "你是资深合同审查律师，逐条检查条款是否存在对当事人在法律上或商业上"
    "不利、不完整、含歧义或可能无效的内容。检查范围包括但不限于："
    "权利义务显失公平、缺少必要要素、涉嫌违反强制性规定、表述含歧义、"
    "责任与赔偿失衡、解除与续约机制异常、限制法定权利、风险分配不合理等。"
    "不要比较条款之间是否矛盾（跨条款一致性由另一位审查员负责）。"
    "只报告有明确文本依据的问题，每条给出条款号、简短风险类型（英文蛇形"
    "命名，可自拟新类型）、严重程度、一句话理由和原文证据。"
    "没有发现问题就返回空数组。严格按 JSON Schema 输出。"
)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # Free-form code: no enum, new risk families allowed.
                        "code": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "clause_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": [
                        "code",
                        "severity",
                        "clause_id",
                        "reason",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["risks"],
        "additionalProperties": False,
    }


def _build_batches(
    contract: Contract, config: AgentReviewConfig
) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    characters = 0
    for clause in contract.document.clauses:
        payload = {
            "clause_id": clause.clause_id,
            "label": clause.label,
            "text": clause.text[:1_500],
        }
        size = len(payload["text"])
        if current and characters + size > config.max_characters_per_batch:
            batches.append(current)
            current = []
            characters = 0
        current.append(payload)
        characters += size
    if current:
        batches.append(current)
    return batches


def _prompt(batch: list[dict[str, str]], number: int, total: int) -> str:
    hints = "、".join(SUGGESTED_CLAUSE_CODES)
    return (
        f"这是第{number}/{total}批条款。逐条审查并输出风险。"
        f"常见的风险类型包括：{hints}——这只是参考，"
        "如有其他风险请自拟新的英文蛇形 code。"
        "clause_id 必须来自本批输入，evidence 摘录条款原文的关键片段。"
        "没有风险就返回空数组。\n\n"
        f"clauses={json.dumps(batch, ensure_ascii=False)}"
    )


def review_clauses(
    contract: Contract,
    config: AgentReviewConfig,
    transport: Transport | None = None,
) -> tuple[list[RiskCandidate], list[str]]:
    """Run ② and return (candidates, errors)."""

    errors: list[str] = []
    candidates: list[RiskCandidate] = []
    if not config.enable_llm:
        return candidates, errors

    valid_ids = {c.clause_id for c in contract.document.clauses}
    batches = _build_batches(contract, config)
    for number, batch in enumerate(batches, start=1):
        try:
            parsed = request_json(
                config.endpoint,
                config.model,
                [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": _prompt(batch, number, len(batches)),
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
            errors.append(f"逐条审查批次{number}/{len(batches)}：{exc}")
            continue
        for item in parsed.get("risks", []):
            if not isinstance(item, dict):
                continue
            clause_id = str(item.get("clause_id", ""))
            if clause_id not in valid_ids:
                continue
            code = str(item.get("code") or "").strip() or "risk"
            candidates.append(
                RiskCandidate(
                    candidate_id=f"KC{len(candidates) + 1:04d}",
                    code=code,
                    severity=str(item.get("severity") or "medium"),
                    message=str(item.get("reason") or ""),
                    clause_ids=[clause_id],
                    evidence=[str(item.get("evidence") or "")],
                    confidence=0.8,
                    source="clause_llm",
                )
            )
    return candidates, errors
