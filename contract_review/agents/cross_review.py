"""③ 跨条款审查Agent: compare clauses with each other — pure model pass.

No deterministic checkers: whether two clauses truly contradict each other
depends on wording and intent (a term clause vs its summary line, a payment
schedule vs an attachment), which the model reads in context. The agent
returns conflict candidates spanning two or more clauses; the retrieval and
adjudication agents decide what survives.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClientError, Transport, request_json
from ..models import Contract
from .state import AgentReviewConfig, RiskCandidate


SCHEMA_NAME = "cross_clause_conflict_review"

_SYSTEM = (
    "你是合同一致性审查专家，只负责比较条款之间是否互相矛盾。关注："
    "金额口径冲突（分项合计与总额、大小写金额、前后不一致的数值）、"
    "日期冲突（期限倒置、交付日与付款日矛盾、摘要与条款不一致）、"
    "主体冲突（同一方在不同条款署名不同）、引用冲突（引用的附件/条款"
    "不存在或指向错误）、以及语义矛盾（条件互斥、权利与义务互相抵消）。"
    "每条冲突必须引用涉及的多个 clause_id 和两处原文证据。"
    "没有发现冲突就返回空数组。严格按 JSON Schema 输出。"
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
                        "code": {"type": "string"},  # open set
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "clause_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "reason": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": [
                        "code",
                        "severity",
                        "clause_ids",
                        "reason",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["conflicts"],
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


def _prompt(
    batch: list[dict[str, str]],
    number: int,
    total: int,
    feedback: list[dict[str, str]] | None,
) -> str:
    focus = ""
    if feedback:
        hints = "; ".join(
            f"{item.get('clause_id', '')}: {item.get('reason', '')}"
            for item in feedback[:8]
        )
        focus = (
            "上一轮审查被指出以下条款的证据不充分，请重点复核这些条款"
            f"与其他条款的关系：{hints}\n\n"
        )
    return (
        f"这是第{number}/{total}批条款，请找出条款之间的冲突。"
        "clause_ids 必须来自本批输入且至少两个。\n\n"
        f"{focus}clauses={json.dumps(batch, ensure_ascii=False)}"
    )


def review_cross_clauses(
    contract: Contract,
    config: AgentReviewConfig,
    transport: Transport | None = None,
    feedback: list[dict[str, str]] | None = None,
) -> tuple[list[RiskCandidate], list[str]]:
    """Run ③ and return (candidates, errors).

    ``feedback`` carries clause-level evidence requests from a previous
    adjudication round; when present the model re-examines those clauses.
    """

    errors: list[str] = []
    candidates: list[RiskCandidate] = []
    if not config.enable_llm:
        return candidates, ["跨条款审查：未启用模型。"]

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
                        "content": _prompt(batch, number, len(batches), feedback),
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
            errors.append(f"跨条款审查批次{number}/{len(batches)}：{exc}")
            continue
        for item in parsed.get("conflicts", []):
            if not isinstance(item, dict):
                continue
            clause_ids = [
                str(cid)
                for cid in item.get("clause_ids", [])
                if str(cid) in valid_ids
            ]
            if len(clause_ids) < 2:
                continue
            candidates.append(
                RiskCandidate(
                    candidate_id=f"XC{len(candidates) + 1:04d}",
                    code=str(item.get("code") or "").strip() or "cross_conflict",
                    severity=str(item.get("severity") or "medium"),
                    message=str(item.get("reason") or ""),
                    clause_ids=clause_ids,
                    evidence=[str(item.get("evidence") or "")],
                    confidence=0.8,
                    source="cross_llm",
                )
            )
    return candidates, errors
