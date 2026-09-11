"""① 合同理解Agent: read the contract once and build its semantic structure.

Pure model pass — there is no keyword fallback. When the model is disabled or
unreachable the node returns an explicit empty structure plus an error, so
downstream agents know the understanding is missing instead of silently
consuming a rough rule-made guess.

The schema stays at the *semantic* layer (type, clause roles, obligations,
payment summary). Parties and the subject are NOT duplicated here — the
elements node (M3) is the single source for structured parties/subject, so
the unified graph never extracts the same information twice.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClientError, Transport, request_json
from ..models import Contract
from .state import AgentReviewConfig


SCHEMA_NAME = "contract_understanding"

_SYSTEM = (
    "你是合同理解分析器。通读条款后输出合同的语义结构："
    "类型、标的概述、价款安排、主要义务，以及每个条款扮演的角色"
    "（如 当事人信息/付款/交付/违约责任/争议解决/其他）。"
    "当事人与结构化要素由专门的抽取环节负责，这里不要重复输出。"
    "严格按 JSON Schema 输出，不补充其他文字。"
)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "contract_type": {"type": "string"},
            "subject_summary": {
                "type": "string",
                "description": "标的的语义描述（结构化标的由要素抽取负责）",
            },
            "payment_summary": {"type": "string"},
            "key_obligations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "clause_roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "clause_id": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["clause_id", "role"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "contract_type",
            "subject_summary",
            "payment_summary",
            "key_obligations",
            "clause_roles",
        ],
        "additionalProperties": False,
    }


def _prompt(contract: Contract) -> str:
    clauses = [
        {
            "clause_id": clause.clause_id,
            "label": clause.label,
            "text": clause.text[:300],
        }
        for clause in contract.document.clauses
    ]
    return (
        "下面是合同按条款切分后的内容，请输出语义结构。\n"
        "clause_roles 必须覆盖每一个 clause_id。\n\n"
        f"clauses={json.dumps(clauses, ensure_ascii=False)}"
    )


def empty_understanding(reason: str) -> dict[str, Any]:
    """Explicit 'unavailable' marker; never dressed up as a real analysis."""

    return {
        "contract_type": "",
        "subject_summary": "",
        "payment_summary": "",
        "key_obligations": [],
        "clause_roles": [],
        "source": "unavailable",
        "unavailable_reason": reason,
    }


def understand_contract(
    contract: Contract,
    config: AgentReviewConfig,
    transport: Transport | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Run ① and return (semantic structure, errors)."""

    if not config.enable_llm:
        return (
            empty_understanding("未启用模型，合同理解不可用。"),
            ["合同理解：未启用模型。"],
        )
    try:
        parsed = request_json(
            config.endpoint,
            config.model,
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(contract)},
            ],
            schema_name=SCHEMA_NAME,
            schema=_schema(),
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            disable_thinking=config.disable_thinking,
            transport=transport,
        )
    except LLMClientError as exc:
        return (
            empty_understanding(f"模型调用失败：{exc}"),
            [f"合同理解：{exc}"],
        )

    valid_ids = {c.clause_id for c in contract.document.clauses}
    roles = [
        item
        for item in parsed.get("clause_roles", [])
        if isinstance(item, dict) and item.get("clause_id") in valid_ids
    ]
    covered = {item["clause_id"] for item in roles}
    for clause in contract.document.clauses:
        if clause.clause_id not in covered:
            roles.append({"clause_id": clause.clause_id, "role": "其他"})
    result = dict(parsed)
    result["clause_roles"] = roles
    result["source"] = "llm"
    return result, []
