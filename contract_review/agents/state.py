"""Shared state and data contracts for the LangGraph review pipeline.

M6 unified graph (see ``graph.py``)::

    合同解析与条款切分（M1/M2 既有产物 Contract）
                  ↓
    ① 合同理解Agent（唯一一次，仅语义层）
                  ↓
      ┌───────────┼───────────┐
 elements      ② 逐条审查   ③ 跨条款审查（N2）
（M3+数学信号）    └─────┬─────┘
      └───────────┼───────┘
              风险候选合并
                  ↓
          ④ 法律检索与适用（七分类）
                  ↓
          ⑤ 最终裁判（LLM）+ 整体结论
                  ↓
          LOOP 跨模块冲突仲裁 →（实质冲突）回到⑤
                  ↓
          F 统一融合（去重/交叉印证/排序）
                  ↓
          UnifiedReviewReport schema=m6 + 人工复核
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from .. import settings
from ..models import Contract


# --------------------------------------------------------------------------
# risk candidates: the intermediate currency between agents
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RiskCandidate:
    """One potential risk produced by the clause / cross-clause agents."""

    candidate_id: str
    code: str
    severity: str
    message: str
    clause_ids: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.8
    source: str = "clause_llm"

    def dedupe_key(self) -> tuple[str, str]:
        return (self.code, ",".join(sorted(self.clause_ids)))


@dataclass(slots=True)
class Citation:
    law_name: str
    article_no: str
    quote: str = ""
    source_url: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "law_name": self.law_name,
            "article_no": self.article_no,
            "quote": self.quote,
            "source_url": self.source_url,
        }


@dataclass(slots=True)
class FinalRisk:
    """A risk after ⑤ final adjudication; ready for the report."""

    risk_id: str
    code: str
    severity: str
    message: str
    citations: list[Citation] = field(default_factory=list)
    clause_ids: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    confidence: float = 0.85
    review_required: bool = False
    rationale: str = ""
    agent_sources: list[str] = field(default_factory=list)
    category: str = ""
    cross_validated: bool = False
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "category_label": ASSESSMENT_LABELS.get(
                self.category, self.category
            ),
            "message": self.message,
            "evidence": self.evidence,
            "citations": [c.to_dict() for c in self.citations],
            "clause_ids": self.clause_ids,
            "block_ids": self.block_ids,
            "confidence": self.confidence,
            "review_required": self.review_required,
            "rationale": self.rationale,
            "agent_sources": self.agent_sources,
            "cross_validated": self.cross_validated,
        }


@dataclass(slots=True)
class ElementSignal:
    """A pure mathematical fact about extracted elements (not a risk).

    Signals never become risk candidates on their own — risk detection stays
    with the models. They only cross-validate model findings (same clause)
    and flag element-level doubts for the human queue.
    """

    kind: str  # amount / date / party / subject / completeness
    note: str
    clause_id: str = ""
    value: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "note": self.note,
            "clause_id": self.clause_id,
            "value": self.value,
        }


@dataclass(slots=True)
class Conflict:
    """One cross-module inconsistency found by the arbitration agent."""

    type: str
    description: str
    involved_ids: list[str] = field(default_factory=list)
    resolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "involved_ids": self.involved_ids,
            "resolution": self.resolution,
        }


# --------------------------------------------------------------------------
# graph state
# --------------------------------------------------------------------------


class ReviewState(TypedDict, total=False):
    """LangGraph state; nodes return partial updates for their own keys."""

    contract: Contract
    understanding: dict[str, Any]
    elements: dict[str, Any]
    element_signals: list[ElementSignal]
    m3_stats: dict[str, Any]
    clause_candidates: list[RiskCandidate]
    cross_candidates: list[RiskCandidate]
    merged_candidates: list[RiskCandidate]
    legal_applications: list[dict[str, Any]]
    final_risks: list[FinalRisk]
    review_queue: list[dict[str, str]]
    conflicts: list[Conflict]
    overall: dict[str, Any]
    run_errors: Annotated[list[str], operator.add]
    stats: dict[str, Any]

    # ---- loops (conditional edges re-enter nodes with feedback) ----
    legal_retry: int
    judge_retry: int
    judge_retry_feedback: list[str]
    judge_ok: bool
    evidence_feedback: list[dict[str, str]]
    evidence_round: int
    arbitration_round: int
    arbitration_material: bool
    arbitration_rerun_requested: bool


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentReviewConfig:
    """Runtime configuration shared by every agent in the graph."""

    endpoint: str = field(default_factory=settings.llm_endpoint)
    model: str = field(default_factory=settings.llm_model)
    api_key: str = field(default_factory=settings.llm_api_key)
    timeout_seconds: int = field(default_factory=lambda: settings.llm_timeout(300))
    disable_thinking: bool = True
    enable_llm: bool = True

    kb_index_dir: Path = Path("data/legal_index")
    ollama_endpoint: str = field(default_factory=settings.ollama_endpoint)
    embed_model: str = field(default_factory=settings.embed_model)
    embed_api_key: str = field(default_factory=settings.embed_api_key)
    top_k: int = 6

    max_characters_per_batch: int = 8_000
    # A mechanically detected risk (date math, amount comparison) is kept even
    # when the model disagrees; anything below this confidence is queued for a
    # human, never silently dropped.
    review_confidence_threshold: float = 0.6

    # Loop bounds for the conditional edges (retrieval retry, judge retry,
    # evidence round-trip back to the clause reviewer, arbitration re-run).
    max_legal_retries: int = 1
    max_judge_retries: int = 1
    max_evidence_rounds: int = 1
    max_arbitration_rounds: int = 1


# Risk codes are an OPEN set: real contracts carry risks nobody enumerated in
# advance, so agents emit free-form snake_case codes. This table only supplies
# human labels for well-known codes (used to build retrieval queries and
# reports); unknown codes fall back to the code itself.
RISK_LABELS: dict[str, str] = {
    "term_over_20y": "租赁期限超过二十年",
    "zero_rent": "租金为零",
    "negative_fee": "费用为负数",
    "deposit_over_20pct": "定金超过主合同标的额百分之二十",
    "extreme_rent": "租金金额极端异常",
    "unlimited_authority": "授予无限授权",
    "unilateral_entry": "单方随意进入/处置",
    "unverified_property": "标的无权属证明",
    "no_report": "受托人无报告义务",
    "unfair_term": "权利义务显失公平",
    "missing_element": "缺少必要条款要素",
    "illegal_clause": "涉嫌违反强制性规定",
    "ambiguous_wording": "条款表述含歧义",
    "semantic_conflict": "跨条款语义矛盾",
    "amount_conflict": "金额一致性问题",
    "date_conflict": "日期一致性问题",
    "party_conflict": "主体一致性问题",
    "reference_conflict": "引用完整性问题",
}

# Suggested categories handed to the model as hints; it may name new ones.
SUGGESTED_CLAUSE_CODES = [
    "unfair_term",
    "missing_element",
    "illegal_clause",
    "ambiguous_wording",
    "term_over_20y",
    "zero_rent",
    "negative_fee",
    "deposit_over_20pct",
    "unlimited_authority",
    "unilateral_entry",
    "unverified_property",
    "no_report",
]

# ④ legal assessment is a multi-class judgement, not a violation flag:
# "not illegal" does NOT mean "reasonable", so only no_substantive_risk may
# ever cause a risk to be dropped downstream.
ASSESSMENT_TYPES = [
    "legal_risk",              # 违反法律强制性规定
    "commercial_unreasonable",  # 商业上明显不合理
    "rights_imbalance",        # 权利义务失衡
    "performance_risk",        # 履约风险
    "insufficient_facts",      # 事实不足，无法判断
    "need_more_material",      # 需要补充材料
    "no_substantive_risk",     # 未发现实质风险
]

ASSESSMENT_LABELS = {
    "legal_risk": "法律风险",
    "commercial_unreasonable": "商业不合理",
    "rights_imbalance": "权利义务失衡",
    "performance_risk": "履约风险",
    "insufficient_facts": "事实不足",
    "need_more_material": "需要补充材料",
    "no_substantive_risk": "未发现实质风险",
}

# Kept regardless of any model verdict — the pipeline's hard guardrail.
PROTECTED_ASSESSMENTS = tuple(
    item for item in ASSESSMENT_TYPES if item != "no_substantive_risk"
)
