"""M6 unified multi-agent review pipeline (LangGraph).

    合同解析与条款切分
              ↓
    ① 合同理解Agent（唯一一次，仅语义层）
              ↓
      ┌───────┼────────┐
 elements    ② 逐条审查  ③ 跨条款审查
（M3+数学信号）  └───┬───┘
      └───────┼───────┘
          风险候选合并
              ↓
    ④ 法律检索与适用（七分类评估）
              ↓
    ⑤ 最终裁判（LLM）+ 整体合理性结论
              ↓
    LOOP 跨模块冲突仲裁 →（实质冲突）回到⑤
              ↓
    F 统一融合（去重/交叉印证/排序）
              ↓
    UnifiedReviewReport schema=m6 + 人工复核
"""

from .graph import (
    build_review_graph,
    run_agent_review,
    run_unified_review,
    save_report,
)
from .state import (
    AgentReviewConfig,
    Citation,
    Conflict,
    ElementSignal,
    FinalRisk,
    ReviewState,
    RiskCandidate,
)

__all__ = [
    "AgentReviewConfig",
    "Citation",
    "Conflict",
    "ElementSignal",
    "FinalRisk",
    "ReviewState",
    "RiskCandidate",
    "build_review_graph",
    "run_agent_review",
    "run_unified_review",
    "save_report",
]
