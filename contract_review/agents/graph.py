"""M6 unified LangGraph orchestration (GOA + LOOP + F).

::

    START → ① understand（唯一一次合同理解，仅语义层）
              ├→ elements（M3 要素 + 数学信号）
              ├→ ② clause_review（逐条，纯模型）
              └→ ③ cross_review（跨条款 = N2，纯模型）
                   ↓ merge（三路汇合）
                ④ legal_apply（七分类评估）
                   ↓ judge（⑤ LLM 裁判 + 整体结论）
                   ↓ arbitrate（跨模块冲突仲裁）
                   ↓ fuse（统一融合）
                   END → UnifiedReviewReport(schema_version="m6")

Conditional edges, all bounded by AgentReviewConfig counters:

- ④ → ④   retrieval retry for candidates without a verdict (wider query);
- ⑤ → ⑤   adjudicator retry when the model call failed;
- ⑤ → ②   evidence round-trip (insufficient_facts / need_more_material);
- arbitrate → ⑤  material cross-module conflict: re-adjudicate with the
  arbitration feedback, then re-arbitrate once.

The retriever is opened by ``run_unified_review`` and closed in ``finally``:
Qdrant's local mode holds an exclusive lock per directory, so a client
created inside a node would block the next run of the same process.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import Contract
from .arbitrate import arbitrate
from .clause_review import review_clauses
from .cross_review import review_cross_clauses
from .elements import extract_contract_elements
from .fuse import fuse
from .judge import adjudicate
from .legal import apply_law, unresolved_candidate_ids
from .merge import merge_candidates
from .state import AgentReviewConfig, ReviewState
from .understanding import understand_contract


def _merge_applications(
    existing: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(app.get("candidate_id")): app for app in existing}
    for app in updates:
        by_id[str(app.get("candidate_id"))] = app
    return list(by_id.values())


def build_review_graph(
    config: AgentReviewConfig,
    kb: Any = None,
    transport: Any = None,
):
    """Compile the unified M6 graph; kb/transport are bound by closure."""

    def understand_node(state: ReviewState) -> dict[str, Any]:
        understanding, errors = understand_contract(
            state["contract"], config, transport
        )
        return {"understanding": understanding, "run_errors": errors}

    def elements_node(state: ReviewState) -> dict[str, Any]:
        filled, elements, signals, m3_stats, errors = (
            extract_contract_elements(state["contract"], config, transport)
        )
        return {
            # Overwrite the contract so every downstream node sees the
            # filled element slots.
            "contract": filled,
            "elements": elements,
            "element_signals": signals,
            "m3_stats": m3_stats,
            "run_errors": errors,
        }

    def clause_node(state: ReviewState) -> dict[str, Any]:
        feedback = state.get("evidence_feedback") or None
        candidates, errors = review_clauses(
            state["contract"], config, transport
        )
        update: dict[str, Any] = {
            "clause_candidates": candidates,
            "run_errors": errors,
            "evidence_feedback": [],
        }
        if feedback:
            # Evidence round-trip: the clause list may change, so any prior
            # verdicts are stale; legal re-runs from scratch.
            update["legal_applications"] = []
            update["evidence_round"] = state.get("evidence_round", 0) + 1
        return update

    def cross_node(state: ReviewState) -> dict[str, Any]:
        candidates, errors = review_cross_clauses(
            state["contract"],
            config,
            transport,
            feedback=state.get("evidence_feedback") or None,
        )
        return {"cross_candidates": candidates, "run_errors": errors}

    def merge_node(state: ReviewState) -> dict[str, Any]:
        merged = merge_candidates(
            state.get("clause_candidates", []),
            state.get("cross_candidates", []),
        )
        return {"merged_candidates": merged}

    def legal_node(state: ReviewState) -> dict[str, Any]:
        candidates = state.get("merged_candidates", [])
        existing = state.get("legal_applications", [])
        if existing:
            # Retry round: only the candidates still lacking a verdict are
            # re-queried, with a wider query.
            unresolved = unresolved_candidate_ids(existing)
            if not unresolved:
                return {}
            updates, errors = apply_law(
                candidates,
                state["contract"],
                config,
                kb=kb,
                transport=transport,
                retry=True,
                only_for=unresolved,
            )
            return {
                "legal_applications": _merge_applications(existing, updates),
                "legal_retry": state.get("legal_retry", 0) + 1,
                "run_errors": errors,
            }
        applications, errors = apply_law(
            candidates, state["contract"], config, kb=kb, transport=transport
        )
        return {
            "legal_applications": applications,
            "legal_retry": 0,
            "run_errors": errors,
        }

    def judge_node(state: ReviewState) -> dict[str, Any]:
        (
            final_risks,
            review_queue,
            overall,
            evidence_feedback,
            ok,
        ) = adjudicate(
            state["contract"],
            state.get("understanding", {}),
            state.get("merged_candidates", []),
            state.get("legal_applications", []),
            config,
            transport=transport,
            feedback=state.get("judge_retry_feedback") or None,
        )
        update: dict[str, Any] = {
            "final_risks": final_risks,
            "review_queue": review_queue,
            "overall": overall,
            "evidence_feedback": evidence_feedback,
            "judge_ok": ok,
        }
        if not ok:
            update["judge_retry"] = state.get("judge_retry", 0) + 1
            update["judge_retry_feedback"] = [
                "上一轮裁判未能完成，请重新裁决全部候选并给出整体结论。"
            ]
            update["run_errors"] = ["最终裁判调用失败，已保守保留全部候选。"]
        return update

    def arbitrate_node(state: ReviewState) -> dict[str, Any]:
        round_no = state.get("arbitration_round", 0)
        feedback = state.get("judge_retry_feedback", []) if round_no else None
        conflicts, material, errors = arbitrate(
            state["contract"],
            state.get("understanding", {}),
            state.get("element_signals", []),
            state.get("final_risks", []),
            config,
            transport=transport,
            feedback=feedback,
        )
        update: dict[str, Any] = {
            "conflicts": conflicts,
            "arbitration_material": material,
            "run_errors": errors,
        }
        if material and round_no < config.max_arbitration_rounds:
            # Bounded round-trip: re-adjudicate with the arbitration notes.
            # The flag records the request itself, because the round counter
            # is already incremented here and would fail the route check.
            update["arbitration_round"] = round_no + 1
            update["arbitration_rerun_requested"] = True
            update["judge_retry_feedback"] = [
                f"仲裁发现实质冲突，请重新裁决：{conflict.description}"
                f"（仲裁意见：{conflict.resolution}）"
                for conflict in conflicts
            ][:5]
        else:
            update["arbitration_rerun_requested"] = False
        return update

    def fuse_node(state: ReviewState) -> dict[str, Any]:
        risks, queue, errors = fuse(
            state.get("understanding", {}),
            state.get("elements", {}),
            state.get("element_signals", []),
            state.get("final_risks", []),
            state.get("conflicts", []),
            config,
        )
        return {
            "final_risks": risks,
            "review_queue": queue,
            "run_errors": errors,
        }

    def route_after_legal(state: ReviewState) -> str:
        unresolved = unresolved_candidate_ids(
            state.get("legal_applications", [])
        )
        if (
            unresolved
            and state.get("legal_retry", 0) < config.max_legal_retries
        ):
            return "legal_apply"
        return "judge"

    def route_after_judge(state: ReviewState) -> str:
        if (
            not state.get("judge_ok", True)
            and state.get("judge_retry", 0) <= config.max_judge_retries
        ):
            return "judge"
        if (
            state.get("evidence_feedback")
            and state.get("evidence_round", 0) < config.max_evidence_rounds
        ):
            return "clause_review"
        return "arbitrate"

    def route_after_arbitrate(state: ReviewState) -> str:
        if state.get("arbitration_rerun_requested"):
            return "judge"
        return "fuse"

    graph = StateGraph(ReviewState)
    graph.add_node("understand", understand_node)
    graph.add_node("elements", elements_node)
    graph.add_node("clause_review", clause_node)
    graph.add_node("cross_review", cross_node)
    graph.add_node("merge", merge_node)
    graph.add_node("legal_apply", legal_node)
    graph.add_node("judge", judge_node)
    graph.add_node("arbitrate", arbitrate_node)
    graph.add_node("fuse", fuse_node)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "elements")
    graph.add_edge("understand", "clause_review")
    graph.add_edge("understand", "cross_review")
    graph.add_edge("elements", "merge")
    graph.add_edge("clause_review", "merge")
    graph.add_edge("cross_review", "merge")
    graph.add_edge("merge", "legal_apply")
    graph.add_conditional_edges(
        "legal_apply",
        route_after_legal,
        {"legal_apply": "legal_apply", "judge": "judge"},
    )
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "judge": "judge",
            "clause_review": "clause_review",
            "arbitrate": "arbitrate",
        },
    )
    graph.add_conditional_edges(
        "arbitrate",
        route_after_arbitrate,
        {"judge": "judge", "fuse": "fuse"},
    )
    graph.add_edge("fuse", END)
    return graph.compile()


def run_unified_review(
    contract: Contract,
    config: AgentReviewConfig | None = None,
    kb: Any = None,
    transport: Any = None,
) -> dict[str, Any]:
    """Execute the unified graph and assemble the M6 review report."""

    active = config or AgentReviewConfig()
    owned_kb = False
    if kb is None and active.kb_index_dir is not None:
        try:
            from ..legal_kb import EmbedderConfig, LegalKB, OllamaEmbedder

            kb = LegalKB(
                index_dir=active.kb_index_dir,
                embedder=OllamaEmbedder(
                    EmbedderConfig(
                        endpoint=active.ollama_endpoint,
                        model=active.embed_model,
                        api_key=active.embed_api_key,
                    )
                ),
            )
            owned_kb = True
        except Exception:  # noqa: BLE001 - retrieval degrades gracefully
            kb = None

    graph = build_review_graph(active, kb=kb, transport=transport)
    started = time.perf_counter()
    try:
        state: ReviewState = graph.invoke(
            {
                "contract": contract,
                "run_errors": [],
            }
        )
    finally:
        if owned_kb and kb is not None:
            kb.close()
    elapsed = round(time.perf_counter() - started, 2)

    final_risks = state.get("final_risks", [])
    by_code: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for risk in final_risks:
        by_code[risk.code] = by_code.get(risk.code, 0) + 1
        by_category[risk.category] = by_category.get(risk.category, 0) + 1
        by_severity[risk.severity] = by_severity.get(risk.severity, 0) + 1

    return {
        "schema_version": "m6",
        "contract_id": contract.contract_id,
        "filename": contract.document.filename,
        "understanding": state.get("understanding", {}),
        "elements": state.get("elements", {}),
        "element_signals": [
            signal.to_dict() for signal in state.get("element_signals", [])
        ],
        "m3_stats": state.get("m3_stats", {}),
        "risks": [risk.to_dict() for risk in final_risks],
        "conflicts": [
            conflict.to_dict() for conflict in state.get("conflicts", [])
        ],
        "overall": state.get("overall", {}),
        "review_queue": state.get("review_queue", []),
        "stats": {
            "clause_candidates": len(state.get("clause_candidates", [])),
            "cross_candidates": len(state.get("cross_candidates", [])),
            "merged_candidates": len(state.get("merged_candidates", [])),
            "final_risks": len(final_risks),
            "review_required": len(state.get("review_queue", [])),
            "cross_validated": sum(
                risk.cross_validated for risk in final_risks
            ),
            "conflicts": len(state.get("conflicts", [])),
            "by_code": dict(sorted(by_code.items())),
            "by_category": dict(sorted(by_category.items())),
            "by_severity": dict(sorted(by_severity.items())),
            "loops": {
                "legal_retries": state.get("legal_retry", 0),
                "judge_retries": state.get("judge_retry", 0),
                "evidence_rounds": state.get("evidence_round", 0),
                "arbitration_rounds": state.get("arbitration_round", 0),
            },
            "elapsed_seconds": elapsed,
        },
        "errors": state.get("run_errors", []),
    }


# Backwards-compatible alias: earlier callers used the M5 entry name.
run_agent_review = run_unified_review


def save_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
