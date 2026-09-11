from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import settings
from .extractor import M3Config, extract_elements
from .llm_splitter import LLMSplitConfig, split_contract_with_llm
from .service import build_contract, parse_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a contract and split it into clauses.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="用本地OpenAI兼容模型做全量语义分割，而不是只用规则切分",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help="输出M2合同统一结构（含要素槽位），而不是解析结果",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="单独抽取当事人/金额/日期/标的等要素（M3）；--review 已含此环节",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="关闭模型：要素抽取只用规则；审查走保守保留（不丢风险）",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="运行综合审查（统一图：要素抽取+逐条/跨条款审查+法条检索+LLM裁判+仲裁+融合），"
        "输出单一 review 字段（schema m6）",
    )
    parser.add_argument(
        "--legal",
        action="store_true",
        help="与 --review 同义（兼容旧参数）",
    )
    parser.add_argument("--endpoint", default=settings.llm_endpoint())
    parser.add_argument("--model", default=settings.llm_model())
    parser.add_argument("--api-key", default=settings.llm_api_key())
    parser.add_argument("--timeout", type=int, default=settings.llm_timeout(300))
    parser.add_argument(
        "--reflow-rounds",
        type=int,
        default=1,
        help="C→L结构回流轮数上限，0表示关闭回流",
    )
    args = parser.parse_args()

    # Windows consoles default to GBK and choke on symbols like ☑ inside
    # contract text; force UTF-8 so full-contract JSON always prints.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    result = parse_contract(args.input.name, args.input.read_bytes())
    if args.llm:
        result, summary = split_contract_with_llm(
            result,
            LLMSplitConfig(
                endpoint=args.endpoint,
                model=args.model,
                api_key=args.api_key,
                timeout_seconds=args.timeout,
                max_reflow_rounds=max(0, args.reflow_rounds),
            ),
        )
        print(
            f"[LLM分割] 规则识别{summary.rule_clauses}条 → 模型识别"
            f"{summary.llm_clauses}条；批次 {summary.succeeded_batches}/{summary.batches} 成功",
            file=sys.stderr,
        )
        if summary.issues_before or summary.reflow_rounds:
            print(
                f"[结构回流] 可回流问题 {summary.issues_before} → {summary.issues_after}；"
                f"回流 {summary.reflow_rounds} 轮，重标注 {summary.reflowed_blocks} 个区块",
                file=sys.stderr,
            )
        for note in summary.reflow_notes:
            print(f"[结构回流] {note}", file=sys.stderr)
        for error in summary.errors:
            print(f"[LLM分割警告] {error}", file=sys.stderr)

    if args.legal:
        args.review = True
    if args.review or args.extract:
        args.contract = True

    review_payload = None
    element_count = 0
    if args.contract:
        contract = build_contract(result)

        if args.review:
            from .agents import AgentReviewConfig, run_unified_review

            review_payload = run_unified_review(
                contract,
                AgentReviewConfig(
                    endpoint=args.endpoint,
                    model=args.model,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout,
                    enable_llm=not args.rules_only,
                ),
            )
            stats = review_payload["stats"]
            overall = review_payload["overall"]
            loops = stats["loops"]
            print(
                f"[综合审查] 理解({review_payload['understanding'].get('source')})"
                f" → 要素({review_payload['m3_stats'].get('merged_elements', 0)})"
                f" → 逐条({stats['clause_candidates']})"
                f" → 跨条款({stats['cross_candidates']})"
                f" → 合并({stats['merged_candidates']})"
                f" → 裁判 {stats['final_risks']} 条风险"
                f"，{stats['review_required']} 条待人工复核"
                f"，交叉印证 {stats['cross_validated']}"
                f"，耗时 {stats['elapsed_seconds']}s",
                file=sys.stderr,
            )
            print(
                f"[回环] 检索重试 {loops['legal_retries']}"
                f" · 裁判重试 {loops['judge_retries']}"
                f" · 证据补充 {loops['evidence_rounds']}"
                f" · 仲裁重裁 {loops['arbitration_rounds']}",
                file=sys.stderr,
            )
            print(
                f"[整体结论] {overall.get('overall_assessment', '—')}"
                f"（合理性 {overall.get('overall_score', '—')}/100）："
                f"{overall.get('summary', '')}",
                file=sys.stderr,
            )
            if review_payload["conflicts"]:
                print(
                    f"[仲裁记录] {len(review_payload['conflicts'])} 条冲突",
                    file=sys.stderr,
                )
                for conflict in review_payload["conflicts"]:
                    print(
                        f"  [{conflict['type']}] {conflict['description']}"
                        f"（意见：{conflict['resolution']}）",
                        file=sys.stderr,
                    )
            for risk in review_payload["risks"]:
                cite = (
                    "；".join(
                        f"《{c['law_name']}》{c['article_no']}"
                        for c in risk["citations"]
                    )
                    or "（无引用）"
                )
                flag = " [待复核]" if risk["review_required"] else ""
                cross = " [已交叉印证]" if risk.get("cross_validated") else ""
                print(
                    f"  [{risk['severity']}] {risk['code']}"
                    f"（{risk.get('category_label', '')}）: {risk['message']}"
                    f" 依据 {cite}{cross}{flag}",
                    file=sys.stderr,
                )
            for error in review_payload["errors"]:
                print(f"[审查警告] {error}", file=sys.stderr)
            element_count = sum(
                len(values) for values in review_payload["elements"].values()
            )
        elif args.extract:
            contract, m3_summary = extract_elements(
                contract,
                M3Config(
                    endpoint=args.endpoint,
                    model=args.model,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout,
                    enable_llm=not args.rules_only,
                ),
            )
            print(
                f"[要素抽取] 规则{ m3_summary.rule_elements }个 + "
                f"模型{m3_summary.llm_elements}个 → 合并{m3_summary.merged_elements}个；"
                f"批次 {m3_summary.succeeded_batches}/{m3_summary.batches} 成功",
                file=sys.stderr,
            )
            print(
                "  分布：" + "、".join(
                    f"{kind} {count}" for kind, count in m3_summary.by_kind.items()
                ),
                file=sys.stderr,
            )
            for error in m3_summary.errors:
                print(f"[要素抽取警告] {error}", file=sys.stderr)
            element_count = m3_summary.merged_elements

        payload_obj = contract.to_dict()
        if review_payload is not None:
            payload_obj["review"] = review_payload
        payload = json.dumps(payload_obj, ensure_ascii=False, indent=2)
        print(
            f"[统一结构] schema {contract.schema_version} · 合同ID {contract.contract_id}"
            f" · 条款 {len(contract.document.clauses)} 条"
            f" · 要素 {element_count} 个",
            file=sys.stderr,
        )
    else:
        payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"已写入 {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
