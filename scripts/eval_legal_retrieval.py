"""Offline retrieval evaluation for the legal knowledge base (M5 step 8).

Ground truth is two-tiered:

- *strong* tags map an issue_code to the articles written for exactly that
  risk (e.g. term_over_20y -> 民法典705), and drive Recall@k / MRR;
- *weak* tags are broad families (lease, contract_performance...) used only
  when no strong article exists; their recall is reported separately because
  a huge family makes Recall@k physically unattainable.

Queries are issued per issue_code (not per contract) so each risk is judged
on its own retrieval quality.

    python scripts/eval_legal_retrieval.py [--top-k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_review.legal_kb import LegalKB  # noqa: E402

ADVERSARIAL_MANIFEST = Path(
    "data/generated_contracts/samr_multi_template_2025/adversarial/manifest.json"
)
STATUTE_FILE = Path("data/法条结构化数据.jsonl")

# issue_code -> (strong tags, weak tags, retrieval query)
ISSUE_SPECS: dict[str, tuple[list[str], list[str], str]] = {
    "term_over_20y": (
        ["term_over_20y"], [], "租赁期限超过二十年 上限"
    ),
    "deposit_over_20pct": (
        ["deposit_over_20pct"], [], "定金不得超过主合同标的额的百分之二十"
    ),
    "penalty_excessive": (
        ["penalty_excessive"], [], "违约金过分高于造成的损失 适当减少"
    ),
    "unlimited_authority": (
        ["unlimited_authority"], ["delegation"], "受托人权限 委托人指示 超越权限"
    ),
    "no_report": (
        ["no_report"], [], "受托人报告义务 报告委托事务"
    ),
    "unilateral_entry": (
        ["unilateral_entry"], [], "出租人交付租赁物 承租人占有使用租赁物 不得妨碍"
    ),
    "unverified_property": (
        ["no_disposition_right", "title_registration"], ["lease"],
        "出租房屋权属 无权处分 权属证明",
    ),
    "liability_exemption": (
        ["standard_terms"], ["validity_review"], "免责条款 格式条款 无效"
    ),
    "no_permit": (
        ["no_permit", "presale_permit"], [], "预售许可 预售条件 批准"
    ),
    "date_reversed": (
        [], ["contract_performance", "delivery"], "合同履行期限 起始 结束 约定"
    ),
    "zero_rent": (
        [], ["lease"], "租赁合同 租金支付 数额"
    ),
    "zero_deposit": (
        [], ["lease"], "租赁押金 保证金约定"
    ),
    "extreme_rent": (
        [], ["lease", "breach_liability"], "租金数额 房屋租赁"
    ),
    "negative_fee": (
        [], ["contract_performance", "breach_liability"], "报酬 支付费用 履行"
    ),
    "fee_mismatch": (
        [], ["contract_performance", "breach_liability"], "费用约定 支付 履行"
    ),
    "impossible_deadline": (
        [], ["contract_performance", "delay"], "履行期限 交付时间 迟延"
    ),
    "insecure_delivery": (
        [], ["delivery", "safety", "risk_allocation"], "交付 标的物毁损灭失 风险"
    ),
    "commission_100pct": (
        [], ["intermediary"], "中介人报酬 中介合同"
    ),
    "negative_commission": (
        [], ["intermediary"], "中介人报酬 促成合同"
    ),
    "success_guarantee": (
        [], ["false_expression", "intermediary"], "中介人如实报告 不得故意隐瞒"
    ),
    "nonexistent_subject": (
        [], ["contract_formation", "validity_review"], "合同成立 标的 物权"
    ),
    "perpetual_license": (
        [], ["lease", "term_over_20y"], "使用期限 租赁期限"
    ),
    "personal_data_no_consent": (
        [], ["disclosure"], "个人信息 处理 同意"
    ),
    "public_disclosure": (
        [], ["disclosure"], "保密 个人信息 披露"
    ),
    "quantity_value_mismatch": (
        [], ["contract_performance"], "标的数量 质量 价款"
    ),
    "retrieval_before_storage": (
        [], [], "保管合同 寄存人 交付保管物"
    ),
    "zero_records": (
        [], [], "保管凭证 收据 寄存"
    ),
    "cash_only": (
        [], ["contract_performance"], "支付方式 价款支付"
    ),
    "zero_quantity": (
        [], ["contract_performance"], "标的数量 质量要求"
    ),
    "zero_fee": (
        [], ["contract_performance"], "报酬 费用 支付"
    ),
}

TEMPLATE_TO_TYPE: dict[str, str] = {
    "housing_rent": "房屋租赁",
    "entrust": "委托",
    "intermediary": "中介",
    "custody": "保管",
    "data_provide": "通用",
    "presale": "商品房预售",
    "sale": "房屋买卖",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 检索召回率评测")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/legal_index")
    args = parser.parse_args()

    manifest = json.loads(ADVERSARIAL_MANIFEST.read_text(encoding="utf-8"))
    records = manifest["records"]
    statutes = [
        json.loads(line)
        for line in STATUTE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kb = LegalKB(index_dir=Path(args.index_dir))

    stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"recall": [], "rank": []}
    )
    for record in records:
        contract_type = TEMPLATE_TO_TYPE.get(
            record.get("template_key", ""), "通用"
        )
        for code in record.get("issue_codes", []):
            spec = ISSUE_SPECS.get(code)
            if spec is None:
                continue
            strong, weak, query = spec
            wanted = set(strong) or set(weak)
            tier = "strong" if strong else "weak"
            if not wanted:
                continue
            expected = {
                a["article_uid"]
                for a in statutes
                if wanted & set(a.get("risk_tags", []))
            }
            if not expected:
                continue
            results = kb.search(
                query,
                contract_types=None,
                top_k=args.top_k,
            )
            ranked = [a.article_uid for a in results]
            first_hit = next(
                (r + 1 for r, uid in enumerate(ranked) if uid in expected), 0
            )
            recall = sum(1 for uid in ranked if uid in expected) / len(expected)
            stats[f"{tier}|{code}"]["recall"].append(recall)
            stats[f"{tier}|{code}"]["rank"].append(first_hit)

    kb.close()

    def mrr(ranks: list[float]) -> float:
        reciprocal = [1.0 / r for r in ranks if r > 0]
        return sum(reciprocal) / len(reciprocal) if reciprocal else 0.0

    strong_rows = [(k, v) for k, v in stats.items() if k.startswith("strong|")]
    weak_rows = [(k, v) for k, v in stats.items() if k.startswith("weak|")]

    def report(rows: list[tuple[str, dict]], title: str) -> None:
        print(f"\n== {title} ==")
        all_recalls: list[float] = []
        all_ranks: list[float] = []
        for key, v in sorted(rows):
            recall = sum(v["recall"]) / len(v["recall"])
            code = key.split("|", 1)[1]
            all_recalls.extend(v["recall"])
            all_ranks.extend(v["rank"])
            print(
                f"  {code:26s} n={len(v['recall'])} "
                f"Recall@{args.top_k}={recall:.3f} MRR={mrr(v['rank']):.3f}"
            )
        if all_recalls:
            print(
                f"  {'— 总体 —':26s} n={len(all_recalls)} "
                f"Recall@{args.top_k}={sum(all_recalls) / len(all_recalls):.3f} "
                f"MRR={mrr(all_ranks):.3f}"
            )

    report(strong_rows, "强金标准（法条直接对应风险）")
    report(weak_rows, "弱金标准（宽泛风险家族，仅供参考）")


if __name__ == "__main__":
    main()
