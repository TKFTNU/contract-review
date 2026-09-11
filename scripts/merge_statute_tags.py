"""Merge annotations from a curated statute file into a full-coverage file.

Joins on ``article_uid``:

- ``risk_tags`` and ``contract_types`` from the curated (old) file are unioned
  into the new file, keeping the new file's order first;
- ``article_no`` is normalized to start with 第 (the full-Civil-Code export
  omits the prefix).

    python scripts/merge_statute_tags.py \
        --new data/民法典法条结构化数据.jsonl \
        --old data/法条结构化数据.jsonl \
        --output data/民法典法条结构化数据.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    old_by_uid = {a["article_uid"]: a for a in load(args.old)}
    articles = load(args.new)

    prefix_fixed = tags_merged = types_merged = 0
    for article in articles:
        curated = old_by_uid.get(article["article_uid"])

        number = article.get("article_no", "")
        if number and not number.startswith("第"):
            article["article_no"] = "第" + number
            prefix_fixed += 1

        if not curated:
            continue
        current_tags = article.get("risk_tags") or []
        merged_tags = current_tags + [
            tag for tag in (curated.get("risk_tags") or []) if tag not in current_tags
        ]
        if merged_tags != current_tags:
            article["risk_tags"] = merged_tags
            tags_merged += 1

        current_types = article.get("contract_types") or []
        merged_types = current_types + [
            t for t in (curated.get("contract_types") or []) if t not in current_types
        ]
        if merged_types != current_types:
            article["contract_types"] = merged_types
            types_merged += 1

    args.output.write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in articles) + "\n",
        encoding="utf-8",
    )
    print(f"输出 {len(articles)} 条 → {args.output}")
    print(f"article_no 补“第”字 {prefix_fixed} 条 | risk_tags 合并 {tags_merged} 条 | contract_types 合并 {types_merged} 条")


if __name__ == "__main__":
    main()
