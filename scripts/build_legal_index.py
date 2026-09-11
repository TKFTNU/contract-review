"""Build the legal-article index for M5 retrieval.

Reads ``data/法条结构化数据.jsonl`` and writes a Qdrant local collection with
two named vectors per article:

- ``dense``  : BGE-M3 (1024d, cosine) via Ollama's /api/embed;
- ``bm25``   : character-level BM25 weights (FastEmbed-style: document side
  carries tf_sat * idf, the query side is all ones).

Run once after the statute file changes:

    python scripts/build_legal_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient, models  # noqa: E402

from contract_review.legal_kb import (  # noqa: E402
    BM25Index,
    COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    OllamaEmbedder,
    EmbedderConfig,
)

STATUTE_FILE = Path("data/法条结构化数据.jsonl")
PAYLOAD_FIELDS = (
    "law_name",
    "article_no",
    "content",
    "authority_level",
    "jurisdiction",
    "effective_from",
    "effective_to",
    "status",
    "contract_types",
    "risk_tags",
    "source_url",
    "version_id",
)


def load_articles(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=STATUTE_FILE,
        help="法条 JSONL 路径（默认 data/法条结构化数据.jsonl）",
    )
    args = parser.parse_args()

    articles = load_articles(args.input)
    print(f"载入法条 {len(articles)} 条（来源 {args.input}）")

    effective = [a for a in articles if a.get("status") == "effective"]
    skipped = len(articles) - len(effective)
    if skipped:
        print(f"跳过非生效条文 {skipped} 条")

    bm25 = BM25Index()
    bm25.fit([a["content"] for a in effective])
    print(f"BM25 词表 {len(bm25.vocabulary)} 项")

    embedder = OllamaEmbedder(EmbedderConfig())
    vectors = embedder.embed([a["content"] for a in effective])
    dim = len(vectors[0])
    print(f"bge-m3 向量 {len(vectors)} 条，维度 {dim}")

    if DEFAULT_INDEX_DIR.exists():
        import shutil

        shutil.rmtree(DEFAULT_INDEX_DIR)
    client = QdrantClient(path=str(DEFAULT_INDEX_DIR))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )

    points = []
    for position, article in enumerate(effective):
        sparse_indices, sparse_values = bm25.document_sparse(position)
        payload = {field: article.get(field) for field in PAYLOAD_FIELDS}
        payload["article_uid"] = article["article_uid"]
        points.append(
            models.PointStruct(
                id=position,
                vector={
                    "dense": vectors[position],
                    "bm25": models.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                },
                payload=payload,
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    client.close()
    DEFAULT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(DEFAULT_INDEX_DIR / "bm25_vocab.json")
    print(f"索引写入 {DEFAULT_INDEX_DIR}：{len(points)} 点")


if __name__ == "__main__":
    main()
