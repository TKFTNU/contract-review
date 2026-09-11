"""M5: legal knowledge base retrieval.

The index side lives in ``scripts/build_legal_index.py``; this module owns the
pieces that must be importable everywhere and testable offline:

- character-level BM25 over Chinese statutes (no tokenizer dependency);
- the Ollama BGE-M3 embedding client (plain HTTP, batched);
- the hybrid retriever (dense + BM25 sparse fused with RRF inside Qdrant).

Qdrant is imported lazily so the BM25 math and the offline evaluation stay
usable even before the client package is installed.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import settings


BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_INDEX_DIR = Path("data/legal_index")
COLLECTION_NAME = "legal_articles"


# --------------------------------------------------------------------------
# tokenization: character unigrams + bigrams survive Chinese statutes well
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[0-9a-z]+")


def tokenize(text: str) -> list[str]:
    """Split statute text into char unigrams + bigrams and latin/number runs."""

    units: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        units.append(match.group(0))
    tokens = list(units)
    for left, right in zip(units, units[1:]):
        # Only join two han characters; "3.5" style runs are already atomic.
        if len(left) == 1 and len(right) == 1:
            tokens.append(left + right)
    return tokens


# --------------------------------------------------------------------------
# BM25 index: vocabulary + IDF + per-document sparse weights
# --------------------------------------------------------------------------


class BM25Index:
    """Standard-library BM25 that feeds Qdrant sparse vectors.

    Document-side weight is ``tf_sat * idf`` and query-side weight is ``1.0``,
    so the sparse dot product inside Qdrant reproduces the BM25 score for
    unit-frequency queries — the same trick FastEmbed's BM25 uses.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.doc_frequency: list[int] = []
        self.term_freqs: list[dict[int, int]] = []
        self.doc_lengths: list[int] = []
        self._avgdl: float = 0.0

    # -- building ----------------------------------------------------------

    def fit(self, documents: list[str]) -> None:
        """Index every document and compute the corpus statistics."""

        tokenized = [tokenize(doc) for doc in documents]
        self.term_freqs = []
        self.doc_lengths = [len(tokens) for tokens in tokenized]
        df: dict[int, int] = {}
        for tokens in tokenized:
            counts: dict[int, int] = {}
            for token in tokens:
                term_id = self.vocabulary.setdefault(token, len(self.vocabulary))
                counts[term_id] = counts.get(term_id, 0) + 1
            self.term_freqs.append(counts)
            for term_id in counts:
                df[term_id] = df.get(term_id, 0) + 1
        total = len(documents)
        self.doc_frequency = [0] * len(self.vocabulary)
        for term_id, freq in df.items():
            self.doc_frequency[term_id] = freq
        # BM25+ style idf floors at a small positive value so a term that
        # appears in every document still carries a bit of signal.
        self.idf = [
            math.log((total - freq + 0.5) / (freq + 0.5) + 1.0)
            for freq in self.doc_frequency
        ]
        self._avgdl = (
            sum(self.doc_lengths) / total if total else 0.0
        )

    # -- sparse vectors ----------------------------------------------------

    def document_sparse(self, doc_index: int) -> tuple[list[int], list[float]]:
        """Return the (indices, values) sparse vector for one indexed doc."""

        length = self.doc_lengths[doc_index]
        denominator_bias = self.k1 * (1 - self.b + self.b * length / self._avgdl)
        indices: list[int] = []
        values: list[float] = []
        for term_id, tf in self.term_freqs[doc_index].items():
            sat = tf * (self.k1 + 1) / (tf + denominator_bias)
            indices.append(term_id)
            values.append(round(sat * self.idf[term_id], 6))
        return indices, values

    def query_sparse(self, query: str) -> tuple[list[int], list[float]]:
        """Return the query-side sparse vector; unknown terms are ignored.

        Values are all ``1.0`` — the IDF already lives on the document side.
        """

        seen: dict[int, None] = {}
        for token in tokenize(query):
            term_id = self.vocabulary.get(token)
            if term_id is not None:
                seen.setdefault(term_id, None)
        indices = list(seen)
        return indices, [1.0] * len(indices)

    # -- direct scoring (offline tests / debugging) -------------------------

    def score(self, query: str, doc_index: int) -> float:
        """Plain BM25 score of one document against the query."""

        length = self.doc_lengths[doc_index]
        denominator_bias = self.k1 * (1 - self.b + self.b * length / self._avgdl)
        total = 0.0
        for token in tokenize(query):
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            tf = self.term_freqs[doc_index].get(term_id, 0)
            if not tf:
                continue
            sat = tf * (self.k1 + 1) / (tf + denominator_bias)
            total += self.idf[term_id] * sat
        return total

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "vocabulary": self.vocabulary,
            "doc_frequency": self.doc_frequency,
            "doc_lengths": self.doc_lengths,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=payload["k1"], b=payload["b"])
        index.vocabulary = payload["vocabulary"]
        index.doc_frequency = payload["doc_frequency"]
        index.doc_lengths = payload["doc_lengths"]
        total = len(index.doc_lengths)
        index.idf = [
            math.log((total - freq + 0.5) / (freq + 0.5) + 1.0)
            for freq in index.doc_frequency
        ]
        index._avgdl = (
            sum(index.doc_lengths) / total if total else 0.0
        )
        # Term frequencies are not persisted: the retrieval path only needs
        # the sparse vectors stored in Qdrant, and score() is a test helper.
        index.term_freqs = [{} for _ in index.doc_lengths]
        return index


# --------------------------------------------------------------------------
# RRF fusion as a pure function (also used by the offline evaluation)
# --------------------------------------------------------------------------


def rrf_fuse(
    rankings: list[list[str]], k: int = 60, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion over ranked id lists; best score first."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position + 1)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if top_k is not None:
        fused = fused[:top_k]
    return fused


# --------------------------------------------------------------------------
# Ollama BGE-M3 embedding client (batched, plain HTTP)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    endpoint: str = field(default_factory=settings.ollama_endpoint)
    model: str = field(default_factory=settings.embed_model)
    # Empty for a stock local Ollama; set it when the embedding endpoint sits
    # behind a gateway that requires a Bearer token.
    api_key: str = field(default_factory=settings.embed_api_key)
    batch_size: int = 32
    timeout_seconds: int = 120


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Calls Ollama's native ``/api/embed`` which accepts batched input."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = max(1, self.config.batch_size)
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            payload = json.dumps(
                {"model": self.config.model, "input": chunk}
            ).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            # Duck-typed configs (tests pass plain objects) may omit api_key.
            api_key = getattr(self.config, "api_key", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(
                self.config.endpoint.rstrip("/") + "/api/embed",
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                raise RuntimeError(f"embedding 调用失败：{exc}") from exc
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(chunk):
                raise RuntimeError("embedding 响应数量与输入不一致。")
            vectors.extend(embeddings)
        return vectors


# --------------------------------------------------------------------------
# retrieved article + hybrid retriever
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievedArticle:
    article_uid: str
    law_name: str
    article_no: str
    content: str
    score: float = 0.0
    authority_level: str = ""
    risk_tags: list[str] = field(default_factory=list)
    contract_types: list[str] = field(default_factory=list)

    def citation(self) -> str:
        return f"《{self.law_name}》{self.article_no}"


class LegalKB:
    """Hybrid retriever over the Qdrant local collection."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embedder: Embedder | None = None,
        client: Any = None,
    ) -> None:
        index_dir = Path(index_dir)
        self.bm25 = BM25Index.load(index_dir / "bm25_vocab.json")
        self.embedder = embedder or OllamaEmbedder()
        if client is not None:
            self._client = client
        else:
            from qdrant_client import QdrantClient  # lazy: optional dependency

            try:
                self._client = QdrantClient(path=str(index_dir))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"法律索引目录 {index_dir} 被其他进程占用"
                    "（Qdrant 本地模式使用独占锁）。"
                    "请等待其他分析任务结束后重试，或改用 Qdrant 服务模式。"
                ) from exc

    def search(
        self,
        query: str,
        contract_types: list[str] | None = None,
        top_k: int = 8,
        candidate_limit: int = 50,
    ) -> list[RetrievedArticle]:
        """Dense + BM25 sparse fusion, optionally filtered by contract type.

        ``contract_types`` values come from the dataset templates
        (通用/房屋租赁/委托/中介/保管/商品房预售/房屋买卖); passing ``None``
        searches the whole collection.
        """

        dense_vector = self.embedder.embed([query])[0]
        sparse_indices, sparse_values = self.bm25.query_sparse(query)
        query_filter = None
        if contract_types:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchAny,
            )

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="contract_types",
                        match=MatchAny(any=list(contract_types)),
                    )
                ]
            )

        from qdrant_client.models import FusionQuery, Prefetch, SparseVector

        prefetch: list[Any] = [
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=candidate_limit,
                filter=query_filter,
            )
        ]
        if sparse_indices:
            prefetch.append(
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                    using="bm25",
                    limit=candidate_limit,
                    filter=query_filter,
                )
            )
        response = self._client.query_points(
            COLLECTION_NAME,
            prefetch=prefetch,
            query=FusionQuery(fusion="rrf"),
            limit=top_k,
            with_payload=True,
        )
        articles: list[RetrievedArticle] = []
        for point in response.points:
            payload = point.payload or {}
            articles.append(_article_from_payload(payload, point.score))
        return articles

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


def _article_from_payload(payload: dict[str, Any], score: float) -> RetrievedArticle:
    return RetrievedArticle(
        article_uid=str(payload.get("article_uid", "")),
        law_name=str(payload.get("law_name", "")),
        article_no=str(payload.get("article_no", "")),
        content=str(payload.get("content", "")),
        score=float(score or 0.0),
        authority_level=str(payload.get("authority_level", "")),
        risk_tags=list(payload.get("risk_tags", [])),
        contract_types=list(payload.get("contract_types", [])),
    )
