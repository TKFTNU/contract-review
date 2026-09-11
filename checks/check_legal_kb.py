from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from contract_review.legal_kb import (
    BM25Index,
    OllamaEmbedder,
    RetrievedArticle,
    rrf_fuse,
    tokenize,
)


DOCS = [
    "租赁期限不得超过二十年。超过二十年的，超过部分无效。",
    "定金不得超过主合同标的额的百分之二十，超过部分不产生定金的效力。",
    "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行等违约责任。",
]


class TokenizeChecks(unittest.TestCase):
    def test_unigrams_and_bigrams(self) -> None:
        tokens = tokenize("租赁期限")
        self.assertIn("租", tokens)
        self.assertIn("租赁", tokens)
        self.assertIn("期限", tokens)

    def test_number_runs_stay_atomic(self) -> None:
        tokens = tokenize("第705条 2026年")
        self.assertIn("705", tokens)
        self.assertIn("2026", tokens)

    def test_punctuation_dropped(self) -> None:
        self.assertNotIn("。", tokenize("十年。二十年"))


class BM25Checks(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index()
        self.index.fit(DOCS)

    def test_vocabulary_covers_documents(self) -> None:
        self.assertGreater(len(self.index.vocabulary), 50)
        self.assertEqual(len(self.index.doc_lengths), len(DOCS))

    def test_relevant_document_scores_higher(self) -> None:
        query = "租赁期限不得超过二十年"
        lease = self.index.score(query, 0)
        breach = self.index.score(query, 2)
        self.assertGreater(lease, breach)

    def test_query_sparse_ignores_unknown_terms(self) -> None:
        indices, values = self.index.query_sparse("租赁期限加上完全不存在的词量子泽")
        self.assertTrue(indices)
        self.assertTrue(all(v == 1.0 for v in values))
        unknown = "子泽"  # not in any statute above
        self.assertNotIn(self.index.vocabulary.get(unknown), indices)

    def test_document_sparse_weights_include_idf(self) -> None:
        indices, values = self.index.document_sparse(0)
        self.assertTrue(indices)
        self.assertTrue(all(v > 0 for v in values))
        self.assertEqual(len(indices), len(values))

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25_vocab.json"
            self.index.save(path)
            loaded = BM25Index.load(path)
            self.assertEqual(loaded.vocabulary, self.index.vocabulary)
            self.assertEqual(loaded.doc_lengths, self.index.doc_lengths)
            indices, values = loaded.query_sparse("租赁期限")
            self.assertTrue(indices)
            self.assertTrue(all(v == 1.0 for v in values))


class RRFFusionChecks(unittest.TestCase):
    def test_fuses_and_ranks(self) -> None:
        fused = rrf_fuse(
            [["a", "b", "c"], ["b", "a", "d"]], k=60, top_k=3
        )
        ids = [item for item, _ in fused]
        # Both channels agree a and b are top, so they outrank c and d.
        self.assertIn("a", ids[:2])
        self.assertIn("b", ids[:2])
        self.assertEqual(len(ids), 3)

    def test_single_channel_is_rank_order(self) -> None:
        fused = rrf_fuse([["x", "y", "z"]], top_k=2)
        self.assertEqual([item for item, _ in fused], ["x", "y"])


class EmbedderChecks(unittest.TestCase):
    def test_batches_and_parses_response(self) -> None:
        calls: list[list[str]] = []

        def fake_urlopen(request, timeout):
            payload = __import__("json").loads(request.data.decode("utf-8"))
            calls.append(payload["input"])
            body = {
                "embeddings": [
                    [0.1, 0.2] for _ in payload["input"]
                ]
            }

            class Response:
                def read(self) -> bytes:
                    return __import__("json").dumps(body).encode("utf-8")

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Response()

        embedder = OllamaEmbedder(
            type("Config", (), {"endpoint": "http://fake", "model": "bge-m3", "batch_size": 2, "timeout_seconds": 5})()
        )
        import contract_review.legal_kb as kb

        original = kb.urllib.request.urlopen
        kb.urllib.request.urlopen = fake_urlopen
        try:
            vectors = embedder.embed(["一", "二", "三"])
        finally:
            kb.urllib.request.urlopen = original
        self.assertEqual(len(vectors), 3)
        self.assertEqual([len(batch) for batch in calls], [2, 1])

    def _run_with_capture(self, config_fields: dict) -> dict:
        captured: dict = {}

        def fake_urlopen(request, timeout):
            captured["auth"] = request.get_header("Authorization")

            class Response:
                def read(self) -> bytes:
                    return b'{"embeddings": [[0.5]]}'

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Response()

        embedder = OllamaEmbedder(type("Config", (), config_fields)())
        import contract_review.legal_kb as kb

        original = kb.urllib.request.urlopen
        kb.urllib.request.urlopen = fake_urlopen
        try:
            embedder.embed(["x"])
        finally:
            kb.urllib.request.urlopen = original
        return captured

    def test_api_key_sends_bearer_header(self) -> None:
        captured = self._run_with_capture(
            {
                "endpoint": "http://fake",
                "model": "bge-m3",
                "batch_size": 2,
                "timeout_seconds": 5,
                "api_key": "sk-secret",
            }
        )
        self.assertEqual(captured["auth"], "Bearer sk-secret")

    def test_without_api_key_no_auth_header(self) -> None:
        # A duck-typed config without the api_key field must stay supported.
        captured = self._run_with_capture(
            {
                "endpoint": "http://fake",
                "model": "bge-m3",
                "batch_size": 2,
                "timeout_seconds": 5,
            }
        )
        self.assertIsNone(captured["auth"])


class ArticleChecks(unittest.TestCase):
    def test_citation_format(self) -> None:
        article = RetrievedArticle(
            article_uid="civil_code_2021_article_705",
            law_name="中华人民共和国民法典",
            article_no="第七百零五条",
            content="租赁期限不得超过二十年。",
        )
        self.assertEqual(
            article.citation(), "《中华人民共和国民法典》第七百零五条"
        )


if __name__ == "__main__":
    unittest.main()
