"""Settings tests: .env parsing, precedence, timeout overrides (fully offline)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from contract_review import settings


class EnvFileParsingTests(unittest.TestCase):
    def _parse(self, text: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(text, encoding="utf-8")
            return settings._parse_env_file(path)

    def test_basic_pairs_and_comments(self) -> None:
        parsed = self._parse(
            "# a comment\n"
            "\n"
            "KEY=value\n"
            "WITH_SPACE = spaced value\n"
            "export EXPORTED=ok\n"
        )
        self.assertEqual(parsed["KEY"], "value")
        self.assertEqual(parsed["WITH_SPACE"], "spaced value")
        self.assertEqual(parsed["EXPORTED"], "ok")

    def test_quotes_are_stripped(self) -> None:
        parsed = self._parse('A="double"\nB=\'single\'\n')
        self.assertEqual(parsed["A"], "double")
        self.assertEqual(parsed["B"], "single")

    def test_malformed_lines_are_skipped(self) -> None:
        parsed = self._parse("NO_EQUALS\n=EMPTY_KEY\nGOOD=x\n")
        self.assertEqual(parsed, {"GOOD": "x"})

    def test_empty_value_is_kept(self) -> None:
        # An empty API key is a legitimate value, not a missing one.
        parsed = self._parse("API_KEY=\n")
        self.assertEqual(parsed["API_KEY"], "")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(
            settings._parse_env_file(Path("Z:/definitely/not/here")), {}
        )


class _IsolatedSettings(unittest.TestCase):
    """Snapshot/restore the process environment and the parsed .env cache."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)
        self._orig_file = dict(settings._FILE_VALUES)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)
        settings._FILE_VALUES.clear()
        settings._FILE_VALUES.update(self._orig_file)

    def _clear(self, name: str) -> None:
        os.environ.pop(name, None)
        settings._FILE_VALUES.pop(name, None)


class PrecedenceTests(_IsolatedSettings):
    def test_env_var_beats_file(self) -> None:
        settings._FILE_VALUES["CONTRACT_LLM_ENDPOINT"] = "http://from-file"
        os.environ["CONTRACT_LLM_ENDPOINT"] = "http://from-env"
        self.assertEqual(settings.llm_endpoint(), "http://from-env")

    def test_file_beats_default(self) -> None:
        settings._FILE_VALUES["CONTRACT_LLM_ENDPOINT"] = "http://from-file"
        os.environ.pop("CONTRACT_LLM_ENDPOINT", None)
        self.assertEqual(settings.llm_endpoint(), "http://from-file")

    def test_default_when_unset(self) -> None:
        self._clear("CONTRACT_LLM_ENDPOINT")
        self.assertEqual(settings.llm_endpoint(), "http://localhost:12345")

    def test_timeout_falls_back_to_client_default(self) -> None:
        self._clear("CONTRACT_LLM_TIMEOUT")
        self.assertEqual(settings.llm_timeout(120), 120)

    def test_timeout_override_applies(self) -> None:
        settings._FILE_VALUES["CONTRACT_LLM_TIMEOUT"] = "45"
        os.environ.pop("CONTRACT_LLM_TIMEOUT", None)
        self.assertEqual(settings.llm_timeout(300), 45)

    def test_timeout_invalid_value_is_ignored(self) -> None:
        settings._FILE_VALUES["CONTRACT_LLM_TIMEOUT"] = "not-a-number"
        os.environ.pop("CONTRACT_LLM_TIMEOUT", None)
        self.assertEqual(settings.llm_timeout(300), 300)


class ConfigDefaultTests(_IsolatedSettings):
    def test_llm_configs_read_endpoint_model_and_key(self) -> None:
        os.environ["CONTRACT_LLM_ENDPOINT"] = "http://custom:9999"
        os.environ["CONTRACT_LLM_MODEL"] = "custom-model"
        os.environ["CONTRACT_LLM_API_KEY"] = "secret"

        from contract_review.agents import AgentReviewConfig
        from contract_review.extractor import M3Config
        from contract_review.llm_review import LLMReviewConfig
        from contract_review.llm_splitter import LLMSplitConfig
        from contract_review.reviewer import M4Config

        for config_cls in (
            M3Config,
            M4Config,
            LLMSplitConfig,
            LLMReviewConfig,
            AgentReviewConfig,
        ):
            config = config_cls()
            self.assertEqual(config.endpoint, "http://custom:9999")
            self.assertEqual(config.model, "custom-model")
            self.assertEqual(config.api_key, "secret")

    def test_embedder_config_reads_ollama_settings(self) -> None:
        os.environ["CONTRACT_OLLAMA_ENDPOINT"] = "http://ollama:1111"
        os.environ["CONTRACT_EMBED_MODEL"] = "bge-custom"
        os.environ["CONTRACT_EMBED_API_KEY"] = "sk-embed"

        from contract_review.legal_kb import EmbedderConfig

        config = EmbedderConfig()
        self.assertEqual(config.endpoint, "http://ollama:1111")
        self.assertEqual(config.model, "bge-custom")
        self.assertEqual(config.api_key, "sk-embed")


if __name__ == "__main__":
    unittest.main()
