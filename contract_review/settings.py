"""Runtime configuration loaded from ``.env`` and environment variables.

Standard library only, matching the project's "no-dependency core" rule.

Resolution order (highest priority first):

1. real environment variables already set in the process;
2. ``KEY=VALUE`` pairs in the project's ``.env`` file;
3. the built-in defaults in :data:`DEFAULTS`.

Copy ``.env.example`` to ``.env`` and adjust it to point the pipeline at your
own model services. CLI flags (``--endpoint`` / ``--model`` / ...) still win:
argparse writes them over these defaults before any config object is built.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

#: Fallbacks used when neither the environment nor .env defines a key.
DEFAULTS: dict[str, str] = {
    # LM Studio (OpenAI-compatible chat completions): segmentation, element
    # extraction, consistency review and the legal judge all share this one.
    "CONTRACT_LLM_ENDPOINT": "http://localhost:12345",
    "CONTRACT_LLM_MODEL": "qwen3.8-27b-nvfp4-mtp",
    "CONTRACT_LLM_API_KEY": "",
    # Ollama: BGE-M3 embeddings for the legal knowledge-base retrieval.
    "CONTRACT_OLLAMA_ENDPOINT": "http://localhost:11434",
    "CONTRACT_EMBED_MODEL": "bge-m3",
    "CONTRACT_EMBED_API_KEY": "",
}


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``.env``: ``KEY=VALUE`` lines, ``#`` comments, quotes.

    Deliberately tiny - no interpolation, no multi-line values. Unknown or
    malformed lines are skipped rather than raising, so a stray line in the
    file can never break the pipeline at import time.
    """

    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


_FILE_VALUES: dict[str, str] = _parse_env_file(ENV_PATH)


def reload() -> None:
    """Re-read the ``.env`` file (useful for long-running servers and tests)."""

    global _FILE_VALUES
    _FILE_VALUES = _parse_env_file(ENV_PATH)


def get(name: str) -> str:
    """Environment variable > .env file > built-in default (may be empty)."""

    if name in os.environ:
        return os.environ[name]
    if name in _FILE_VALUES:
        return _FILE_VALUES[name]
    return DEFAULTS.get(name, "")


def llm_endpoint() -> str:
    return get("CONTRACT_LLM_ENDPOINT")


def llm_model() -> str:
    return get("CONTRACT_LLM_MODEL")


def llm_api_key() -> str:
    return get("CONTRACT_LLM_API_KEY")


def llm_timeout(default: int) -> int:
    """Optional ``CONTRACT_LLM_TIMEOUT`` override of a client's own default."""

    raw = os.environ.get("CONTRACT_LLM_TIMEOUT")
    if raw is None:
        raw = _FILE_VALUES.get("CONTRACT_LLM_TIMEOUT")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


def ollama_endpoint() -> str:
    return get("CONTRACT_OLLAMA_ENDPOINT")


def embed_model() -> str:
    return get("CONTRACT_EMBED_MODEL")


def embed_api_key() -> str:
    """Bearer token for the embedding endpoint (empty for stock local Ollama)."""

    return get("CONTRACT_EMBED_API_KEY")
