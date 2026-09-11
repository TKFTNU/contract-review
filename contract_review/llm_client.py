from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]
DEFAULT_MAX_TOKENS = 8_000


class LLMClientError(RuntimeError):
    """Raised when an OpenAI-compatible model cannot produce valid JSON output."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def chat_url(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMClientError(f"模型服务返回HTTP {exc.code}：{detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise LLMClientError(f"无法连接模型服务：{exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMClientError("模型请求超时。") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMClientError("模型服务返回的响应不是有效JSON。") from exc


def build_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _extract_json(response: dict[str, Any]) -> dict[str, Any]:
    if "choices" not in response:
        # Test doubles and future proxy layers may return the payload directly.
        if isinstance(response, dict):
            return response
        raise LLMClientError("模型响应格式无法识别。")
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("模型响应缺少choices[0].message.content。") from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        raise LLMClientError("模型响应content不是文本。")
    content = content.strip()
    if not content:
        raise LLMClientError("模型返回了空内容，可能是思考未关闭或超出长度限制。")
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMClientError("模型没有返回有效JSON。") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("模型返回的JSON必须是对象。")
    return parsed


def request_json(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    schema_name: str,
    schema: dict[str, Any],
    api_key: str = "",
    timeout_seconds: int = 300,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    seed: int = 42,
    disable_thinking: bool = True,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Call one chat completion and return the parsed JSON object of the reply."""

    active = transport or post_json
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "response_format": build_response_format(schema_name, schema),
    }
    if disable_thinking:
        # LM Studio ignores chat_template_kwargs and /no_think, but honors
        # reasoning_effort="none"; verify with a small probe before changing.
        payload["reasoning_effort"] = "none"
    try:
        response = active(chat_url(endpoint), payload, headers, timeout_seconds)
    except LLMClientError as exc:
        # vLLM/SGLang reject unknown fields with HTTP 4xx; retry without it.
        if (
            not disable_thinking
            or exc.status is None
            or not 400 <= exc.status < 500
        ):
            raise
        payload.pop("reasoning_effort", None)
        response = active(chat_url(endpoint), payload, headers, timeout_seconds)
    return _extract_json(response)
