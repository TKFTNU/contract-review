"""Lightweight single-page frontend served with the standard library only.

Run with::

    python -m contract_review.web_server
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agents import AgentReviewConfig, run_unified_review
from .extractor import M3Config, extract_elements
from .llm_client import LLMClientError
from .llm_splitter import LLMSplitConfig, split_contract_with_llm
from .models import Contract
from .parsers import ParseError, validate_upload
from .service import build_contract, parse_contract
from .upload_server import parse_multipart_file


DEFAULT_PORT = 8600
SAMPLE_PATH = (
    Path(__file__).resolve().parent.parent / "samples" / "tech-development-contract.txt"
)
UI_PATH = Path(__file__).resolve().parent / "webui.html"
DOCUMENT_TTL_SECONDS = 60 * 60
MAX_REQUEST_SIZE = 30 * 1024 * 1024

_LOCK = threading.Lock()
_DOCUMENTS: dict[str, tuple[float, str, bytes]] = {}
# Latest contract object per document, so the M2/M3 views reflect whatever
# segmentation and extraction the page is currently showing.
_CONTRACTS: dict[str, Contract] = {}

# Qdrant's local mode holds an exclusive lock per directory, so the server
# keeps one shared retriever for its lifetime and serializes M5 runs.
_LEGAL_LOCK = threading.Lock()
_KB = None


def _get_kb():
    """Lazy process-wide retriever; None means retrieval is unavailable."""

    global _KB
    if _KB is not None:
        return _KB
    try:
        from .legal_kb import EmbedderConfig, LegalKB, OllamaEmbedder

        _KB = LegalKB(embedder=OllamaEmbedder(EmbedderConfig()))
    except Exception:  # noqa: BLE001 - M5 degrades to no-retrieval mode
        return None
    return _KB


def _cleanup_documents() -> None:
    cutoff = time.monotonic() - DOCUMENT_TTL_SECONDS
    for doc_id, (created, _, _) in list(_DOCUMENTS.items()):
        if created < cutoff:
            _DOCUMENTS.pop(doc_id, None)
            _CONTRACTS.pop(doc_id, None)


def store_document(filename: str, content: bytes) -> str:
    doc_id = uuid.uuid4().hex
    with _LOCK:
        _cleanup_documents()
        _DOCUMENTS[doc_id] = (time.monotonic(), filename, content)
    return doc_id


def load_document(doc_id: str) -> tuple[str, bytes] | None:
    if not doc_id:
        return None
    with _LOCK:
        _cleanup_documents()
        stored = _DOCUMENTS.get(doc_id)
    return (stored[1], stored[2]) if stored else None


class _Handler(BaseHTTPRequestHandler):
    server_version = "ContractFrontend/1.0"
    protocol_version = "HTTP/1.1"

    # ---------- helpers ----------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        if length > MAX_REQUEST_SIZE:
            # Draining a body this large is not bounded; close the connection
            # so leftover bytes cannot poison the next pipelined request.
            self.close_connection = True
            return b""
        return self.rfile.read(length)

    def _read_json(self) -> dict:
        body = self._read_body()
        if not body:
            return {}
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _document_payload(self, filename: str, content: bytes, doc_id: str) -> dict:
        result = parse_contract(filename, content)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "result": result.to_dict(),
        }

    # ---------- routes ----------
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            try:
                html = UI_PATH.read_bytes()
            except OSError:
                self._send(500, b"webui.html not found", "text/plain; charset=utf-8")
                return
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            # Browsers request this automatically on every page load. An
            # abnormal reply here (501/404 with connection teardown) closes a
            # keep-alive connection the browser may reuse for the very next
            # upload POST, which surfaced as "first upload fails, second
            # works". A concrete empty-icon answer keeps the connection clean.
            self._send(204, b"", "image/x-icon")
            return
        if path == "/health":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/sample":
            self._handle_sample()
        elif path == "/api/upload":
            self._handle_upload()
        elif path == "/api/parse":
            self._handle_parse()
        elif path == "/api/split":
            self._handle_split()
        elif path == "/api/contract":
            self._handle_contract()
        elif path == "/api/extract":
            self._handle_extract()
        elif path in {"/api/review", "/api/legal"}:
            # Both routes run the same M6 unified graph; /api/legal remains
            # as a compatibility alias.
            self._handle_review()
        else:
            # Unknown endpoint: the body was not drained by any handler, so
            # force-close instead of leaving garbage on the keep-alive socket.
            self.close_connection = True
            self._send_json(404, {"error": "未知接口。"})

    # ---------- handlers ----------
    def _handle_sample(self) -> None:
        # The page posts an empty JSON object ("{}") here. It must be drained
        # even though it is ignored: on a keep-alive HTTP/1.1 connection the
        # unread bytes would be parsed as the start of the next request line
        # ("{}POST /api/upload"), producing 501s and a broken first upload.
        self._read_body()
        try:
            content = SAMPLE_PATH.read_bytes()
        except OSError:
            self._send_json(404, {"error": "内置样例合同不存在。"})
            return
        filename = SAMPLE_PATH.name
        try:
            payload = self._document_payload(
                filename, content, store_document(filename, content)
            )
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, payload)

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(400, {"error": "上传格式无效。"})
            return
        body = self._read_body()
        if not body:
            self._send_json(413, {"error": "上传内容为空或超过 30 MB。"})
            return
        try:
            uploaded = parse_multipart_file(content_type, body)
            validate_upload(uploaded.filename, uploaded.content)
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        try:
            payload = self._document_payload(
                uploaded.filename,
                uploaded.content,
                store_document(uploaded.filename, uploaded.content),
            )
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, payload)

    def _handle_parse(self) -> None:
        payload = self._read_json()
        doc_id = str(payload.get("doc_id", ""))
        loaded = load_document(doc_id)
        if loaded is None:
            self._send_json(404, {"error": "文档不存在或已过期，请重新上传。"})
            return
        filename, content = loaded
        try:
            parsed = parse_contract(filename, content)
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        _CONTRACTS[doc_id] = build_contract(parsed)
        self._send_json(
            200,
            {
                "doc_id": doc_id,
                "filename": filename,
                "result": parsed.to_dict(),
            },
        )

    def _contract_for(self, doc_id: str, filename: str, content: bytes) -> Contract:
        contract = _CONTRACTS.get(doc_id)
        if contract is None:
            contract = build_contract(parse_contract(filename, content))
            _CONTRACTS[doc_id] = contract
        return contract

    def _handle_contract(self) -> None:
        """Build the M2 unified contract for the segmentation on screen."""

        payload = self._read_json()
        doc_id = str(payload.get("doc_id", ""))
        loaded = load_document(doc_id)
        if loaded is None:
            self._send_json(404, {"error": "文档不存在或已过期，请重新上传。"})
            return
        filename, content = loaded
        try:
            contract = self._contract_for(doc_id, filename, content)
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(
            200,
            {
                "doc_id": doc_id,
                "filename": filename,
                "contract": contract.to_dict(),
            },
        )

    def _handle_extract(self) -> None:
        """M3: fill the element slots with parties/amounts/dates/subjects."""

        payload = self._read_json()
        doc_id = str(payload.get("doc_id", ""))
        loaded = load_document(doc_id)
        if loaded is None:
            self._send_json(404, {"error": "文档不存在或已过期，请重新上传。"})
            return
        filename, content = loaded
        try:
            contract = self._contract_for(doc_id, filename, content)
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        # Read defaults from an instance: slots=True dataclasses expose slot
        # descriptors instead of default values on the class itself.
        defaults = M3Config()
        config = M3Config(
            endpoint=str(payload.get("endpoint") or defaults.endpoint),
            model=str(payload.get("model") or defaults.model),
            api_key=str(payload.get("api_key") or ""),
            enable_llm=bool(payload.get("enable_llm", True)),
        )
        started = time.monotonic()
        try:
            contract, summary = extract_elements(contract, config)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the page
            self._send_json(500, {"error": f"要素抽取异常：{exc}"})
            return
        _CONTRACTS[doc_id] = contract
        self._send_json(
            200,
            {
                "doc_id": doc_id,
                "filename": filename,
                "contract": contract.to_dict(),
                "summary": {
                    "merged_elements": summary.merged_elements,
                    "rule_elements": summary.rule_elements,
                    "llm_elements": summary.llm_elements,
                    "by_kind": summary.by_kind,
                    "batches": summary.batches,
                    "succeeded_batches": summary.succeeded_batches,
                    "errors": summary.errors,
                    "elapsed": round(time.monotonic() - started, 1),
                },
            },
        )

    def _handle_review(self) -> None:
        """综合审查（M6 统一图）：要素+双路审查+检索+裁判+仲裁+融合。"""

        payload = self._read_json()
        doc_id = str(payload.get("doc_id", ""))
        loaded = load_document(doc_id)
        if loaded is None:
            self._send_json(404, {"error": "文档不存在或已过期，请重新上传。"})
            return
        filename, content = loaded
        try:
            contract = self._contract_for(doc_id, filename, content)
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        defaults = AgentReviewConfig()
        config = AgentReviewConfig(
            endpoint=str(payload.get("endpoint") or defaults.endpoint),
            model=str(payload.get("model") or defaults.model),
            api_key=str(payload.get("api_key") or ""),
            enable_llm=bool(payload.get("enable_llm", True)),
        )
        started = time.monotonic()
        try:
            with _LEGAL_LOCK:
                report = run_unified_review(contract, config, kb=_get_kb())
        except Exception as exc:  # noqa: BLE001 - surface any failure to the page
            self._send_json(500, {"error": f"综合审查异常：{exc}"})
            return
        self._send_json(
            200,
            {
                "doc_id": doc_id,
                "filename": filename,
                "report": report,
                "summary": {
                    **report["stats"],
                    "overall": report["overall"],
                    "errors": report["errors"],
                    "elapsed": round(time.monotonic() - started, 1),
                },
            },
        )

    def _handle_split(self) -> None:
        payload = self._read_json()
        doc_id = str(payload.get("doc_id", ""))
        loaded = load_document(doc_id)
        if loaded is None:
            self._send_json(404, {"error": "文档不存在或已过期，请重新上传。"})
            return
        filename, content = loaded
        # Read defaults from an instance: slots=True dataclasses expose slot
        # descriptors instead of default values on the class itself.
        defaults = LLMSplitConfig()
        config = LLMSplitConfig(
            endpoint=str(payload.get("endpoint") or defaults.endpoint),
            model=str(payload.get("model") or defaults.model),
            api_key=str(payload.get("api_key") or ""),
        )
        try:
            base = parse_contract(filename, content)
        except ParseError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        started = time.monotonic()
        try:
            result, summary = split_contract_with_llm(base, config)
        except LLMClientError as exc:
            self._send_json(502, {"error": f"无法连接模型服务：{exc}"})
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure to the page
            self._send_json(500, {"error": f"LLM分割异常：{exc}"})
            return

        if summary.succeeded_batches == 0:
            self._send_json(
                502,
                {
                    "error": "模型服务不可用，已保留规则切分结果："
                    + "；".join(summary.errors)
                },
            )
            return

        _CONTRACTS[doc_id] = build_contract(result)
        self._send_json(
            200,
            {
                "doc_id": doc_id,
                "filename": filename,
                "result": result.to_dict(),
                "summary": {
                    "batches": summary.batches,
                    "succeeded_batches": summary.succeeded_batches,
                    "rule_clauses": summary.rule_clauses,
                    "llm_clauses": summary.llm_clauses,
                    "errors": summary.errors,
                    "elapsed": round(time.monotonic() - started, 1),
                    "reflow_rounds": summary.reflow_rounds,
                    "reflowed_blocks": summary.reflowed_blocks,
                    "issues_before": summary.issues_before,
                    "issues_after": summary.issues_after,
                    "reflow_notes": summary.reflow_notes,
                },
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        # Restore the default access trace (and, via log_error, malformed
        # request diagnostics); log_error funnels through this method too.
        import sys

        sys.stderr.write("[web] %s\n" % (format % args))


def _bind(preferred_port: int, attempts: int = 30) -> ThreadingHTTPServer:
    """Bind the first free port, since Windows HTTP.SYS often squats on ports."""

    last_error: OSError | None = None
    for candidate in range(preferred_port, preferred_port + attempts):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), _Handler)
            server.daemon_threads = True
            return server
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"端口 {preferred_port} 起连续 {attempts} 个端口都被占用。") from last_error


def serve(port: int = DEFAULT_PORT) -> None:
    server = _bind(port)
    actual_port = int(server.server_address[1])
    print(f"合同条款切分前端已启动：http://127.0.0.1:{actual_port}", flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="合同条款切分轻量前端")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
