from __future__ import annotations

import html
import secrets
import threading
import time
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .parsers import MAX_FILE_SIZE, ParseError, validate_upload


MAX_REQUEST_SIZE = MAX_FILE_SIZE + 1024 * 1024
UPLOAD_TTL_SECONDS = 30 * 60
RETURN_URL = "http://localhost:8501/"


@dataclass(frozen=True, slots=True)
class UploadedContract:
    filename: str
    content: bytes


_LOCK = threading.Lock()
_UPLOADS: dict[str, tuple[float, UploadedContract]] = {}
_SERVER: ThreadingHTTPServer | None = None
_SERVER_THREAD: threading.Thread | None = None
_SERVER_NONCE = secrets.token_urlsafe(24)


def _cleanup_expired() -> None:
    cutoff = time.monotonic() - UPLOAD_TTL_SECONDS
    expired = [token for token, (created, _) in _UPLOADS.items() if created < cutoff]
    for token in expired:
        _UPLOADS.pop(token, None)


def _store_upload(filename: str, content: bytes) -> str:
    token = secrets.token_urlsafe(32)
    with _LOCK:
        _cleanup_expired()
        _UPLOADS[token] = (time.monotonic(), UploadedContract(filename, content))
    return token


def claim_upload(token: str) -> UploadedContract | None:
    if not token:
        return None
    with _LOCK:
        _cleanup_expired()
        stored = _UPLOADS.pop(token, None)
    return stored[1] if stored else None


def parse_multipart_file(content_type: str, body: bytes) -> UploadedContract:
    if "\r" in content_type or "\n" in content_type:
        raise ParseError("上传格式无效。")
    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    for part in message.walk():
        filename = part.get_filename()
        if not filename:
            continue
        content = part.get_payload(decode=True) or b""
        safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        validate_upload(safe_filename, content)
        return UploadedContract(safe_filename, content)
    raise ParseError("请求中没有找到合同文件。")


def _upload_page(port: int) -> bytes:
    action = f"http://127.0.0.1:{port}/upload?nonce={_SERVER_NONCE}"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>选择合同文件</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fb;margin:0;color:#182230}}
.card{{max-width:620px;margin:10vh auto;background:white;border:1px solid #dfe5ee;border-radius:16px;padding:32px;box-shadow:0 8px 30px #24324a18}}
h1{{font-size:24px;margin:0 0 10px}} p{{color:#526071;line-height:1.6}}
input{{display:block;width:100%;box-sizing:border-box;border:1px dashed #8fa1b7;border-radius:12px;padding:24px;margin:24px 0;background:#f9fbfd}}
button{{width:100%;border:0;border-radius:10px;padding:12px 18px;background:#ff4b4b;color:white;font-size:16px;font-weight:600;cursor:pointer}}
.hint{{font-size:13px;color:#6d7888}}
</style></head><body><main class="card">
<h1>上传合同</h1><p>选择 DOCX、文本型 PDF 或 TXT。文件仅传到本机解析服务。</p>
<form action="{html.escape(action)}" method="post" enctype="multipart/form-data">
<input type="file" name="contract" accept=".docx,.pdf,.txt" required>
<button type="submit">上传并开始解析</button></form>
<div class="hint">单个文件不超过25 MB。上传完成后会自动返回合同解析页面。</div>
</main></body></html>""".encode("utf-8")


class _UploadHandler(BaseHTTPRequestHandler):
    server_version = "ContractUpload/1.0"

    def _send_html(self, status: int, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _valid_nonce(self) -> bool:
        query = parse_qs(urlparse(self.path).query)
        provided = query.get("nonce", [""])[0]
        return secrets.compare_digest(provided, _SERVER_NONCE)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_html(200, b"ok")
            return
        if parsed.path != "/" or not self._valid_nonce():
            self._send_html(403, "无效或已过期的上传入口。".encode("utf-8"))
            return
        port = int(self.server.server_address[1])
        self._send_html(200, _upload_page(port))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path != "/upload" or not self._valid_nonce():
            self._send_html(403, "无效或已过期的上传请求。".encode("utf-8"))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_SIZE:
            self._send_html(413, "上传内容为空或超过25 MB。".encode("utf-8"))
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_html(400, "上传格式无效。".encode("utf-8"))
            return
        try:
            uploaded = parse_multipart_file(content_type, self.rfile.read(length))
            token = _store_upload(uploaded.filename, uploaded.content)
        except ParseError as exc:
            self._send_html(400, html.escape(str(exc)).encode("utf-8"))
            return
        self.send_response(303)
        self.send_header("Location", f"{RETURN_URL}?upload_token={token}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def start_upload_server(preferred_port: int = 8505) -> tuple[int, str]:
    global _SERVER, _SERVER_THREAD
    with _LOCK:
        if _SERVER is not None:
            return int(_SERVER.server_address[1]), _SERVER_NONCE
        last_error: OSError | None = None
        for port in range(preferred_port, preferred_port + 10):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), _UploadHandler)
            except OSError as exc:
                last_error = exc
                continue
            _SERVER = server
            server.daemon_threads = True
            _SERVER_THREAD = threading.Thread(
                target=server.serve_forever,
                name="contract-upload-server",
                daemon=True,
            )
            _SERVER_THREAD.start()
            return port, _SERVER_NONCE
        raise RuntimeError("无法启动本地上传服务。") from last_error


def upload_page_url(port: int, nonce: str) -> str:
    return f"http://127.0.0.1:{port}/?nonce={nonce}"
