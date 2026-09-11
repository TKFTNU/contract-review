from __future__ import annotations

import unittest

from contract_review.upload_server import (
    _store_upload,
    claim_upload,
    parse_multipart_file,
)


class UploadServerChecks(unittest.TestCase):
    def test_parses_multipart_and_claims_token_once(self) -> None:
        boundary = "----contract-demo-boundary"
        content = "第一条 服务内容".encode("utf-8")
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="contract"; filename="demo.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

        uploaded = parse_multipart_file(
            f"multipart/form-data; boundary={boundary}", body
        )
        self.assertEqual(uploaded.filename, "demo.txt")
        self.assertEqual(uploaded.content, content)

        token = _store_upload(uploaded.filename, uploaded.content)
        claimed = claim_upload(token)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.content, content)
        self.assertIsNone(claim_upload(token))


if __name__ == "__main__":
    unittest.main()
