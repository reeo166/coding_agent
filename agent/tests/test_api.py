from __future__ import annotations

import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from coding_agent.api import APIError, ChatCompletionClient
from coding_agent.config import Settings


class FakeResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.stream = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.stream.read()


class RedirectTargetHandler(BaseHTTPRequestHandler):
    authorization: str | None = None

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        type(self).authorization = self.headers.get("Authorization")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"choices":[{"message":{"content":"wrong origin"}}]}')

    def log_message(self, _format: str, *args: object) -> None:
        return


class RedirectSourceHandler(BaseHTTPRequestHandler):
    location = ""

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        self.send_response(302)
        self.send_header("Location", type(self).location)
        self.end_headers()

    def log_message(self, _format: str, *args: object) -> None:
        return


class APIClientTests(unittest.TestCase):
    def test_endpoint_and_tool_call_normalization(self) -> None:
        settings = Settings(
            api_key="secret",
            base_url="https://example.test/v1/",
            model="tool-model",
            max_retries=1,
        )
        client = ChatCompletionClient(settings)
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "I should inspect the files first.",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "list_files",
                                    "arguments": {},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        with patch.object(client._opener, "open", return_value=FakeResponse(response)) as mocked:
            message = client.complete([{"role": "user", "content": "hi"}], [])

        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        self.assertEqual(
            message["reasoning_content"], "I should inspect the files first."
        )
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], "{}")

    def test_invalid_reasoning_content_is_rejected(self) -> None:
        with self.assertRaisesRegex(APIError, "reasoning_content"):
            ChatCompletionClient._extract_message(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "answer",
                                "reasoning_content": {"unexpected": "object"},
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

    def test_full_endpoint_with_query_is_preserved(self) -> None:
        settings = Settings(
            api_key="",
            base_url="https://example.test/openai/chat/completions?api-version=next",
            model="tool-model",
        )
        self.assertEqual(
            settings.endpoint,
            "https://example.test/openai/chat/completions?api-version=next",
        )
        self.assertEqual(
            settings.display_endpoint,
            "https://example.test/openai/chat/completions?<query-hidden>",
        )

    def test_length_finish_reason_is_not_reported_as_success(self) -> None:
        with self.assertRaises(APIError):
            ChatCompletionClient._extract_message(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "partial"},
                            "finish_reason": "length",
                        }
                    ]
                }
            )

    def test_deepseek_resource_interruption_is_not_reported_as_success(self) -> None:
        with self.assertRaisesRegex(APIError, "推理资源不足"):
            ChatCompletionClient._extract_message(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "partial"},
                            "finish_reason": "insufficient_system_resource",
                        }
                    ]
                }
            )

    def test_deepseek_insufficient_balance_hint(self) -> None:
        message = ChatCompletionClient._http_error_message(402, '{"error":"balance"}')
        self.assertIn("余额不足", message)

    def test_truncated_tool_call_is_not_executed(self) -> None:
        with self.assertRaises(APIError):
            ChatCompletionClient._extract_message(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "partial",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": '{"path":"a.txt","content":"partial"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "length",
                        }
                    ]
                }
            )

    def test_redirect_is_rejected_without_forwarding_authorization(self) -> None:
        target = ThreadingHTTPServer(("127.0.0.1", 0), RedirectTargetHandler)
        source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectSourceHandler)
        RedirectTargetHandler.authorization = None
        RedirectSourceHandler.location = f"http://127.0.0.1:{target.server_port}/capture"
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        source_thread = threading.Thread(target=source.serve_forever, daemon=True)
        target_thread.start()
        source_thread.start()
        try:
            client = ChatCompletionClient(
                Settings(
                    api_key="synthetic-secret",
                    base_url=f"http://127.0.0.1:{source.server_port}/v1",
                    model="mock",
                    max_retries=0,
                )
            )
            with self.assertRaises(APIError):
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            source.shutdown()
            target.shutdown()
            source.server_close()
            target.server_close()
            source_thread.join(timeout=5)
            target_thread.join(timeout=5)
        self.assertIsNone(RedirectTargetHandler.authorization)

    def test_success_status_error_body_is_redacted(self) -> None:
        settings = Settings(
            api_key="synthetic-secret",
            base_url="https://example.test/v1",
            model="mock",
            max_retries=0,
            extra_headers={"X-Token": "header-secret"},
        )
        client = ChatCompletionClient(settings)
        response = {
            "error": {"message": "synthetic-secret and header-secret must not leak"}
        }
        with patch.object(client._opener, "open", return_value=FakeResponse(response)):
            with self.assertRaises(APIError) as raised:
                client.complete([{"role": "user", "content": "hi"}], [])
        message = str(raised.exception)
        self.assertNotIn("synthetic-secret", message)
        self.assertNotIn("header-secret", message)


if __name__ == "__main__":
    unittest.main()
