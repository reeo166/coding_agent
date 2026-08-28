"""Minimal DeepSeek/OpenAI-compatible Chat Completions HTTP client."""

from __future__ import annotations

import json
import socket
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import Settings


class APIError(RuntimeError):
    """A user-facing API or protocol error."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Never forward API credentials to a redirected origin."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class ChatCompletionClient:
    """Call DeepSeek's OpenAI-compatible /chat/completions endpoint.

    Only the model's native tool-calling protocol is used. No agent framework,
    hosted code runner, or hosted file API is involved.
    """

    RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._opener = build_opener(_NoRedirectHandler())

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }
        data = self._post_json(payload)
        return self._extract_message(data)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "local-coding-agent/0.1.0",
        }
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        headers.update(self.settings.extra_headers)

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            request = Request(self.settings.endpoint, data=body, headers=headers, method="POST")
            try:
                with self._opener.open(request, timeout=self.settings.timeout) as response:
                    raw = response.read().decode("utf-8")
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    raise APIError("API 返回的顶层 JSON 不是对象")
                if decoded.get("error"):
                    raise APIError(self._redact(self._error_message(decoded["error"])))
                return decoded
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                message = self._redact(self._http_error_message(exc.code, error_body))
                last_error = APIError(message)
                if exc.code not in self.RETRYABLE_STATUS or attempt >= self.settings.max_retries:
                    raise last_error from exc
            except (URLError, socket.timeout, TimeoutError) as exc:
                last_error = APIError(
                    self._redact(f"无法连接 API：{getattr(exc, 'reason', exc)}")
                )
                if attempt >= self.settings.max_retries:
                    raise last_error from exc
            except json.JSONDecodeError as exc:
                raise APIError("API 返回的不是有效 JSON") from exc
            if attempt < self.settings.max_retries:
                time.sleep(min(2**attempt, 4))
        raise APIError(str(last_error or "未知 API 错误"))

    def _redact(self, message: str) -> str:
        secrets = [self.settings.api_key, *self.settings.extra_headers.values()]
        redacted = message
        for secret in secrets:
            if len(secret) >= 4:
                redacted = redacted.replace(secret, "***")
        return redacted

    @staticmethod
    def _error_message(error: Any) -> str:
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(error)

    @classmethod
    def _http_error_message(cls, status: int, body: str) -> str:
        detail = body[:2_000]
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict) and "error" in decoded:
                detail = cls._error_message(decoded["error"])
        except json.JSONDecodeError:
            pass
        hint = ""
        if status in {401, 403}:
            hint = "；请检查 API Key、Base URL 和账号权限"
        elif status == 402:
            hint = "；DeepSeek 账户余额不足，请充值后重试"
        elif status == 404:
            hint = "；请检查 Base URL 和模型名是否正确"
        elif status == 429:
            hint = "；请求频率或额度可能已受限"
        return f"API HTTP {status}: {detail}{hint}"

    @classmethod
    def _extract_message(cls, data: dict[str, Any]) -> dict[str, Any]:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError("API 响应缺少 choices[0].message") from exc
        if not isinstance(message, dict):
            raise APIError("API 响应中的 message 不是对象")

        normalized: dict[str, Any] = {
            "role": "assistant",
            "content": cls._normalize_content(message.get("content")),
        }
        if "reasoning_content" in message:
            reasoning_content = message["reasoning_content"]
            if reasoning_content is not None and not isinstance(reasoning_content, str):
                raise APIError("API 响应中的 reasoning_content 不是字符串")
            normalized["reasoning_content"] = reasoning_content
        raw_calls = message.get("tool_calls") or []
        if not raw_calls and message.get("function_call"):
            raw_calls = [
                {
                    "id": f"legacy_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": message["function_call"],
                }
            ]
        if raw_calls:
            if not isinstance(raw_calls, list):
                raise APIError("API 响应中的 tool_calls 不是数组")
            normalized["tool_calls"] = [cls._normalize_tool_call(call) for call in raw_calls]
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish_reason == "length":
            raise APIError("模型输出达到长度上限，未得到完整最终答复")
        if finish_reason == "content_filter":
            raise APIError("模型输出被内容过滤器截断")
        if finish_reason == "insufficient_system_resource":
            raise APIError("DeepSeek 推理资源不足，模型输出被中断；请稍后重试")
        if not normalized.get("tool_calls") and not (normalized.get("content") or "").strip():
            raise APIError("模型既未返回文本，也未返回工具调用")
        return normalized

    @staticmethod
    def _normalize_content(content: Any) -> str | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for block in content:
                if isinstance(block, str):
                    pieces.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    pieces.append(block["text"])
            if pieces:
                return "\n".join(pieces)
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _normalize_tool_call(call: Any) -> dict[str, Any]:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise APIError("模型返回了格式无效的工具调用")
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise APIError("工具调用缺少函数名")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = str(arguments)
        return {
            "id": str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
