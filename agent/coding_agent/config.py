"""Configuration loading without third-party dependencies."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is invalid."""


ALLOWED_DOTENV_KEYS = {
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "CODING_AGENT_API_KEY",
    "CODING_AGENT_BASE_URL",
    "CODING_AGENT_MODEL",
    "CODING_AGENT_EXTRA_HEADERS",
    "CODING_AGENT_TIMEOUT",
    "CODING_AGENT_MAX_STEPS",
    "CODING_AGENT_MAX_CONTEXT_CHARS",
    "CODING_AGENT_MAX_RETRIES",
}


def load_env_file(path: Path) -> None:
    """Load a small, predictable subset of dotenv syntax.

    Existing environment variables win. This keeps CI and shell configuration
    authoritative and avoids adding python-dotenv as a runtime dependency.
    """

    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"无法读取配置文件 {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"{path.name} 第 {line_number} 行缺少 '='")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
            raise ConfigurationError(f"{path.name} 第 {line_number} 行变量名无效")
        if key not in ALLOWED_DOTENV_KEYS:
            raise ConfigurationError(
                f"{path.name} 第 {line_number} 行变量 {key} 不在允许白名单中"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _api_key() -> str:
    return _first_env("DEEPSEEK_API_KEY", "CODING_AGENT_API_KEY") or ""


def _positive_int(value: int | str | None, default: int, label: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} 必须是整数") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{label} 必须大于 0")
    return parsed


def _nonnegative_int(value: int | str | None, default: int, label: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} 必须是整数") from exc
    if parsed < 0:
        raise ConfigurationError(f"{label} 不能小于 0")
    return parsed


def _positive_float(value: float | str | None, default: float, label: str) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} 必须是数字") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigurationError(f"{label} 必须大于 0")
    return parsed


def _extra_headers() -> dict[str, str]:
    raw = _first_env("CODING_AGENT_EXTRA_HEADERS")
    if not raw:
        return {}
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("CODING_AGENT_EXTRA_HEADERS 必须是 JSON 对象") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ConfigurationError("CODING_AGENT_EXTRA_HEADERS 的键和值都必须是字符串")
    if any(
        not key.strip() or "\r" in key or "\n" in key or "\r" in item or "\n" in item
        for key, item in value.items()
    ):
        raise ConfigurationError("CODING_AGENT_EXTRA_HEADERS 不能包含空名称或换行符")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    timeout: float = 120.0
    max_steps: int = 30
    max_context_chars: int = 120_000
    max_retries: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    @property
    def display_endpoint(self) -> str:
        """Endpoint safe for logs (query values and fragments are not printed)."""

        parsed = urlsplit(self.endpoint)
        query = "<query-hidden>" if parsed.query else ""
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))

    @classmethod
    def from_sources(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        max_context_chars: int | None = None,
        timeout: float | None = None,
    ) -> "Settings":
        resolved_base = (
            base_url
            or _first_env("DEEPSEEK_BASE_URL", "CODING_AGENT_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        parsed_url = urlparse(resolved_base)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigurationError("API Base URL 必须是完整的 http(s) 地址")
        if parsed_url.username or parsed_url.password:
            raise ConfigurationError("不要把凭据放进 Base URL；请使用 API Key 或自定义请求头")

        resolved_model = (
            model
            or _first_env("DEEPSEEK_MODEL", "CODING_AGENT_MODEL")
            or "deepseek-v4-flash"
        )

        resolved_timeout = timeout if timeout is not None else _first_env("CODING_AGENT_TIMEOUT")
        resolved_steps = (
            max_steps if max_steps is not None else _first_env("CODING_AGENT_MAX_STEPS")
        )
        resolved_context = (
            max_context_chars
            if max_context_chars is not None
            else _first_env("CODING_AGENT_MAX_CONTEXT_CHARS")
        )
        context_chars = _positive_int(
            resolved_context, 120_000, "max_context_chars"
        )
        if context_chars < 4_000:
            raise ConfigurationError("max_context_chars 不能小于 4000")

        api_key = _api_key()
        extra_headers = _extra_headers()
        hostname = (parsed_url.hostname or "").casefold()
        is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed_url.scheme == "http" and not is_loopback:
            raise ConfigurationError("远程 API 必须使用 HTTPS；HTTP 只允许本机地址")
        if hostname == "api.deepseek.com" and not api_key:
            raise ConfigurationError(
                "缺少 DeepSeek API Key：请在配置文件中设置 DEEPSEEK_API_KEY"
            )

        return cls(
            api_key=api_key,
            base_url=resolved_base,
            model=resolved_model,
            timeout=_positive_float(resolved_timeout, 120.0, "timeout"),
            max_steps=_positive_int(resolved_steps, 30, "max_steps"),
            max_context_chars=context_chars,
            max_retries=_nonnegative_int(
                _first_env("CODING_AGENT_MAX_RETRIES"), 2, "max_retries"
            ),
            extra_headers=extra_headers,
        )
