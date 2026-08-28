"""Local file and command tools exposed to the model."""

from __future__ import annotations

import fnmatch
import json
import locale
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


class ToolFailure(RuntimeError):
    """A recoverable tool error that should be shown to the model."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    mutating: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ApprovalPolicy:
    """Require a human decision before state-changing local operations."""

    def __init__(
        self,
        auto_approve: bool = False,
        *,
        interactive: bool | None = None,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.auto_approve = auto_approve
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.input_fn = input_fn

    def approve(self, spec: ToolSpec, arguments: dict[str, Any]) -> tuple[bool, str]:
        if not spec.mutating or self.auto_approve:
            return True, ""
        if not self.interactive:
            return False, "非交互模式不会自动执行写文件或命令；确认任务可信后加 --yes"
        preview = self._preview(arguments)
        answer = self.input_fn(f"\n允许工具 {spec.name} 执行 {preview}？[y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return True, ""
        return False, "用户拒绝了本次本地变更"

    @staticmethod
    def _preview(arguments: dict[str, Any]) -> str:
        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in {"content", "old_text", "new_text"} and isinstance(value, str):
                safe[key] = f"<{len(value)} chars>"
            elif key == "command" and isinstance(value, str):
                safe[key] = value
            elif isinstance(value, str) and len(value) > 200:
                safe[key] = value[:197] + "..."
            else:
                safe[key] = value
        return json.dumps(safe, ensure_ascii=False)


class ToolRegistry:
    """Workspace-scoped tools plus their JSON schemas."""

    EXCLUDED_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        ".codex_deps",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
    }
    SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
    SENSITIVE_NAMES = {
        ".env",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "_netrc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "service-account.json",
    }
    SENSITIVE_DIRS = {".aws", ".docker", ".gnupg", ".kube", ".ssh"}
    MAX_COMMAND_OUTPUT_BYTES = 2_000_000
    SENSITIVE_ENV_MARKERS = (
        "API_KEY",
        "APIKEY",
        "ACCESS_KEY",
        "ACCESS_TOKEN",
        "AUTH_TOKEN",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "PRIVATE_KEY",
        "CLIENT_SECRET",
        "CREDENTIAL",
    )

    def __init__(
        self,
        workspace: Path,
        approval: ApprovalPolicy | None = None,
        *,
        output_limit: int = 30_000,
        protected_paths: Iterable[Path] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        self.approval = approval or ApprovalPolicy()
        self.output_limit = output_limit
        self.protected_paths = {
            path.expanduser().resolve(strict=False) for path in (protected_paths or [])
        }
        self.protected_file_ids: set[tuple[int, int]] = set()
        for protected in self.protected_paths:
            try:
                info = protected.stat()
                self.protected_file_ids.add((info.st_dev, info.st_ino))
            except OSError:
                continue
        self._specs = {spec.name: spec for spec in self._build_specs()}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self._specs.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        spec = self._specs.get(name)
        if spec is None:
            return self._encode(False, error=f"未知工具：{name}")
        if not isinstance(arguments, dict):
            return self._encode(False, error="工具参数必须是 JSON 对象")
        approved, reason = self.approval.approve(spec, arguments)
        if not approved:
            return self._encode(False, error=reason, denied=True)
        try:
            result = spec.handler(**arguments)
            return self._encode(True, result=result)
        except ToolFailure as exc:
            return self._encode(False, error=str(exc))
        except TypeError as exc:
            return self._encode(False, error=f"工具参数不匹配：{exc}")
        except Exception as exc:  # Keep the agent loop alive on unexpected local failures.
            return self._encode(False, error=f"工具内部错误：{type(exc).__name__}: {exc}")

    def _encode(self, ok: bool, **payload: Any) -> str:
        value = {"ok": ok, **payload}
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded) <= self.output_limit:
            return encoded
        compact = {
            "ok": ok,
            "truncated": True,
            "output": self._truncate(encoded, self.output_limit - 120),
        }
        return json.dumps(compact, ensure_ascii=False)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = max(1, limit * 2 // 3)
        tail = max(1, limit - head - 80)
        removed = len(text) - head - tail
        return f"{text[:head]}\n... <truncated {removed} chars> ...\n{text[-tail:]}"

    def _build_specs(self) -> list[ToolSpec]:
        obj = {"type": "object", "additionalProperties": False}
        return [
            ToolSpec(
                "list_files",
                "List non-secret files inside the workspace. Results are relative paths.",
                {
                    **obj,
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory", "default": "."},
                        "pattern": {"type": "string", "description": "Glob such as *.py", "default": "*"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 300},
                    },
                },
                self.list_files,
            ),
            ToolSpec(
                "read_file",
                "Read a UTF-8 text file with line numbers. Use line ranges for large files.",
                {
                    **obj,
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1, "default": 1},
                        "end_line": {"type": "integer", "minimum": 1, "description": "Inclusive; at most 1000 lines per call"},
                    },
                    "required": ["path"],
                },
                self.read_file,
            ),
            ToolSpec(
                "search_text",
                "Search literal text in UTF-8 workspace files and return path:line matches.",
                {
                    **obj,
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "pattern": {"type": "string", "description": "File glob", "default": "*"},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    },
                    "required": ["query"],
                },
                self.search_text,
            ),
            ToolSpec(
                "write_file",
                "Create or fully rewrite a UTF-8 file inside the workspace.",
                {
                    **obj,
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "overwrite": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "content"],
                },
                self.write_file,
                mutating=True,
            ),
            ToolSpec(
                "replace_text",
                "Replace exact text in an existing UTF-8 file. By default the old text must occur exactly once.",
                {
                    **obj,
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
                self.replace_text,
                mutating=True,
            ),
            ToolSpec(
                "run_command",
                "Run one local program from the workspace. Shell operators are not interpreted; use separate calls for separate commands.",
                {
                    **obj,
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 300, "default": 120},
                    },
                    "required": ["command"],
                },
                self.run_command,
                mutating=True,
            ),
        ]

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    @classmethod
    def _is_sensitive(cls, relative: Path) -> bool:
        parts = {part.lower() for part in relative.parts}
        name = relative.name.lower()
        if ".git" in parts or parts.intersection(cls.SENSITIVE_DIRS):
            return True
        if name in cls.SENSITIVE_NAMES:
            return True
        if name.startswith(".env.") and name != ".env.example":
            return True
        return relative.suffix.lower() in cls.SENSITIVE_SUFFIXES

    def _resolve(self, raw_path: str, *, expect_directory: bool | None = None) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise ToolFailure("路径不能为空或包含 NUL")
        entered = Path(raw_path)
        candidate = entered if entered.is_absolute() else self.workspace / entered
        try:
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(self.workspace)
        except (OSError, ValueError) as exc:
            raise ToolFailure("路径越过了 workspace 边界") from exc
        if self._is_sensitive(relative):
            raise ToolFailure("出于密钥保护，智能体不能访问该敏感文件")
        if self._is_protected_path(resolved):
            raise ToolFailure("智能体不能访问当前运行所使用的配置文件")
        if expect_directory is True and (not resolved.exists() or not resolved.is_dir()):
            raise ToolFailure(f"目录不存在：{raw_path}")
        if expect_directory is False and (not resolved.exists() or not resolved.is_file()):
            raise ToolFailure(f"文件不存在：{raw_path}")
        return resolved

    def _iter_files(self, directory: Path, pattern: str) -> Iterable[Path]:
        for current, dirnames, filenames in os.walk(directory, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames if name not in self.EXCLUDED_DIRS
            )
            current_path = Path(current)
            for filename in sorted(filenames):
                path = current_path / filename
                try:
                    relative = path.resolve(strict=False).relative_to(self.workspace)
                except (OSError, ValueError):
                    continue
                if self._is_sensitive(relative):
                    continue
                if self._is_protected_path(path.resolve(strict=False)):
                    continue
                relative_text = relative.as_posix()
                if fnmatch.fnmatch(relative_text, pattern) or fnmatch.fnmatch(filename, pattern):
                    yield path

    def _is_protected_path(self, path: Path) -> bool:
        if path in self.protected_paths:
            return True
        try:
            info = path.stat()
        except OSError:
            return False
        return (info.st_dev, info.st_ino) in self.protected_file_ids

    @staticmethod
    def _bounded_int(value: int, *, minimum: int, maximum: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolFailure(f"{label} 必须是整数")
        if not minimum <= value <= maximum:
            raise ToolFailure(f"{label} 必须在 {minimum} 到 {maximum} 之间")
        return value

    @staticmethod
    def _read_utf8(path: Path, *, max_bytes: int = 5_000_000) -> str:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ToolFailure(f"无法读取文件信息：{exc}") from exc
        if size > max_bytes:
            raise ToolFailure(f"文件过大（{size} bytes），单次最多读取 {max_bytes} bytes")
        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            raise ToolFailure("该文件像是二进制文件，不能作为文本读取")
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolFailure("文件不是有效 UTF-8；请先用本地命令确认编码") from exc

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def list_files(
        self, path: str = ".", pattern: str = "*", max_results: int = 300
    ) -> dict[str, Any]:
        directory = self._resolve(path, expect_directory=True)
        limit = self._bounded_int(max_results, minimum=1, maximum=1000, label="max_results")
        files: list[dict[str, Any]] = []
        truncated = False
        for file_path in self._iter_files(directory, pattern):
            if len(files) >= limit:
                truncated = True
                break
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            files.append({"path": self._relative(file_path), "bytes": size})
        return {"files": files, "count": len(files), "truncated": truncated}

    def read_file(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> dict[str, Any]:
        file_path = self._resolve(path, expect_directory=False)
        start = self._bounded_int(start_line, minimum=1, maximum=10_000_000, label="start_line")
        if end_line is None:
            end = start + 399
        else:
            end = self._bounded_int(end_line, minimum=1, maximum=10_000_000, label="end_line")
        if end < start:
            raise ToolFailure("end_line 不能小于 start_line")
        if end - start + 1 > 1000:
            raise ToolFailure("每次最多读取 1000 行")
        lines = self._read_utf8(file_path).splitlines()
        selected = lines[start - 1 : end]
        numbered = "\n".join(
            f"{line_number:>6}: {line}"
            for line_number, line in enumerate(selected, start=start)
        )
        actual_end = start + len(selected) - 1 if selected else min(start - 1, len(lines))
        return {
            "path": self._relative(file_path),
            "start_line": start,
            "end_line": actual_end,
            "total_lines": len(lines),
            "content": numbered,
        }

    def search_text(
        self,
        query: str,
        path: str = ".",
        pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query:
            raise ToolFailure("query 不能为空")
        directory = self._resolve(path, expect_directory=True)
        limit = self._bounded_int(max_results, minimum=1, maximum=500, label="max_results")
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        for file_path in self._iter_files(directory, pattern):
            try:
                text = self._read_utf8(file_path, max_bytes=2_000_000)
            except ToolFailure:
                continue
            scanned += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    if len(matches) >= limit:
                        truncated = True
                        break
                    matches.append(
                        {
                            "path": self._relative(file_path),
                            "line": line_number,
                            "text": self._truncate(line, 500),
                        }
                    )
            if truncated:
                break
        return {
            "matches": matches,
            "count": len(matches),
            "files_scanned": scanned,
            "truncated": truncated,
        }

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ToolFailure("content 必须是字符串")
        if not isinstance(overwrite, bool):
            raise ToolFailure("overwrite 必须是布尔值")
        if len(content.encode("utf-8")) > 2_000_000:
            raise ToolFailure("单次写入不能超过 2 MB")
        file_path = self._resolve(path)
        if file_path == self.workspace or file_path.is_dir():
            raise ToolFailure("目标必须是文件路径")
        existed = file_path.exists()
        if existed and not overwrite:
            raise ToolFailure("文件已存在；确认完整覆盖后请传 overwrite=true")
        self._atomic_write(file_path, content)
        return {
            "path": self._relative(file_path),
            "created": not existed,
            "bytes": len(content.encode("utf-8")),
        }

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(old_text, str) or not old_text:
            raise ToolFailure("old_text 不能为空")
        if not isinstance(new_text, str):
            raise ToolFailure("new_text 必须是字符串")
        if not isinstance(replace_all, bool):
            raise ToolFailure("replace_all 必须是布尔值")
        file_path = self._resolve(path, expect_directory=False)
        original = self._read_utf8(file_path)
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise ToolFailure("没有找到 old_text；请重新读取文件并提供精确文本")
        if occurrences > 1 and not replace_all:
            raise ToolFailure(f"old_text 出现 {occurrences} 次；请扩大上下文或传 replace_all=true")
        updated = original.replace(old_text, new_text, -1 if replace_all else 1)
        if len(updated.encode("utf-8")) > 2_000_000:
            raise ToolFailure("修改后的文件超过 2 MB")
        self._atomic_write(file_path, updated)
        return {
            "path": self._relative(file_path),
            "replacements": occurrences if replace_all else 1,
            "bytes": len(updated.encode("utf-8")),
        }

    def run_command(self, command: str, timeout: int = 120) -> dict[str, Any]:
        if not isinstance(command, str) or not command.strip():
            raise ToolFailure("command 不能为空")
        if len(command) > 10_000:
            raise ToolFailure("command 过长，最多 10000 个字符")
        self._guard_command(command)
        timeout_value = self._bounded_int(timeout, minimum=1, maximum=300, label="timeout")
        sanitized_env = {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in self.SENSITIVE_ENV_MARKERS)
            and key.upper() != "CODING_AGENT_EXTRA_HEADERS"
        }
        process_options: dict[str, Any] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        process_arguments: str | list[str]
        if os.name == "nt":
            process_arguments = self._windows_command(command, sanitized_env)
        else:
            try:
                process_arguments = shlex.split(command)
            except ValueError as exc:
                raise ToolFailure(f"command 引号不匹配：{exc}") from exc
            if not process_arguments:
                raise ToolFailure("command 不能为空")
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                process_arguments,
                cwd=self.workspace,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                env=sanitized_env,
                **process_options,
            )
            deadline = time.monotonic() + timeout_value
            timed_out = False
            output_limit_exceeded = False
            while process.poll() is None:
                captured_bytes = os.fstat(stdout_file.fileno()).st_size + os.fstat(
                    stderr_file.fileno()
                ).st_size
                if captured_bytes > self.MAX_COMMAND_OUTPUT_BYTES:
                    output_limit_exceeded = True
                    self._terminate_process_tree(process)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    self._terminate_process_tree(process)
                    break
                time.sleep(min(0.05, remaining))

            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass

            stdout_size = os.fstat(stdout_file.fileno()).st_size
            stderr_size = os.fstat(stderr_file.fileno()).st_size
            if stdout_size + stderr_size > self.MAX_COMMAND_OUTPUT_BYTES:
                output_limit_exceeded = True
            stdout = self._read_capture(stdout_file, 14_000)
            stderr = self._read_capture(stderr_file, 14_000)
            return {
                "exit_code": process.returncode,
                "timed_out": timed_out,
                "output_limit_exceeded": output_limit_exceeded,
                "stdout": stdout,
                "stderr": stderr,
            }

    def _windows_command(self, command: str, environment: dict[str, str]) -> str | list[str]:
        match = re.match(r'^\s*(?:"([^"]+)"|(\S+))', command)
        executable = (match.group(1) or match.group(2)) if match else ""
        resolved: str | None = None
        if executable:
            entered = Path(executable)
            if entered.is_absolute() or "\\" in executable or "/" in executable:
                candidate = entered if entered.is_absolute() else self.workspace / entered
                if candidate.is_file():
                    resolved = str(candidate.resolve())
            else:
                resolved = shutil.which(executable, path=environment.get("PATH"))
        if resolved and Path(resolved).suffix.casefold() in {".bat", ".cmd"}:
            command_shell = environment.get("COMSPEC") or os.environ.get("COMSPEC") or "cmd.exe"
            return [command_shell, "/d", "/s", "/c", command]
        return command

    @staticmethod
    def _guard_command(command: str) -> None:
        """Block common secret/path escapes even when --yes is enabled.

        This is defense in depth, not an OS sandbox. Arbitrary local programs are
        powerful, so the default human approval remains the primary boundary.
        """

        lowered = command.casefold().replace(".env.example", "")
        if re.search(r"(^|[\s\\/\"'=])\.env(?:[.\w-]*)(?=$|[\s\\/\"'])", lowered):
            raise ToolFailure("命令不能读取或操作 .env 凭据文件")
        if re.search(r"(^|[\s\\/\"'=])\.\.(?:[\\/]|$)", command):
            raise ToolFailure("命令不能使用 .. 越过 workspace")
        if re.search(r"(?i)(?:^|[\s\"'])[^\s\"']+\.(?:key|pem|p12|pfx)(?:$|[\s\"'])", command):
            raise ToolFailure("命令不能读取或操作私钥/证书凭据文件")
        if re.search(
            r"(?i)(?:^|[\s\\/\"'])(?:\.npmrc|\.pypirc|\.netrc|_netrc|\.git-credentials|credentials\.json)(?:$|[\s\\/\"'])",
            command,
        ):
            raise ToolFailure("命令不能读取或操作凭据文件")
        if re.search(r"(?i)(?:^|[\s\\/\"'])(?:\.aws|\.docker|\.kube|\.ssh)(?:[\\/]|$)", command):
            raise ToolFailure("命令不能读取或操作凭据目录")
        if ToolRegistry._has_unquoted_shell_operator(command):
            raise ToolFailure("每次只能运行一个程序，不能使用 shell 管道、重定向或命令串联")

    @staticmethod
    def _has_unquoted_shell_operator(command: str) -> bool:
        in_double_quotes = False
        escaped_quote = False
        for index, character in enumerate(command):
            if escaped_quote:
                escaped_quote = False
                continue
            if (
                character == "\\"
                and in_double_quotes
                and index + 1 < len(command)
                and command[index + 1] == '"'
            ):
                escaped_quote = True
                continue
            if character == '"':
                in_double_quotes = not in_double_quotes
                continue
            # Single quotes do not quote metacharacters for cmd.exe, so they
            # intentionally do not affect this conservative check.
            if not in_double_quotes and character in {"&", "|", "<", ">"}:
                return True
        return False

    @classmethod
    def _read_capture(cls, stream: Any, limit: int) -> str:
        stream.flush()
        size = os.fstat(stream.fileno()).st_size
        stream.seek(0)
        if size <= limit:
            return cls._decode_output(stream.read())
        head_size = limit * 2 // 3
        tail_size = limit - head_size
        head = stream.read(head_size)
        stream.seek(-tail_size, os.SEEK_END)
        tail = stream.read(tail_size)
        removed = size - head_size - tail_size
        return (
            cls._decode_output(head)
            + f"\n... <truncated {removed} bytes> ...\n"
            + cls._decode_output(tail)
        )

    @staticmethod
    def _decode_output(data: bytes | None) -> str:
        if not data:
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode(locale.getpreferredencoding(False), errors="replace")

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
