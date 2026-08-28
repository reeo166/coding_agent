"""Command-line entry point for the local coding agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from coding_agent.agent import AgentError, AgentLimitError, CodingAgent
from coding_agent.api import APIError, ChatCompletionClient
from coding_agent.config import ConfigurationError, Settings, load_env_file
from coding_agent.tools import ApprovalPolicy, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A small local coding agent using DeepSeek's tool-calling API."
    )
    parser.add_argument("task", nargs="*", help="Task for the agent. Omit for interactive mode.")
    parser.add_argument("--workspace", default=".", help="Workspace the agent may access.")
    parser.add_argument(
        "--config",
        help="Explicit dotenv config file. It is never auto-loaded from an untrusted workspace.",
    )
    parser.add_argument(
        "--base-url", help="API base URL (DeepSeek: https://api.deepseek.com)."
    )
    parser.add_argument("--model", help="Tool-calling model name.")
    parser.add_argument("--max-steps", type=int, help="Maximum model/tool loop iterations.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        help="Approximate context budget before older messages are compacted.",
    )
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Approve file writes and shell commands automatically (trusted tasks only).",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration without calling the model.",
    )
    parser.add_argument("--version", action="version", version="coding-agent 0.1.0")
    return parser


def _preview_arguments(arguments: dict[str, Any]) -> str:
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"content", "old_text", "new_text"} and isinstance(value, str):
            safe[key] = f"<{len(value)} chars>"
        elif isinstance(value, str) and len(value) > 160:
            safe[key] = value[:157] + "..."
        else:
            safe[key] = value
    return json.dumps(safe, ensure_ascii=False)


def console_event(event: str, data: dict[str, Any]) -> None:
    if event == "step":
        print(f"\n[步骤 {data['step']}/{data['max_steps']}] 正在请求模型...", flush=True)
    elif event == "tool_start":
        print(
            f"  -> {data['name']} {_preview_arguments(data.get('arguments', {}))}",
            flush=True,
        )
    elif event == "tool_end":
        status = "完成" if data.get("ok") else "失败"
        print(f"  <- {data['name']}: {status}", flush=True)
    elif event == "context_compacted":
        print("  [上下文] 已压缩较早的工具输出。", flush=True)


def create_agent(
    args: argparse.Namespace,
    workspace: Path,
    settings: Settings,
    config_path: Path | None = None,
) -> CodingAgent:
    approval = ApprovalPolicy(auto_approve=args.yes)
    protected_paths = [config_path] if config_path is not None else []
    registry = ToolRegistry(
        workspace=workspace, approval=approval, protected_paths=protected_paths
    )
    client = ChatCompletionClient(settings)
    return CodingAgent(
        client=client,
        tools=registry,
        workspace=workspace,
        max_steps=settings.max_steps,
        max_context_chars=settings.max_context_chars,
        event_sink=console_event,
    )


def print_config(settings: Settings, workspace: Path, auto_approve: bool) -> None:
    print("配置有效：")
    print(f"  workspace : {workspace}")
    print(f"  endpoint  : {settings.display_endpoint}")
    print(f"  model     : {settings.model}")
    print(f"  api key   : {'已设置' if settings.api_key else '未设置（仅适合无需鉴权的本地服务）'}")
    if settings.extra_headers:
        print(f"  headers   : 已设置 {len(settings.extra_headers)} 个自定义请求头（值已隐藏）")
    print(f"  approval  : {'自动批准' if auto_approve else '写文件/命令执行前询问'}")


def run_single(agent: CodingAgent, task: str) -> int:
    answer = agent.run(task)
    print(f"\n智能体：\n{answer}")
    return 0


def run_interactive(agent: CodingAgent) -> int:
    print("进入交互模式。输入 :clear 清空对话，:quit 退出。")
    while True:
        try:
            task = input("\n你：").strip()
        except EOFError:
            print()
            return 0
        if not task:
            continue
        if task in {":quit", ":q", "quit", "exit"}:
            return 0
        if task == ":clear":
            agent.reset()
            print("对话已清空。")
            continue
        try:
            run_single(agent, task)
        except (APIError, AgentError) as exc:
            print(f"\n错误：{exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"workspace 不存在或不是目录：{workspace}")

    config_path: Path | None = None
    try:
        if args.config:
            config_path = Path(args.config).expanduser()
            if not config_path.is_absolute():
                config_path = workspace / config_path
            config_path = config_path.resolve()
            if not config_path.is_file():
                raise ConfigurationError(f"配置文件不存在：{config_path}")
            load_env_file(config_path)
        settings = Settings.from_sources(
            base_url=args.base_url,
            model=args.model,
            max_steps=args.max_steps,
            max_context_chars=args.max_context_chars,
            timeout=args.timeout,
        )
    except ConfigurationError as exc:
        parser.error(str(exc))

    if args.check_config:
        print_config(settings, workspace, args.yes)
        return 0

    print_config(settings, workspace, args.yes)
    agent = create_agent(args, workspace, settings, config_path)
    task = " ".join(args.task).strip()
    return run_single(agent, task) if task else run_interactive(agent)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中止。", file=sys.stderr)
        raise SystemExit(130)
    except AgentLimitError as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(4)
    except AgentError as exc:
        print(f"\n智能体错误：{exc}", file=sys.stderr)
        raise SystemExit(4)
    except APIError as exc:
        print(f"\nAPI 错误：{exc}", file=sys.stderr)
        raise SystemExit(3)
