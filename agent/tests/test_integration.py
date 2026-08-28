from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from coding_agent.agent import CodingAgent
from coding_agent.api import ChatCompletionClient
from coding_agent.config import Settings
from coding_agent.tools import ApprovalPolicy, ToolRegistry


class MockGatewayHandler(BaseHTTPRequestHandler):
    responses: ClassVar[list[dict[str, Any]]] = []
    requests: ClassVar[list[dict[str, Any]]] = []
    paths: ClassVar[list[str]] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(body)
        type(self).paths.append(self.path)
        if not type(self).responses:
            self.send_error(500, "No mock response left")
            return
        payload = json.dumps(type(self).responses.pop(0), ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *args: object) -> None:
        return


def model_message(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    call_id: str | None = None,
    name: str = "",
    arguments: str = "{}",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    finish_reason = "stop"
    if call_id:
        finish_reason = "tool_calls"
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


class LocalGatewayIntegrationTests(unittest.TestCase):
    def test_http_tool_loop_edits_and_verifies_project(self) -> None:
        command = (
            'python -c "from pathlib import Path; '
            "assert Path('bug.py').read_text(encoding='utf-8') == 'value = 2\\n'\""
        )
        MockGatewayHandler.requests = []
        MockGatewayHandler.paths = []
        MockGatewayHandler.responses = [
            model_message(
                reasoning_content="先读取目标文件。",
                call_id="read-1",
                name="read_file",
                arguments=json.dumps({"path": "bug.py"}),
            ),
            model_message(
                reasoning_content="根据文件内容执行精确替换。",
                call_id="edit-1",
                name="replace_text",
                arguments=json.dumps(
                    {"path": "bug.py", "old_text": "value = 1", "new_text": "value = 2"}
                ),
            ),
            model_message(
                reasoning_content="运行验证命令。",
                call_id="test-1",
                name="run_command",
                arguments=json.dumps({"command": command, "timeout": 20}),
            ),
            model_message(content="已修复并验证。", reasoning_content="验证已通过。"),
        ]

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockGatewayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "bug.py"
                target.write_text("value = 1\n", encoding="utf-8")
                settings = Settings(
                    api_key="test-only-key",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="mock-tool-model",
                    timeout=5,
                    max_retries=0,
                )
                agent = CodingAgent(
                    client=ChatCompletionClient(settings),
                    tools=ToolRegistry(root, approval=ApprovalPolicy(auto_approve=True)),
                    workspace=root,
                    max_steps=6,
                )
                answer = agent.run("修复 bug.py 并验证")

                self.assertEqual(answer, "已修复并验证。")
                self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(MockGatewayHandler.paths, ["/v1/chat/completions"] * 4)
        self.assertEqual(len(MockGatewayHandler.requests), 4)
        self.assertFalse(MockGatewayHandler.requests[0]["stream"])
        second_messages = MockGatewayHandler.requests[1]["messages"]
        first_assistant = next(
            message
            for message in second_messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        self.assertEqual(first_assistant["reasoning_content"], "先读取目标文件。")
        final_messages = MockGatewayHandler.requests[-1]["messages"]
        reasoning_history = [
            message.get("reasoning_content")
            for message in final_messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        self.assertEqual(
            reasoning_history,
            ["先读取目标文件。", "根据文件内容执行精确替换。", "运行验证命令。"],
        )
        self.assertEqual(final_messages[-1]["role"], "tool")
        self.assertEqual(final_messages[-1]["tool_call_id"], "test-1")


if __name__ == "__main__":
    unittest.main()
