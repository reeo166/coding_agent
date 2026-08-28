from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from coding_agent.agent import AgentLimitError, CodingAgent, ContextManager
from coding_agent.tools import ApprovalPolicy, ToolRegistry


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.requests.append(messages)
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


class CodingAgentTests(unittest.TestCase):
    def test_tool_loop_then_final_answer(self) -> None:
        responses = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "任务完成。"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = ToolRegistry(root, approval=ApprovalPolicy(auto_approve=True))
            client = FakeClient(responses)
            agent = CodingAgent(client=client, tools=tools, workspace=root, max_steps=3)
            answer = agent.run("检查目录")

        self.assertEqual(answer, "任务完成。")
        second_request = client.requests[1]
        self.assertEqual(second_request[-2]["role"], "assistant")
        self.assertEqual(second_request[-1]["role"], "tool")
        self.assertEqual(second_request[-1]["tool_call_id"], "call_1")

    def test_malformed_tool_arguments_are_returned_to_model(self) -> None:
        responses = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bad",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{"},
                    }
                ],
            },
            {"role": "assistant", "content": "已处理参数错误。"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeClient(responses)
            agent = CodingAgent(
                client=client,
                tools=ToolRegistry(root, approval=ApprovalPolicy(auto_approve=True)),
                workspace=root,
            )
            agent.run("测试错误")
        self.assertIn("不是有效 JSON", client.requests[1][-1]["content"])

    def test_multiple_tool_calls_receive_matching_replies(self) -> None:
        responses = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "first",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    },
                    {
                        "id": "second",
                        "type": "function",
                        "function": {"name": "search_text", "arguments": '{"query":"x"}'},
                    },
                ],
            },
            {"role": "assistant", "content": "完成。"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeClient(responses)
            agent = CodingAgent(
                client=client,
                tools=ToolRegistry(root, approval=ApprovalPolicy(auto_approve=True)),
                workspace=root,
            )
            agent.run("并行检查")
        replies = [msg for msg in client.requests[1] if msg.get("role") == "tool"]
        self.assertEqual([msg["tool_call_id"] for msg in replies], ["first", "second"])

    def test_context_compaction_keeps_tool_call_with_reply(self) -> None:
        manager = ContextManager(max_chars=4_000)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "do work"},
        ]
        for number in range(8):
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"c{number}",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"c{number}",
                        "name": "read_file",
                        "content": "x" * 2_000,
                    },
                ]
            )
        prepared, compacted = manager.prepare(messages)
        self.assertTrue(compacted)
        for index, message in enumerate(prepared):
            if message.get("role") == "tool":
                self.assertGreater(index, 0)
                self.assertEqual(prepared[index - 1].get("role"), "assistant")
                self.assertTrue(prepared[index - 1].get("tool_calls"))

    def test_repeated_identical_tool_batches_stop_the_loop(self) -> None:
        responses = []
        for number in range(3):
            responses.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"repeat-{number}",
                            "type": "function",
                            "function": {"name": "list_files", "arguments": "{}"},
                        }
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = CodingAgent(
                client=FakeClient(responses),
                tools=ToolRegistry(root, approval=ApprovalPolicy(auto_approve=True)),
                workspace=root,
                max_steps=10,
            )
            with self.assertRaises(AgentLimitError):
                agent.run("不要无限循环")

    def test_context_does_not_keep_an_old_answer_without_its_question(self) -> None:
        manager = ContextManager(max_chars=4_000)
        prepared, compacted = manager.prepare(
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "x" * 5_000},
                {"role": "assistant", "content": "orphan answer"},
                {"role": "user", "content": "current task"},
            ]
        )
        self.assertTrue(compacted)
        self.assertNotIn("orphan answer", [msg.get("content") for msg in prepared])
        self.assertEqual(prepared[-1], {"role": "user", "content": "current task"})


if __name__ == "__main__":
    unittest.main()
