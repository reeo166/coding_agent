from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.tools import ApprovalPolicy, ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = ToolRegistry(
            self.root, approval=ApprovalPolicy(auto_approve=True), output_limit=20_000
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, name: str, **arguments: object) -> dict[str, object]:
        return json.loads(self.registry.execute(name, arguments))

    def test_write_read_search_and_replace(self) -> None:
        written = self.execute(
            "write_file", path="src/demo.py", content="value = 1\n", overwrite=False
        )
        self.assertTrue(written["ok"])

        read = self.execute("read_file", path="src/demo.py")
        self.assertIn("value = 1", read["result"]["content"])  # type: ignore[index]

        found = self.execute("search_text", query="VALUE", pattern="*.py")
        self.assertEqual(found["result"]["count"], 1)  # type: ignore[index]

        replaced = self.execute(
            "replace_text", path="src/demo.py", old_text="1", new_text="2"
        )
        self.assertTrue(replaced["ok"])
        self.assertEqual((self.root / "src/demo.py").read_text(encoding="utf-8"), "value = 2\n")

    def test_path_escape_and_secrets_are_rejected(self) -> None:
        escaped = self.execute("read_file", path="../outside.txt")
        self.assertFalse(escaped["ok"])
        secret = self.execute("write_file", path=".env", content="KEY=x")
        self.assertFalse(secret["ok"])
        (self.root / ".npmrc").write_text("//registry/:_authToken=secret", encoding="utf-8")
        npm_secret = self.execute("read_file", path=".npmrc")
        self.assertFalse(npm_secret["ok"])
        listed = self.execute("list_files")
        listed_paths = [item["path"] for item in listed["result"]["files"]]  # type: ignore[index]
        self.assertNotIn(".npmrc", listed_paths)

    def test_explicit_protected_config_path_is_hidden(self) -> None:
        config_path = self.root / "agent.env"
        config_path.write_text("CODING_AGENT_API_KEY=secret", encoding="utf-8")
        registry = ToolRegistry(
            self.root,
            approval=ApprovalPolicy(auto_approve=True),
            protected_paths=[config_path],
        )
        read = json.loads(registry.execute("read_file", {"path": "agent.env"}))
        listed = json.loads(registry.execute("list_files", {}))
        self.assertFalse(read["ok"])
        self.assertNotIn(
            "agent.env", [item["path"] for item in listed["result"]["files"]]
        )

    def test_existing_file_requires_explicit_overwrite(self) -> None:
        (self.root / "a.txt").write_text("old", encoding="utf-8")
        result = self.execute("write_file", path="a.txt", content="new")
        self.assertFalse(result["ok"])
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "old")

    def test_noninteractive_mutation_is_denied_without_yes(self) -> None:
        registry = ToolRegistry(
            self.root,
            approval=ApprovalPolicy(auto_approve=False, interactive=False),
        )
        result = json.loads(
            registry.execute("write_file", {"path": "denied.txt", "content": "x"})
        )
        self.assertFalse(result["ok"])
        self.assertTrue(result["denied"])
        self.assertFalse((self.root / "denied.txt").exists())

    @patch.dict(
        os.environ,
        {"CODING_AGENT_API_KEY": "do-not-leak", "GITHUB_TOKEN": "do-not-leak"},
        clear=False,
    )
    def test_commands_do_not_inherit_secret_environment_values(self) -> None:
        code = (
            "import os; "
            "print(os.getenv('CODING_AGENT_API_KEY')); "
            "print(os.getenv('GITHUB_TOKEN'))"
        )
        command = subprocess.list2cmdline([sys.executable, "-c", code])
        result = self.execute("run_command", command=command, timeout=20)
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"].splitlines(), ["None", "None"])  # type: ignore[index]

    def test_commands_reject_common_secret_and_parent_path_access(self) -> None:
        secret = self.execute("run_command", command="cmd /c type .env")
        escaped = self.execute("run_command", command="cmd /c type ..\\outside.txt")
        chained = self.execute("run_command", command=r"cmd /c echo first \& echo second")
        self.assertFalse(secret["ok"])
        self.assertFalse(escaped["ok"])
        self.assertFalse(chained["ok"])

    def test_approval_preview_does_not_hide_command_tail(self) -> None:
        command = "safe-prefix " + ("x" * 500) + " destructive-tail"
        preview = ApprovalPolicy._preview({"command": command})
        self.assertIn("safe-prefix", preview)
        self.assertIn("destructive-tail", preview)

    def test_command_timeout_terminates_promptly(self) -> None:
        command = subprocess.list2cmdline(
            [sys.executable, "-c", "import time; time.sleep(10)"]
        )
        started = time.monotonic()
        result = self.execute("run_command", command=command, timeout=1)
        elapsed = time.monotonic() - started
        self.assertTrue(result["ok"])
        self.assertTrue(result["result"]["timed_out"])  # type: ignore[index]
        self.assertLess(elapsed, 5)

    def test_command_output_is_bounded_during_execution(self) -> None:
        command = subprocess.list2cmdline(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 5000000)"]
        )
        result = self.execute("run_command", command=command, timeout=20)
        self.assertTrue(result["ok"])
        self.assertTrue(result["result"]["output_limit_exceeded"])  # type: ignore[index]
        self.assertLess(len(result["result"]["stdout"]), 20_000)  # type: ignore[index]

    @unittest.skipUnless(os.name == "nt", "Windows batch behavior")
    def test_windows_batch_entrypoint_runs_without_manual_cmd_wrapper(self) -> None:
        batch = self.root / "hello.cmd"
        batch.write_text("@echo batch-ok\n", encoding="utf-8")
        result = self.execute("run_command", command=r".\hello.cmd", timeout=20)
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["exit_code"], 0)  # type: ignore[index]
        self.assertIn("batch-ok", result["result"]["stdout"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
