from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.config import ConfigurationError, Settings, load_env_file


class ConfigurationTests(unittest.TestCase):
    def test_deepseek_defaults_and_specific_environment_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-secret",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com/beta",
                "DEEPSEEK_MODEL": "deepseek-v4-pro",
                "CODING_AGENT_API_KEY": "legacy-secret",
                "CODING_AGENT_BASE_URL": "https://legacy.example/v1",
                "CODING_AGENT_MODEL": "legacy-model",
            },
            clear=True,
        ):
            settings = Settings.from_sources()

        self.assertEqual(settings.api_key, "deepseek-secret")
        self.assertEqual(settings.base_url, "https://api.deepseek.com/beta")
        self.assertEqual(settings.model, "deepseek-v4-pro")

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "deepseek-secret"}, clear=True
        ):
            defaults = Settings.from_sources()
        self.assertEqual(defaults.base_url, "https://api.deepseek.com")
        self.assertEqual(defaults.endpoint, "https://api.deepseek.com/chat/completions")
        self.assertEqual(defaults.model, "deepseek-v4-flash")

    def test_deepseek_dotenv_variables_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "DEEPSEEK_API_KEY=file-key\n"
                "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
                "DEEPSEEK_MODEL=deepseek-v4-pro\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(path)
                settings = Settings.from_sources()
        self.assertEqual(settings.api_key, "file-key")
        self.assertEqual(settings.model, "deepseek-v4-pro")

    def test_official_deepseek_endpoint_requires_a_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "DEEPSEEK_API_KEY"):
                Settings.from_sources()

    def test_dotenv_does_not_override_shell_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'CODING_AGENT_MODEL="from-file"\nCODING_AGENT_TIMEOUT=45\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODING_AGENT_MODEL": "from-shell"}, clear=False):
                os.environ.pop("CODING_AGENT_TIMEOUT", None)
                load_env_file(path)
                self.assertEqual(os.environ["CODING_AGENT_MODEL"], "from-shell")
                self.assertEqual(os.environ["CODING_AGENT_TIMEOUT"], "45")
                os.environ.pop("CODING_AGENT_TIMEOUT", None)

    def test_explicit_zero_limits_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings.from_sources(
                base_url="http://localhost:1234/v1", model="mock", max_steps=0
            )
        with self.assertRaises(ConfigurationError):
            Settings.from_sources(
                base_url="http://localhost:1234/v1", model="mock", timeout=0
            )
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ConfigurationError):
                Settings.from_sources(
                    base_url="http://localhost:1234/v1", model="mock", timeout=invalid
                )

    def test_context_budget_has_a_practical_minimum(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings.from_sources(
                base_url="http://localhost:1234/v1",
                model="mock",
                max_context_chars=3_999,
            )

    def test_openai_key_is_not_implicitly_reused(self) -> None:
        with patch.dict(
            os.environ,
            {"CODING_AGENT_API_KEY": "", "OPENAI_API_KEY": "global-key"},
            clear=True,
        ):
            settings = Settings.from_sources(
                base_url="http://localhost:1234/v1", model="mock"
            )
        self.assertEqual(settings.api_key, "")

    def test_dotenv_rejects_unrelated_or_proxy_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("HTTPS_PROXY=http://attacker.invalid\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_env_file(path)

    def test_remote_http_with_credentials_is_rejected(self) -> None:
        with patch.dict(os.environ, {"CODING_AGENT_API_KEY": "secret"}, clear=False):
            with self.assertRaises(ConfigurationError):
                Settings.from_sources(base_url="http://example.test/v1", model="mock")


if __name__ == "__main__":
    unittest.main()
