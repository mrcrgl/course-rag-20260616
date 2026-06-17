from __future__ import annotations

import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_course.commands.chat import run_chat
from rag_course.config import AppConfig


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
            ),
        ]
        return iter(chunks)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


class ChatTests(unittest.TestCase):
    def test_run_chat_streams_response_and_writes_audit_log(self) -> None:
        fake_client = _FakeOpenAIClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_dir = project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "system_prompt.md").write_text(
                "You are a conversational assistant.\n",
                encoding="utf-8",
            )

            with patch("rag_course.commands.chat._project_root", return_value=project_root):
                with patch("rag_course.commands.chat.build_client", return_value=fake_client):
                    with patch("builtins.input", side_effect=["Hello there", "exit"]):
                        stdout = io.StringIO()
                        with patch("sys.stdout", stdout):
                            with self.assertLogs("rag_course.commands.chat", level="DEBUG") as logs:
                                turns = run_chat(
                                    AppConfig(
                                        openai_api_key="sk-test",
                                        chat_model="gpt-4.1-mini",
                                    )
                                )

            self.assertEqual(turns, 1)
            self.assertIn("Assistant: Hello there", stdout.getvalue())
            self.assertTrue(any("prompt_tokens=12" in line for line in logs.output))
            self.assertTrue(any("completion_tokens=3" in line for line in logs.output))

            audit_files = sorted((project_root / "auditlog").glob("*.md"))
            self.assertEqual(len(audit_files), 1)
            audit_text = audit_files[0].read_text(encoding="utf-8")
            self.assertIn("Hello there", audit_text)
            self.assertIn("prompt_tokens: 12", audit_text)
            self.assertIn("You are a conversational assistant.", audit_text)

            create_kwargs = fake_client.chat.completions.calls[0]
            self.assertEqual(create_kwargs["model"], "gpt-4.1-mini")
            self.assertTrue(create_kwargs["stream"])
            self.assertEqual(create_kwargs["stream_options"], {"include_usage": True})
            self.assertEqual([message["role"] for message in create_kwargs["messages"]], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
