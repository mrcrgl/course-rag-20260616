from __future__ import annotations

import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_course.commands.chat import run_chat
from rag_course.commands.query import RagResult
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
                    with patch(
                        "rag_course.commands.chat.retrieve_rag_results",
                        return_value=[
                            RagResult(
                                point_id="00000000-0000-0000-0000-000000000001",
                                score=0.91,
                                text="Retrieved chunk text.",
                                metadata={
                                    "canonical_url": "https://example.com/story.txt",
                                    "page_number": 15,
                                    "headline_1": "Story",
                                },
                            )
                        ],
                    ) as retrieve_rag_results:
                        with patch("builtins.input", side_effect=["Hello there", "exit"]):
                            stdout = io.StringIO()
                            with patch("sys.stdout", stdout):
                                with self.assertLogs("rag_course.commands.chat", level="DEBUG") as logs:
                                    turns = run_chat(
                                        AppConfig(
                                            openai_api_key="sk-test",
                                            chat_model="gpt-4.1-mini",
                                            rag_score_threshold=0.5,
                                            rag_top_k=5,
                                            rag_context_token_budget_total=800,
                                            rag_context_token_budget_per_entry=200,
                                        )
                                    )

            self.assertEqual(turns, 1)
            self.assertIn("Assistant: Hello there", stdout.getvalue())
            self.assertTrue(any("prompt_tokens=12" in line for line in logs.output))
            self.assertTrue(any("completion_tokens=3" in line for line in logs.output))
            retrieve_rag_results.assert_called_once_with(
                unittest.mock.ANY,
                "Hello there",
                top_k=5,
                score_threshold=0.5,
            )

            audit_files = sorted((project_root / "auditlog").glob("*.md"))
            self.assertEqual(len(audit_files), 1)
            audit_text = audit_files[0].read_text(encoding="utf-8")
            self.assertIn("Hello there", audit_text)
            self.assertIn("prompt_tokens: 12", audit_text)
            self.assertIn("You are a conversational assistant.", audit_text)
            self.assertIn("Retrieved chunk text.", audit_text)
            self.assertIn("meta_page_number: 15", audit_text)
            self.assertIn("meta_headline_1: Story", audit_text)
            self.assertIn("Response Footer:", audit_text)
            self.assertIn("https://example.com/story.txt#page=15", audit_text)

            create_kwargs = fake_client.chat.completions.calls[0]
            self.assertEqual(create_kwargs["model"], "gpt-4.1-mini")
            self.assertTrue(create_kwargs["stream"])
            self.assertEqual(create_kwargs["stream_options"], {"include_usage": True})
            self.assertEqual([message["role"] for message in create_kwargs["messages"]], ["system", "user"])
            self.assertIn("Retrieved context:", create_kwargs["messages"][0]["content"])
            self.assertIn("Retrieved chunk text.", create_kwargs["messages"][0]["content"])
            self.assertIn("meta_page_number: 15", create_kwargs["messages"][0]["content"])

            self.assertIn("Assistant: Hello there", stdout.getvalue())
            self.assertIn("Response Footer:", stdout.getvalue())
            self.assertIn("https://example.com/story.txt#page=15", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
