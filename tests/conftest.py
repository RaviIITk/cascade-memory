from __future__ import annotations

import json
from typing import Any

import pytest

from cascade_memory.schema import Message


class FakeSummarizerClient:
    """Deterministic stand-in for a real LLM, used to test the summarizer contract."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        self.calls.append(messages)
        user_msg = next(m for m in messages if m["role"] == "user")
        span_len = user_msg["content"].count('"role"')
        return json.dumps(
            {
                "task": "Build the widget",
                "percent_complete": 42.0,
                "completed_steps": [f"processed span of {span_len} messages"],
                "remaining_steps": ["finish remaining work"],
                "key_decisions": ["used fake summarizer in tests"],
                "tool_call_summaries": [
                    {"purpose": "test tool call", "outcome": "succeeded", "memory_id_ref": None}
                ],
                "summary_text": f"Compressed a span containing {span_len} messages.",
            }
        )


@pytest.fixture
def fake_summarizer_client() -> FakeSummarizerClient:
    return FakeSummarizerClient()


def make_messages(n: int, start: int = 0) -> list[Message]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 200}
        for i in range(start, start + n)
    ]
