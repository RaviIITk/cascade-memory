from __future__ import annotations

import json

import pytest

from cascade_memory.schema import ProgressState
from cascade_memory.summarizer import SummarizationParseError, Summarizer
from tests.conftest import FakeSummarizerClient, make_messages


def test_summarize_parses_fake_client_response(
    fake_summarizer_client: FakeSummarizerClient,
) -> None:
    summarizer = Summarizer(fake_summarizer_client)
    span = make_messages(6)
    result = summarizer.summarize(span, ProgressState())

    assert "6 messages" in result.summary_text
    assert result.progress_state.task == "Build the widget"
    assert result.progress_state.percent_complete == 42.0
    assert result.tool_call_summaries[0].purpose == "test tool call"


def test_summarize_rejects_empty_span(fake_summarizer_client: FakeSummarizerClient) -> None:
    summarizer = Summarizer(fake_summarizer_client)
    with pytest.raises(ValueError):
        summarizer.summarize([], ProgressState())


def test_summarize_handles_markdown_fenced_json() -> None:
    class FencedClient:
        def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            payload = {
                "task": "t",
                "percent_complete": 10,
                "completed_steps": [],
                "remaining_steps": [],
                "key_decisions": [],
                "tool_call_summaries": [],
                "summary_text": "fenced summary",
            }
            return f"```json\n{json.dumps(payload)}\n```"

    summarizer = Summarizer(FencedClient())
    result = summarizer.summarize(make_messages(2), ProgressState())
    assert result.summary_text == "fenced summary"


def test_summarize_raises_on_invalid_json() -> None:
    class BadClient:
        def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            return "not json at all"

    summarizer = Summarizer(BadClient())
    with pytest.raises(SummarizationParseError):
        summarizer.summarize(make_messages(2), ProgressState())


def test_summarize_raises_when_summary_text_missing() -> None:
    class NoSummaryTextClient:
        def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            return json.dumps({"task": "t"})

    summarizer = Summarizer(NoSummaryTextClient())
    with pytest.raises(SummarizationParseError):
        summarizer.summarize(make_messages(2), ProgressState())
