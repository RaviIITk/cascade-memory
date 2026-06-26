"""Dedicated summarization call, separate from the main agent loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from cascade_memory.model_client import ModelClient
from cascade_memory.prompts import SUMMARIZER_SYSTEM_PROMPT, build_summarizer_user_prompt
from cascade_memory.schema import Message, ProgressState, ToolCallSummary

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(slots=True)
class SummarizerResult:
    summary_text: str
    progress_state: ProgressState
    tool_call_summaries: list[ToolCallSummary]


class SummarizationParseError(ValueError):
    """Raised when the summarizer's response can't be parsed as the expected JSON shape."""


def _extract_json(raw_text: str) -> dict[str, Any]:
    match = _JSON_FENCE_RE.search(raw_text)
    candidate = match.group(1) if match else raw_text.strip()
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SummarizationParseError(
            f"Summarizer response was not valid JSON: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise SummarizationParseError("Summarizer response JSON must be an object")
    return result


class Summarizer:
    """Wraps a `ModelClient` with the cascade-memory summarization prompt/parsing contract."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def summarize(self, span: list[Message], prior_progress: ProgressState) -> SummarizerResult:
        if not span:
            raise ValueError("Cannot summarize an empty message span")

        messages: list[Message] = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": build_summarizer_user_prompt(prior_progress, span)},
        ]
        raw_response = self._client.complete(messages)
        parsed = _extract_json(raw_response)

        progress_state = ProgressState(
            task=str(parsed.get("task", prior_progress.task)),
            percent_complete=float(parsed.get("percent_complete", prior_progress.percent_complete)),
            completed_steps=list(parsed.get("completed_steps", prior_progress.completed_steps)),
            remaining_steps=list(parsed.get("remaining_steps", prior_progress.remaining_steps)),
            key_decisions=list(parsed.get("key_decisions", prior_progress.key_decisions)),
        )
        tool_call_summaries = [
            ToolCallSummary.from_dict(item) for item in parsed.get("tool_call_summaries", [])
        ]
        summary_text = str(parsed.get("summary_text", "")).strip()
        if not summary_text:
            raise SummarizationParseError("Summarizer response missing 'summary_text'")

        return SummarizerResult(
            summary_text=summary_text,
            progress_state=progress_state,
            tool_call_summaries=tool_call_summaries,
        )
