"""Core data model shared across the middleware, summarizer, and memory stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

Message = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ProgressState:
    """Cumulative task-tracking state carried forward across every cascade step."""

    task: str = ""
    percent_complete: float = 0.0
    completed_steps: list[str] = field(default_factory=list)
    remaining_steps: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "percent_complete": self.percent_complete,
            "completed_steps": list(self.completed_steps),
            "remaining_steps": list(self.remaining_steps),
            "key_decisions": list(self.key_decisions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressState:
        return cls(
            task=data.get("task", ""),
            percent_complete=float(data.get("percent_complete", 0.0)),
            completed_steps=list(data.get("completed_steps", [])),
            remaining_steps=list(data.get("remaining_steps", [])),
            key_decisions=list(data.get("key_decisions", [])),
        )


@dataclass(slots=True)
class ToolCallSummary:
    purpose: str
    outcome: str
    memory_id_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "outcome": self.outcome,
            "memory_id_ref": self.memory_id_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallSummary:
        return cls(
            purpose=data.get("purpose", ""),
            outcome=data.get("outcome", ""),
            memory_id_ref=data.get("memory_id_ref"),
        )


@dataclass(slots=True)
class SummaryBlock:
    """A single compressed span of conversation history, retrievable by `memory_id`."""

    memory_id: str
    session_id: str
    covers_message_range: tuple[int, int]
    token_count_original: int
    summary_text: str
    progress_state: ProgressState
    tool_call_summaries: list[ToolCallSummary] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def render(self) -> str:
        """Render this block as the text injected into the model's context."""
        lines = [
            f"[memory_id: {self.memory_id}] (covers messages "
            f"{self.covers_message_range[0]}-{self.covers_message_range[1]}, "
            f"~{self.token_count_original} tokens)",
            self.summary_text,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "session_id": self.session_id,
            "covers_message_range": list(self.covers_message_range),
            "token_count_original": self.token_count_original,
            "summary_text": self.summary_text,
            "progress_state": self.progress_state.to_dict(),
            "tool_call_summaries": [t.to_dict() for t in self.tool_call_summaries],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryBlock:
        start, end = data["covers_message_range"]
        return cls(
            memory_id=data["memory_id"],
            session_id=data["session_id"],
            covers_message_range=(start, end),
            token_count_original=data["token_count_original"],
            summary_text=data["summary_text"],
            progress_state=ProgressState.from_dict(data.get("progress_state", {})),
            tool_call_summaries=[
                ToolCallSummary.from_dict(t) for t in data.get("tool_call_summaries", [])
            ],
            created_at=data.get("created_at", utc_now_iso()),
        )


@dataclass(slots=True)
class MemoryRecord:
    """Full-fidelity raw messages replaced by a `SummaryBlock`, kept for `load_memory`."""

    memory_id: str
    session_id: str
    original_messages: list[Message]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "session_id": self.session_id,
            "original_messages": self.original_messages,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        return cls(
            memory_id=data["memory_id"],
            session_id=data["session_id"],
            original_messages=data["original_messages"],
            created_at=data.get("created_at", utc_now_iso()),
        )


@dataclass(slots=True)
class SessionMemoryState:
    """Per-session state: the summary cascade plus the verbatim preserved tail."""

    session_id: str
    summary_blocks: list[SummaryBlock] = field(default_factory=list)
    preserved_messages: list[Message] = field(default_factory=list)
    next_message_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary_blocks": [b.to_dict() for b in self.summary_blocks],
            "preserved_messages": self.preserved_messages,
            "next_message_index": self.next_message_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMemoryState:
        return cls(
            session_id=data["session_id"],
            summary_blocks=[SummaryBlock.from_dict(b) for b in data.get("summary_blocks", [])],
            preserved_messages=list(data.get("preserved_messages", [])),
            next_message_index=data.get("next_message_index", 0),
        )
