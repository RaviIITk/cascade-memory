"""Library-specific exceptions."""

from __future__ import annotations


class CascadeMemoryError(Exception):
    """Base class for all `cascade_memory` errors."""


class MemoryRecordNotFoundError(CascadeMemoryError):
    """Raised when `load_memory` is called with an unknown `memory_id`."""

    def __init__(self, session_id: str, memory_id: str) -> None:
        super().__init__(f"No memory record {memory_id!r} found for session {session_id!r}")
        self.session_id = session_id
        self.memory_id = memory_id


class SummarizerNotConfiguredError(CascadeMemoryError):
    """Raised when an eviction is needed but no summarizer client was configured."""
