"""The `load_memory` tool: lets the agent pull back raw content behind a summary block."""

from __future__ import annotations

from typing import Any

from cascade_memory.memory_store.base import MemoryStore
from cascade_memory.schema import Message

LOAD_MEMORY_TOOL_SPEC: dict[str, Any] = {
    "name": "load_memory",
    "description": (
        "Retrieve the original raw messages that a summary block compressed, given its "
        "memory_id. Use this only when the summary text isn't enough detail to proceed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory_id shown in the summary block, e.g. 'mem_3f9a'.",
            }
        },
        "required": ["memory_id"],
    },
}


class LoadMemoryHandler:
    """Resolves a `load_memory` tool call against a `MemoryStore` for one session."""

    def __init__(self, store: MemoryStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    def __call__(self, memory_id: str) -> list[Message]:
        record = self._store.get_record(self._session_id, memory_id)
        return record.original_messages
