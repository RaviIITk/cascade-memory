"""The single generic interface a caller must implement to power summarization.

`cascade-memory` ships no provider SDKs. Wrap whichever LLM you already use
(Anthropic, OpenAI, a local model, ...) in an object satisfying this Protocol
and pass it to `MemoryConfig(summarizer_client=...)`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cascade_memory.schema import Message


@runtime_checkable
class ModelClient(Protocol):
    def complete(self, messages: list[Message], **kwargs: object) -> str:
        """Send `messages` to an LLM and return the assistant's text response."""
        ...
