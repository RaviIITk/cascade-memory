"""Generic adapter: wraps any `messages -> response` chat function with the middleware.

Use this for raw API loops or any framework without a dedicated adapter — it requires
no knowledge of the framework's internals, only that it exposes a callable taking
`messages` and returning a response.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from cascade_memory.middleware import CascadingMemoryMiddleware
from cascade_memory.schema import Message

R = TypeVar("R")


def wrap_chat_function(
    chat_fn: Callable[[list[Message]], R],
    middleware: CascadingMemoryMiddleware,
    session_id: str,
    system_prompt: str | None = None,
) -> Callable[[list[Message]], R]:
    """Return a drop-in replacement for `chat_fn` that runs messages through the
    middleware before every call.

    Example:
        client = MyLLMClient()
        chat = wrap_chat_function(client.chat, middleware, session_id="abc123")
        response = chat(messages)
    """

    def wrapped(messages: list[Message]) -> R:
        compacted = middleware.process(messages, session_id=session_id, system_prompt=system_prompt)
        return chat_fn(compacted)

    return wrapped
