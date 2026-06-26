"""LangChain adapter: a callback handler that compacts `messages` before each LLM call.

LangChain is not a hard dependency of `cascade-memory` — this module only imports it
when actually used, so the base package install stays lean even though it's bundled.
"""

from __future__ import annotations

from typing import Any

from cascade_memory.middleware import CascadingMemoryMiddleware
from cascade_memory.schema import Message


class CascadingMemoryCallbackHandler:
    """Use as a LangChain `BaseCallbackHandler`-compatible `on_chat_model_start` hook.

    LangChain callbacks cannot mutate the outgoing request, so this handler is meant
    to be paired with a `Runnable` wrapper (see `wrap_runnable`) rather than used alone.
    """

    def __init__(self, middleware: CascadingMemoryMiddleware, session_id: str) -> None:
        self.middleware = middleware
        self.session_id = session_id


def wrap_runnable(runnable: Any, middleware: CascadingMemoryMiddleware, session_id: str) -> Any:
    """Wrap a LangChain `Runnable` so its `invoke`/`ainvoke` input messages are
    compacted through the middleware first.

    `runnable` is expected to accept a `dict` with a `"messages"` key (the standard
    LangGraph/LCEL chat-message convention).
    """
    try:
        from langchain_core.runnables import RunnableLambda  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "langchain_core is required for cascade_memory.adapters.langchain.wrap_runnable"
        ) from exc

    def _compact(payload: dict[str, Any]) -> dict[str, Any]:
        messages: list[Message] = payload.get("messages", [])
        compacted = middleware.process(messages, session_id=session_id)
        return {**payload, "messages": compacted}

    return RunnableLambda(_compact) | runnable
