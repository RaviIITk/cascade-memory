"""Strands Agents SDK plugin adapter.

Strands extends agent behavior via `Plugin` objects: `@hook`-decorated methods are
registered against typed lifecycle events (e.g. `BeforeModelCallEvent`), and
`@tool`-decorated methods are added to the agent's tool list. `strands` is not a
hard dependency of this package — it's only actually imported when
`make_strands_plugin` is called (the import happens at module load time of *this*
submodule, not the top-level `cascade_memory` package).

Strands messages are Bedrock-style content blocks
(`{"role": ..., "content": [{"text": ...}, ...]}`), not the flat
`{"role": ..., "content": "..."}` dicts `cascade_memory` works on, and
`BeforeModelCallEvent` itself carries no `messages` field -- the live conversation
lives on `event.agent.messages`. This adapter converts at that boundary; only plain
text content blocks round-trip (tool-use/tool-result/image blocks are not preserved
through compaction in this version).

Note: Strands' `@hook` decorator infers the event type from the callback's type
hint via `typing.get_type_hints`, which resolves names against the function's
*module* globals -- so the Strands imports below must be real module-level names
(not local imports inside a factory function) for the hook to register correctly.
"""

from __future__ import annotations

import json
from typing import Any

from cascade_memory.middleware import CascadingMemoryMiddleware
from cascade_memory.tools import LoadMemoryHandler

try:
    # NOTE: BeforeModelCallEvent must keep this exact name at module level --
    # Strands' @hook decorator resolves the callback's string annotation against
    # this module's globals via typing.get_type_hints.
    from strands import tool as _strands_tool
    from strands.hooks import BeforeModelCallEvent
    from strands.plugins import Plugin as _StrandsPlugin
    from strands.plugins import hook as _strands_hook

    _STRANDS_AVAILABLE = True
except ImportError:
    _STRANDS_AVAILABLE = False


def _strands_messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        text = "".join(block.get("text", "") for block in message.get("content", []))
        result.append({"role": message["role"], "content": text})
    return result


def _dicts_to_strands_messages(messages: list[dict[str, Any]]) -> list[Any]:
    return [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]


if _STRANDS_AVAILABLE:

    class _CascadingMemoryPlugin(_StrandsPlugin):  # type: ignore[misc]
        name = "cascade-memory"

        def __init__(self, middleware: CascadingMemoryMiddleware, session_id: str) -> None:
            self._middleware = middleware
            self._session_id = session_id
            self._load_memory = LoadMemoryHandler(store=middleware.store, session_id=session_id)
            super().__init__()

        @_strands_hook  # type: ignore[untyped-decorator]
        def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
            as_dicts = _strands_messages_to_dicts(event.agent.messages)
            compacted = self._middleware.process(as_dicts, session_id=self._session_id)
            event.agent.messages = _dicts_to_strands_messages(compacted)

        @_strands_tool  # type: ignore[untyped-decorator]
        def load_memory_tool(self, memory_id: str) -> str:
            """Retrieve the original raw messages behind a summary block's memory_id."""
            return json.dumps(self._load_memory(memory_id=memory_id))


def make_strands_plugin(middleware: CascadingMemoryMiddleware, session_id: str) -> Any:
    """Build a Strands `Plugin` that compacts conversation messages before every
    model call, and exposes `load_memory` as a tool the agent can call.

    Use it when constructing a Strands `Agent`:

        from cascade_memory.adapters.strands import make_strands_plugin

        agent = Agent(plugins=[make_strands_plugin(memory_middleware, session_id="s1")])
    """
    if not _STRANDS_AVAILABLE:
        raise ImportError("strands-agents is required for cascade_memory.adapters.strands")
    return _CascadingMemoryPlugin(middleware, session_id)
