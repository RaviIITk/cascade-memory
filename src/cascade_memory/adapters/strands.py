"""Strands Agents SDK plugin adapter.

Strands extends agent behavior via `Plugin` objects: `@hook`-decorated methods are
registered against typed lifecycle events (e.g. `BeforeModelCallEvent`), and
`@tool`-decorated methods are added to the agent's tool list. `strands` is not a
hard dependency of this package — only imported when `make_strands_plugin` is
actually called.
"""

from __future__ import annotations

from typing import Any

from cascade_memory.middleware import CascadingMemoryMiddleware
from cascade_memory.tools import LoadMemoryHandler


def make_strands_plugin(middleware: CascadingMemoryMiddleware, session_id: str) -> Any:
    """Build a Strands `Plugin` that compacts conversation messages before every
    model call, and exposes `load_memory` as a tool the agent can call.

    Use it when constructing a Strands `Agent`:

        from cascade_memory.adapters.strands import make_strands_plugin

        agent = Agent(plugins=[make_strands_plugin(memory_middleware, session_id="s1")])
    """
    try:
        from strands.hooks import BeforeModelCallEvent  # type: ignore[import-not-found]
        from strands.plugins import Plugin, hook, tool  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "strands-agents is required for cascade_memory.adapters.strands"
        ) from exc

    load_memory = LoadMemoryHandler(store=middleware.store, session_id=session_id)

    class CascadingMemoryPlugin(Plugin):  # type: ignore[misc]
        name = "cascade-memory"

        @hook  # type: ignore[untyped-decorator]
        def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
            event.request.messages = middleware.process(
                event.request.messages, session_id=session_id
            )

        @tool  # type: ignore[untyped-decorator]
        def load_memory_tool(self, memory_id: str) -> list[dict[str, Any]]:
            """Retrieve the original raw messages behind a summary block's memory_id."""
            return load_memory(memory_id=memory_id)

        def init_agent(self, agent: Any) -> None:
            pass

    return CascadingMemoryPlugin()
