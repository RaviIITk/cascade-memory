"""DeepAgents / LangChain 1.0 middleware adapter.

DeepAgents (and LangChain agents generally) run on a composable `AgentMiddleware`
stack where each middleware implements `wrap_model_call(request, handler)` to wrap
every model invocation. `langchain` is not a hard dependency of this package — it's
only imported when `make_deepagents_middleware` is actually called.
"""

from __future__ import annotations

from typing import Any

from cascade_memory.middleware import CascadingMemoryMiddleware


def make_deepagents_middleware(middleware: CascadingMemoryMiddleware, session_id: str) -> Any:
    """Build an `AgentMiddleware` instance that compacts `request.messages` via
    `cascade_memory` before every model call.

    Use it in a DeepAgents / `create_agent` middleware list:

        from cascade_memory.adapters.deepagents import make_deepagents_middleware

        agent = create_deep_agent(
            model=model,
            tools=tools,
            middleware=[make_deepagents_middleware(memory_middleware, session_id="s1")],
        )
    """
    try:
        from langchain.agents.middleware import AgentMiddleware  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "langchain>=1.0 is required for cascade_memory.adapters.deepagents"
        ) from exc

    class CascadingMemoryAgentMiddleware(AgentMiddleware):  # type: ignore[misc]
        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            compacted = middleware.process(request.messages, session_id=session_id)
            return handler(request.override(messages=compacted))

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            compacted = middleware.process(request.messages, session_id=session_id)
            return await handler(request.override(messages=compacted))

    return CascadingMemoryAgentMiddleware()
