"""Claude Agent SDK adapter: compacts message history before each model turn.

Imports the SDK lazily so it stays optional in practice even though the base
package install bundles all adapter modules.
"""

from __future__ import annotations

from typing import Any

from cascade_memory.middleware import CascadingMemoryMiddleware
from cascade_memory.schema import Message


class CascadingMemoryHook:
    """A pre-model-call hook compatible with the Claude Agent SDK's hook lifecycle.

    Register `self.before_model_call` wherever the SDK exposes a "before sending to
    model" extension point; it mutates the outgoing message list in place via the
    return value, following whatever hook-result contract that SDK version expects.
    """

    def __init__(self, middleware: CascadingMemoryMiddleware, session_id: str) -> None:
        self.middleware = middleware
        self.session_id = session_id

    def before_model_call(self, messages: list[Message], **_: Any) -> list[Message]:
        return self.middleware.process(messages, session_id=self.session_id)
