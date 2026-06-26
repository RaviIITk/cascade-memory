# cascade-memory

Framework-agnostic cascading context/memory manager for LLM agent loops.

When a conversation's token count exceeds a configurable threshold, `cascade-memory`
compresses everything except the last `N` messages into a structured summary block
(carrying forward task progress, completed/remaining steps, and key decisions) and
keeps the original raw messages retrievable by `memory_id` via a `load_memory` tool.
Each time the threshold fires again, a new summary block is appended — never merged
into prior ones — forming a cascade of compression spans across the session.

## Install

```bash
pip install cascade-memory
```

## Quick start

```python
from cascade_memory import CascadingMemoryMiddleware, MemoryConfig

class MySummarizerClient:
    """Implements the ModelClient protocol: complete(messages) -> str."""
    def complete(self, messages, **kwargs):
        return my_llm.chat(messages)  # any provider, your own wiring

config = MemoryConfig(
    summarizer_client=MySummarizerClient(),
    token_threshold=10_000,
    preserve_last_n=20,
    storage_backend="filesystem",
    storage_path="./memory",
)
middleware = CascadingMemoryMiddleware(config)

compacted_messages = middleware.process(messages, session_id="session-123")
# send compacted_messages to your main agent's LLM call as usual
```

To let the agent recover original content behind a summary block, register
`cascade_memory.tools.LOAD_MEMORY_TOOL_SPEC` as a tool and route calls through
`LoadMemoryHandler(store, session_id)`.

## Storage backends

- `filesystem` (default): `./memory/{session_id}/state.json` and `.../records/{memory_id}.json`.
- `s3`: same key layout under `s3://{bucket}/{prefix}/...`.

## Adapters

`cascade_memory.adapters` ships a generic `wrap_chat_function` for any raw chat
function, plus optional LangChain and Claude Agent SDK adapters.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```
