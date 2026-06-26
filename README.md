# cascade-memory

Framework-agnostic cascading context/memory manager for LLM agent loops.

When a conversation's token count exceeds a configurable threshold, `cascade-memory`
compresses everything except the last `N` messages into a structured summary block
(carrying forward task progress, completed/remaining steps, and key decisions) and
keeps the original raw messages retrievable by `memory_id` via a `load_memory` tool.
Each time the threshold fires again, a **new** summary block is appended — never
merged into prior ones — forming a cascade of compression spans across the session:

```
[system prompt]
summary_block_1 (memory_id: mem_001)   <- oldest compressed span
summary_block_2 (memory_id: mem_002)
...
summary_block_n (memory_id: mem_00n)   <- most recent compressed span
[preserved last N raw messages]
[current turn]
```

## Install

> Not yet published to PyPI — install directly from GitHub or from a local clone for now.

```bash
# straight from GitHub, no clone needed
pip install git+https://github.com/RaviIITk/cascade-memory.git

# or, from a local clone
git clone https://github.com/RaviIITk/cascade-memory.git
cd cascade-memory
pip install -e .          # editable install
# pip install -e ".[dev]" # + test/lint/type-check tooling
```

Once published, the above will collapse to `pip install cascade-memory`.

## How it works, end to end

### 1. Implement `ModelClient` for your summarizer call

The package ships zero provider SDKs. You give it **one object** with a `complete()`
method — wrap whatever LLM client you already use (Anthropic, OpenAI, a local model,
anything). This is a *separate* call from your main agent's model; it only runs when
history needs compressing.

```python
from cascade_memory import ModelClient

class MySummarizerClient:
    def complete(self, messages: list[dict], **kwargs) -> str:
        # messages is [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        # return the assistant's raw text response (cascade-memory parses the JSON itself)
        response = my_llm_sdk.chat(messages=messages, model="gpt-4o-mini")
        return response.text
```

Any object satisfying `ModelClient` (structurally — no subclassing required) works.

### 2. Configure and build the middleware

```python
from cascade_memory import CascadingMemoryMiddleware, MemoryConfig

config = MemoryConfig(
    summarizer_client=MySummarizerClient(),
    token_threshold=10_000,     # default: compress once context exceeds this many tokens
    preserve_last_n=20,         # default: always keep the last 20 messages verbatim
    storage_backend="filesystem",  # or "s3"
    storage_path="./memory",       # fs root dir, or S3 bucket name when backend="s3"
)
middleware = CascadingMemoryMiddleware(config)
```

Other `MemoryConfig` fields:

| Field | Default | Purpose |
|---|---|---|
| `summarizer_client` | required | Your `ModelClient` for the summarization call |
| `token_threshold` | `10_000` | Trigger point for compression |
| `preserve_last_n` | `20` | Messages always kept raw |
| `token_counter` | tiktoken `cl100k_base` | Swap in your own `Callable[[list[dict]], int]` to match your model's tokenizer |
| `storage_backend` | `"filesystem"` | `"filesystem"` or `"s3"` |
| `storage_path` | `"./memory"` | FS root, or S3 bucket name |
| `store` | `None` | Pass a custom `MemoryStore` instance directly instead of backend/path |

### 3. Run every message list through it before calling your main agent's LLM

This is a pure function: `messages -> messages`, keyed by `session_id`. Call it
immediately before every turn, anywhere in your loop:

```python
session_id = "user-42-conversation-7"

def run_turn(history: list[dict]) -> dict:
    compacted = middleware.process(history, session_id=session_id, system_prompt="You are a helpful agent.")
    response = my_main_agent_llm.chat(messages=compacted, tools=[...])
    return response
```

`process()` automatically:
- diffs `history` against what it already has stored for `session_id` and appends only new messages,
- checks token count of the reconstructed context,
- if over `token_threshold`, evicts the oldest messages beyond `preserve_last_n` into a new cascaded summary block (calling your `summarizer_client` once per eviction),
- persists everything to the configured store,
- returns the final context to send to the model.

### 4. Let the agent recover original content on demand

Evicted raw messages aren't gone — they're stored under the `memory_id` shown in
each summary block's rendered text. Wire up the `load_memory` tool so the main
agent can pull them back when a summary alone isn't enough:

```python
from cascade_memory.tools import LOAD_MEMORY_TOOL_SPEC, LoadMemoryHandler

load_memory = LoadMemoryHandler(store=middleware.store, session_id=session_id)

# 1. include the tool spec in your LLM call alongside your other tools
tools = [LOAD_MEMORY_TOOL_SPEC, ...]

# 2. when the model calls it, resolve with the handler
def handle_tool_call(tool_call) -> list[dict]:
    if tool_call.name == "load_memory":
        return load_memory(memory_id=tool_call.input["memory_id"])
    ...
```

The handler returns the original raw message list that summary block replaced —
inject it back as a tool result for that turn (don't permanently merge it into
preserved history; it's a one-off lookup).

### 5. Storage backends

- **`filesystem`** (default): `{storage_path}/{session_id}/state.json` and
  `{storage_path}/{session_id}/records/{memory_id}.json`.
- **`s3`**: same logical layout under `s3://{storage_path}/{session_id}/...`. Uses
  `boto3` with default credential resolution (env vars, profile, instance role, etc.):

  ```python
  config = MemoryConfig(
      summarizer_client=MySummarizerClient(),
      storage_backend="s3",
      storage_path="my-bucket-name",
  )
  ```

  For a custom prefix or an explicit `boto3` client (e.g. a non-default region or
  assumed-role session), construct the store yourself and pass it directly:

  ```python
  from cascade_memory.memory_store.s3 import S3MemoryStore
  import boto3

  store = S3MemoryStore(bucket="my-bucket", prefix="agents/prod", client=boto3.client("s3", region_name="eu-west-1"))
  config = MemoryConfig(summarizer_client=MySummarizerClient(), store=store)
  ```

### 6. Framework adapters

For raw API loops (no framework), wrap your chat function directly:

```python
from cascade_memory.adapters import wrap_chat_function

chat = wrap_chat_function(my_llm_client.chat, middleware, session_id="session-123")
response = chat(messages)  # messages are compacted automatically before the call
```

For LangChain (LCEL/LangGraph `Runnable`s expecting `{"messages": [...]}`):

```python
from cascade_memory.adapters.langchain import wrap_runnable

chain = wrap_runnable(my_runnable, middleware, session_id="session-123")
chain.invoke({"messages": history})
```

For the Claude Agent SDK, register the hook at whatever "before model call" extension
point that SDK version exposes:

```python
from cascade_memory.adapters.claude_agent_sdk import CascadingMemoryHook

hook = CascadingMemoryHook(middleware, session_id="session-123")
# register hook.before_model_call per the SDK's hook registration API
```

For **DeepAgents** (LangChain 1.0's middleware-based agent harness) — DeepAgents
runs every model call through a composable `AgentMiddleware` stack, where each
middleware implements `wrap_model_call(request, handler)` and can rewrite
`request.messages` before calling `handler()`. `make_deepagents_middleware` builds
exactly that middleware around `cascade_memory`:

```python
from deepagents import create_deep_agent
from cascade_memory.adapters.deepagents import make_deepagents_middleware

agent = create_deep_agent(
    model=model,
    tools=tools,
    middleware=[make_deepagents_middleware(middleware, session_id="session-123")],
)
agent.invoke({"messages": history})
```

Because it's a normal middleware entry, it composes cleanly alongside DeepAgents'
other built-ins (`SubAgentMiddleware`, `FilesystemMiddleware`, `SummarizationToolMiddleware`,
etc.) — `cascade_memory` simply runs as one more layer in that same stack, guaranteeing
every model call sees a bounded context regardless of what the other middleware does.

For **Strands Agents** (AWS's agent SDK) — Strands extends agents via `Plugin`
objects: `@hook`-decorated methods register against typed lifecycle events (here,
`BeforeModelCallEvent`), and `@tool`-decorated methods are added to the agent's
tool list automatically. `make_strands_plugin` builds a plugin that both compacts
messages before every model call *and* registers `load_memory` as a callable tool
in one step:

```python
from strands import Agent
from cascade_memory.adapters.strands import make_strands_plugin

agent = Agent(
    model="anthropic.claude-sonnet-4-6",
    plugins=[make_strands_plugin(middleware, session_id="session-123")],
)
agent("What's the status of the task we started earlier?")
```

No separate tool wiring needed here — the plugin's `init_agent` hook attaches
`load_memory_tool` to the agent for you, so the model can call it directly whenever
a summary block isn't detailed enough.

### 7. Integrating as a hook/middleware plugin in *any* framework

`cascade_memory.adapters` ships ready-made wrappers for raw chat loops, LangChain,
and the Claude Agent SDK — but the middleware is designed to drop into **any**
agentic framework's hook/middleware system, even ones with no dedicated adapter
yet. The contract is always the same one-line shape:

```python
new_messages = middleware.process(messages, session_id=session_id, system_prompt=system_prompt)
```

Because `process()` is a pure `list[dict] -> list[dict]` transform with no side
effects on the framework's own objects, plugging it in is just a matter of finding
where your framework lets you intercept the outgoing message list right before the
LLM call, and calling `process()` there. Three common shapes:

**1. Pre-call hook (most frameworks)** — register a callback that runs before each
model invocation and returns the (possibly modified) messages:

```python
def cascade_memory_hook(messages: list[dict], **context) -> list[dict]:
    return middleware.process(messages, session_id=context["session_id"])

agent_framework.register_pre_model_hook(cascade_memory_hook)
```

**2. Middleware/interceptor chain (frameworks with an explicit middleware stack)**
— wrap the next handler so it always receives compacted input:

```python
class CascadingMemoryPlugin:
    def __init__(self, middleware, session_id):
        self.middleware = middleware
        self.session_id = session_id

    def __call__(self, request, call_next):
        request.messages = self.middleware.process(request.messages, session_id=self.session_id)
        return call_next(request)

agent_framework.use_middleware(CascadingMemoryPlugin(middleware, session_id="session-123"))
```

**3. Decorator around the model-call function** — for frameworks that just expose
a plain function you call to talk to the model (this is exactly what
`cascade_memory.adapters.wrap_chat_function` does — see §6 above):

```python
def with_cascading_memory(chat_fn, middleware, session_id):
    def wrapped(messages, **kwargs):
        return chat_fn(middleware.process(messages, session_id=session_id), **kwargs)
    return wrapped

agent.chat = with_cascading_memory(agent.chat, middleware, session_id="session-123")
```

Whichever shape your framework uses, the rule is: **call `process()` last, right
before the messages leave your code for the model** — that's what guarantees the
context sent to the LLM is always within `token_threshold`, regardless of how big
the underlying conversation history has grown.

## Full minimal example

```python
from cascade_memory import CascadingMemoryMiddleware, MemoryConfig
from cascade_memory.tools import LOAD_MEMORY_TOOL_SPEC, LoadMemoryHandler

class AnthropicSummarizerClient:
    def __init__(self, client):
        self.client = client

    def complete(self, messages, **kwargs) -> str:
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

import anthropic
client = anthropic.Anthropic()

middleware = CascadingMemoryMiddleware(MemoryConfig(
    summarizer_client=AnthropicSummarizerClient(client),
    token_threshold=10_000,
    preserve_last_n=20,
))

session_id = "demo-session"
history: list[dict] = []

def turn(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    compacted = middleware.process(history, session_id=session_id)
    resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, messages=compacted)
    reply = resp.content[0].text
    history.append({"role": "assistant", "content": reply})
    return reply
```

## Live verification

Beyond the unit/integration test suite (mocked `ModelClient`), every adapter was
also run against a real LLM (NVIDIA NIM's OpenAI-compatible endpoint) to catch
issues that only show up with real message shapes and real model behavior. This
surfaced two real bugs, both fixed and covered by what's documented above:

| Framework | What was actually run | Result |
|---|---|---|
| **Raw adapter** | 6-turn live conversation through `wrap_chat_function` | 3 cascaded summary blocks created; `progress_state` correctly carried forward and merged across them; `load_memory` confirmed to return the exact original evicted messages |
| **LangChain Runnable** | 5-turn live conversation through `wrap_runnable` | 3 cascaded summary blocks created |
| **DeepAgents** | 5-turn live conversation through `make_deepagents_middleware` with `create_deep_agent` | Found and fixed a real bug: LangChain passes `BaseMessage` objects (`HumanMessage`/`AIMessage`/...), not plain dicts — this crashed the JSON-backed store on first eviction. Fixed by converting at the boundary with `convert_to_openai_messages`/`convert_to_messages`. After the fix: full live run, multiple cascaded blocks created |
| **Claude Agent SDK** | `CascadingMemoryHook.before_model_call` called directly with a live summarizer | 3 cascaded blocks created correctly. The SDK's own agent loop only talks to Anthropic models, so a full SDK-driven run wasn't possible here — the hook logic itself is what's verified |
| **Strands Agents** | Plugin mechanics (hook firing, `@tool` auto-registration, message conversion) plus one full live turn | Found and fixed two real bugs: `BeforeModelCallEvent` carries no `messages` field (the live conversation is on `event.agent.messages`), and Strands uses Bedrock-style content blocks (`[{"text": ...}]`), not flat `{"role", "content"}` dicts. Also discovered Strands' `@hook` decorator infers its event type via `typing.get_type_hints`, which fails if the event type isn't a real module-level name — a local import inside a factory function silently breaks registration. After fixing all three: hook fires correctly, tool auto-registers, one full multi-turn cascade observed live; further runs were occasionally blocked by intermittent hangs in NVIDIA NIM's endpoint itself (reproduced even with a bare `Agent`, no plugin involved) |

Takeaways if you're integration-testing against your own LLM endpoint:
- Don't assume `list[dict]` — frameworks built on LangChain pass typed message
  objects, and Strands passes Bedrock-style content blocks. Convert at the
  adapter boundary, not inside the core middleware.
- If a framework's event/hook type inference relies on `typing.get_type_hints`
  (Strands does), keep the relevant imports as real module-level names, not
  local imports inside a closure/factory function.
- Smaller instruction-tuned models can be unreliable tool-callers (hallucinated
  tool names, ignoring "don't use tools" instructions) — this looks like a
  framework or adapter bug but is actually a model-capability issue. Test with
  a model known to support function-calling well before debugging further.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

CI (`.github/workflows/ci.yml`) runs lint, type-check, and tests on Python 3.10–3.12
on every push/PR, and publishes to PyPI on `v*` tag pushes (once `PYPI_API_TOKEN` is
configured as a repo secret).
