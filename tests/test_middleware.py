from __future__ import annotations

from cascade_memory.config import MemoryConfig
from cascade_memory.memory_store.filesystem import FilesystemMemoryStore
from cascade_memory.middleware import CascadingMemoryMiddleware
from tests.conftest import FakeSummarizerClient, make_messages


def _build_middleware(
    tmp_path: object, fake_summarizer_client: FakeSummarizerClient, **overrides: object
) -> CascadingMemoryMiddleware:
    store = FilesystemMemoryStore(root=str(tmp_path) + "/memory")
    config = MemoryConfig(
        summarizer_client=fake_summarizer_client,
        token_threshold=overrides.pop("token_threshold", 300),  # type: ignore[arg-type]
        preserve_last_n=overrides.pop("preserve_last_n", 5),  # type: ignore[arg-type]
        store=store,
        **overrides,  # type: ignore[arg-type]
    )
    return CascadingMemoryMiddleware(config)


def test_no_eviction_when_under_threshold(
    tmp_path: object, fake_summarizer_client: FakeSummarizerClient
) -> None:
    middleware = _build_middleware(tmp_path, fake_summarizer_client, token_threshold=100_000)
    messages = make_messages(4)
    result = middleware.process(messages, session_id="s1")
    assert result == messages
    assert fake_summarizer_client.calls == []


def test_eviction_preserves_last_n_and_creates_summary_block(
    tmp_path: object, fake_summarizer_client: FakeSummarizerClient
) -> None:
    middleware = _build_middleware(tmp_path, fake_summarizer_client)
    messages = make_messages(20)
    result = middleware.process(messages, session_id="s1")

    state = middleware.store.get_state("s1")
    assert state is not None
    assert len(state.summary_blocks) == 1
    assert len(state.preserved_messages) == 5

    summary_messages = [m for m in result if m.get("role") == "system"]
    assert any("mem_" in m["content"] for m in summary_messages)
    assert result[-5:] == messages[-5:]


def test_cascade_appends_new_block_without_overwriting_prior(
    tmp_path: object, fake_summarizer_client: FakeSummarizerClient
) -> None:
    middleware = _build_middleware(tmp_path, fake_summarizer_client)

    first_batch = make_messages(20, start=0)
    middleware.process(first_batch, session_id="s1")
    state_after_first = middleware.store.get_state("s1")
    assert state_after_first is not None
    first_block_id = state_after_first.summary_blocks[0].memory_id

    second_batch = first_batch + make_messages(20, start=20)
    middleware.process(second_batch, session_id="s1")
    state_after_second = middleware.store.get_state("s1")
    assert state_after_second is not None

    assert len(state_after_second.summary_blocks) == 2
    assert state_after_second.summary_blocks[0].memory_id == first_block_id
    ranges = [b.covers_message_range for b in state_after_second.summary_blocks]
    assert ranges[0][1] < ranges[1][0]  # contiguous, non-overlapping spans


def test_load_memory_returns_original_evicted_messages(
    tmp_path: object, fake_summarizer_client: FakeSummarizerClient
) -> None:
    middleware = _build_middleware(tmp_path, fake_summarizer_client)
    messages = make_messages(20)
    middleware.process(messages, session_id="s1")

    state = middleware.store.get_state("s1")
    assert state is not None
    memory_id = state.summary_blocks[0].memory_id

    original = middleware.load_memory("s1", memory_id)
    expected_span = messages[: len(messages) - 5]
    assert original == expected_span


def test_progress_state_carries_forward_across_cascades(
    tmp_path: object, fake_summarizer_client: FakeSummarizerClient
) -> None:
    middleware = _build_middleware(tmp_path, fake_summarizer_client)
    middleware.process(make_messages(20, start=0), session_id="s1")
    middleware.process(make_messages(20, start=0) + make_messages(20, start=20), session_id="s1")

    state = middleware.store.get_state("s1")
    assert state is not None
    assert len(fake_summarizer_client.calls) == 2
    # second call's user prompt should include the first block's progress state
    second_call_user_prompt = fake_summarizer_client.calls[1][1]["content"]
    assert "Build the widget" in second_call_user_prompt
