from __future__ import annotations

from cascade_memory.schema import (
    MemoryRecord,
    ProgressState,
    SessionMemoryState,
    SummaryBlock,
    ToolCallSummary,
)


def test_summary_block_roundtrip() -> None:
    block = SummaryBlock(
        memory_id="mem_abc123",
        session_id="s1",
        covers_message_range=(0, 4),
        token_count_original=120,
        summary_text="Did some stuff.",
        progress_state=ProgressState(task="Build X", percent_complete=50.0),
        tool_call_summaries=[ToolCallSummary(purpose="read file", outcome="ok")],
    )
    restored = SummaryBlock.from_dict(block.to_dict())
    assert restored == block


def test_summary_block_render_contains_memory_id_and_text() -> None:
    block = SummaryBlock(
        memory_id="mem_xyz",
        session_id="s1",
        covers_message_range=(0, 9),
        token_count_original=500,
        summary_text="Span summary here.",
        progress_state=ProgressState(),
    )
    rendered = block.render()
    assert "mem_xyz" in rendered
    assert "Span summary here." in rendered


def test_session_memory_state_roundtrip() -> None:
    state = SessionMemoryState(
        session_id="s1",
        summary_blocks=[
            SummaryBlock(
                memory_id="mem_1",
                session_id="s1",
                covers_message_range=(0, 1),
                token_count_original=10,
                summary_text="text",
                progress_state=ProgressState(task="t"),
            )
        ],
        preserved_messages=[{"role": "user", "content": "hi"}],
        next_message_index=5,
    )
    restored = SessionMemoryState.from_dict(state.to_dict())
    assert restored == state


def test_memory_record_roundtrip() -> None:
    record = MemoryRecord(
        memory_id="mem_1", session_id="s1", original_messages=[{"role": "user", "content": "hi"}]
    )
    restored = MemoryRecord.from_dict(record.to_dict())
    assert restored == record
