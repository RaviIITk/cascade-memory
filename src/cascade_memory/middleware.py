"""Core engine: reconciles incoming messages against stored session state and,
once token_threshold is exceeded, evicts overflow into a new cascaded summary block.
"""

from __future__ import annotations

import uuid

from cascade_memory.config import MemoryConfig
from cascade_memory.schema import (
    MemoryRecord,
    Message,
    ProgressState,
    SessionMemoryState,
    SummaryBlock,
)
from cascade_memory.summarizer import Summarizer


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:8]}"


class CascadingMemoryMiddleware:
    """Framework-agnostic context manager: `process(messages, session_id) -> messages`.

    Call this immediately before every LLM invocation. It is a pure transform over
    the message list — no side effects on the caller's objects, only on the
    configured `MemoryStore` for the given `session_id`.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.store = config.build_store()
        self.summarizer = Summarizer(config.summarizer_client)

    def process(
        self, messages: list[Message], session_id: str, system_prompt: str | None = None
    ) -> list[Message]:
        state = self.store.get_state(session_id)
        if state is None:
            state = SessionMemoryState(session_id=session_id)

        self._merge_new_messages(state, messages)
        self._evict_if_over_threshold(state, system_prompt)

        self.store.put_state(session_id, state)
        return self._render_context(state, system_prompt)

    def load_memory(self, session_id: str, memory_id: str) -> list[Message]:
        return self.store.get_record(session_id, memory_id).original_messages

    def _merge_new_messages(self, state: SessionMemoryState, messages: list[Message]) -> None:
        already_seen = len(state.preserved_messages) + sum(
            block.covers_message_range[1] - block.covers_message_range[0] + 1
            for block in state.summary_blocks
        )
        new_messages = messages[already_seen:]
        state.preserved_messages.extend(new_messages)
        state.next_message_index += len(new_messages)

    def _evict_if_over_threshold(
        self, state: SessionMemoryState, system_prompt: str | None
    ) -> None:
        preserve_n = self.config.preserve_last_n
        while len(state.preserved_messages) > preserve_n:
            context = self._render_context(state, system_prompt)
            total_tokens = self.config.token_counter(context)
            if total_tokens <= self.config.token_threshold:
                break
            self._evict_one_span(state)

    def _evict_one_span(self, state: SessionMemoryState) -> None:
        preserve_n = self.config.preserve_last_n
        overflow_count = len(state.preserved_messages) - preserve_n
        span = state.preserved_messages[:overflow_count]
        if not span:
            return

        prior_progress = (
            state.summary_blocks[-1].progress_state if state.summary_blocks else ProgressState()
        )

        result = self.summarizer.summarize(span, prior_progress)

        start_index = state.next_message_index - len(state.preserved_messages)
        end_index = start_index + overflow_count - 1
        memory_id = _new_memory_id()

        block = SummaryBlock(
            memory_id=memory_id,
            session_id=state.session_id,
            covers_message_range=(start_index, end_index),
            token_count_original=self.config.token_counter(span),
            summary_text=result.summary_text,
            progress_state=result.progress_state,
            tool_call_summaries=result.tool_call_summaries,
        )
        record = MemoryRecord(
            memory_id=memory_id, session_id=state.session_id, original_messages=span
        )

        self.store.put_record(state.session_id, record)
        state.summary_blocks.append(block)
        state.preserved_messages = state.preserved_messages[overflow_count:]

    def _render_context(
        self, state: SessionMemoryState, system_prompt: str | None
    ) -> list[Message]:
        context: list[Message] = []
        if system_prompt:
            context.append({"role": "system", "content": system_prompt})
        for block in state.summary_blocks:
            context.append({"role": "system", "content": block.render()})
        context.extend(state.preserved_messages)
        return context
