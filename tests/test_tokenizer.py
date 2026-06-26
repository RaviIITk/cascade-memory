from __future__ import annotations

from cascade_memory.tokenizer import default_tiktoken_counter


def test_counts_increase_with_more_messages() -> None:
    one = default_tiktoken_counter([{"role": "user", "content": "hello"}])
    two = default_tiktoken_counter(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
    )
    assert two > one


def test_empty_messages_is_zero() -> None:
    assert default_tiktoken_counter([]) == 0
