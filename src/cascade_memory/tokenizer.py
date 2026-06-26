"""Token counting for message lists. Pluggable so callers can match their model's tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from cascade_memory.schema import Message

if TYPE_CHECKING:
    from tiktoken import Encoding

_ENCODING_NAME = "cl100k_base"
_encoder: Encoding | None = None


class TokenCounter(Protocol):
    def __call__(self, messages: list[Message]) -> int: ...


def _get_encoder() -> Encoding:
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoder


def default_tiktoken_counter(messages: list[Message]) -> int:
    """Approximate token count using tiktoken's cl100k_base encoding.

    This is an approximation for non-OpenAI models, but stable and dependency-light
    enough to use as a default trigger signal.
    """
    encoder = _get_encoder()
    total = 0
    for message in messages:
        for value in message.values():
            text = value if isinstance(value, str) else str(value)
            total += len(encoder.encode(text))
        total += 4  # per-message role/formatting overhead
    return total
