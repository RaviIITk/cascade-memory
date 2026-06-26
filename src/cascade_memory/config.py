"""User-facing configuration for `CascadingMemoryMiddleware`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cascade_memory.memory_store.base import MemoryStore
from cascade_memory.model_client import ModelClient
from cascade_memory.tokenizer import TokenCounter, default_tiktoken_counter

StorageBackend = Literal["filesystem", "s3"]


@dataclass(slots=True)
class MemoryConfig:
    """Tunables for the cascading memory middleware.

    `summarizer_client` is required to actually compress overflow history; without
    it the middleware can still track state but raises on the first eviction.
    """

    summarizer_client: ModelClient
    token_threshold: int = 10_000
    preserve_last_n: int = 20
    token_counter: TokenCounter = field(default=default_tiktoken_counter)
    storage_backend: StorageBackend = "filesystem"
    storage_path: str = "./memory"
    store: MemoryStore | None = None

    def __post_init__(self) -> None:
        if self.token_threshold <= 0:
            raise ValueError("token_threshold must be positive")
        if self.preserve_last_n < 0:
            raise ValueError("preserve_last_n cannot be negative")

    def build_store(self) -> MemoryStore:
        if self.store is not None:
            return self.store
        if self.storage_backend == "filesystem":
            from cascade_memory.memory_store.filesystem import FilesystemMemoryStore

            return FilesystemMemoryStore(root=self.storage_path)
        if self.storage_backend == "s3":
            from cascade_memory.memory_store.s3 import S3MemoryStore

            return S3MemoryStore(bucket=self.storage_path)
        raise ValueError(f"Unknown storage_backend: {self.storage_backend!r}")
