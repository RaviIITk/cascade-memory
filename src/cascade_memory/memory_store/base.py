"""Storage interface for session state and evicted memory records."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cascade_memory.schema import MemoryRecord, SessionMemoryState


class MemoryStore(ABC):
    """Backend-agnostic persistence for `SessionMemoryState` and `MemoryRecord`s.

    Implementations key everything by `session_id`, so swapping backends (e.g.
    filesystem -> S3) requires no changes to the middleware.
    """

    @abstractmethod
    def get_state(self, session_id: str) -> SessionMemoryState | None: ...

    @abstractmethod
    def put_state(self, session_id: str, state: SessionMemoryState) -> None: ...

    @abstractmethod
    def get_record(self, session_id: str, memory_id: str) -> MemoryRecord: ...

    @abstractmethod
    def put_record(self, session_id: str, record: MemoryRecord) -> None: ...
