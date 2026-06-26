"""Local filesystem implementation of `MemoryStore`, one directory per session."""

from __future__ import annotations

import json
from pathlib import Path

from cascade_memory.exceptions import MemoryRecordNotFoundError
from cascade_memory.memory_store.base import MemoryStore
from cascade_memory.schema import MemoryRecord, SessionMemoryState


class FilesystemMemoryStore(MemoryStore):
    """Lays out `{root}/{session_id}/state.json` and `.../records/{memory_id}.json`."""

    def __init__(self, root: str | Path = "./memory") -> None:
        self.root = Path(root)

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"

    def _record_path(self, session_id: str, memory_id: str) -> Path:
        return self._session_dir(session_id) / "records" / f"{memory_id}.json"

    def get_state(self, session_id: str) -> SessionMemoryState | None:
        path = self._state_path(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionMemoryState.from_dict(data)

    def put_state(self, session_id: str, state: SessionMemoryState) -> None:
        path = self._state_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def get_record(self, session_id: str, memory_id: str) -> MemoryRecord:
        path = self._record_path(session_id, memory_id)
        if not path.exists():
            raise MemoryRecordNotFoundError(session_id, memory_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return MemoryRecord.from_dict(data)

    def put_record(self, session_id: str, record: MemoryRecord) -> None:
        path = self._record_path(session_id, record.memory_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
