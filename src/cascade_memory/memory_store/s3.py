"""S3 implementation of `MemoryStore`, mirroring the filesystem store's key layout."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError

from cascade_memory.exceptions import MemoryRecordNotFoundError
from cascade_memory.memory_store.base import MemoryStore
from cascade_memory.schema import MemoryRecord, SessionMemoryState

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


class S3MemoryStore(MemoryStore):
    """Lays out `s3://{bucket}/{prefix}{session_id}/state.json` and
    `.../records/{memory_id}.json`."""

    def __init__(self, bucket: str, prefix: str = "", client: S3Client | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client: S3Client = client if client is not None else boto3.client("s3")

    def _key_prefix(self, session_id: str) -> str:
        return f"{self.prefix}/{session_id}" if self.prefix else session_id

    def _state_key(self, session_id: str) -> str:
        return f"{self._key_prefix(session_id)}/state.json"

    def _record_key(self, session_id: str, memory_id: str) -> str:
        return f"{self._key_prefix(session_id)}/records/{memory_id}.json"

    def _get_json(self, key: str) -> dict[str, Any] | None:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        body = response["Body"].read()
        result: dict[str, Any] = json.loads(body)
        return result

    def _put_json(self, key: str, data: dict[str, Any]) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def get_state(self, session_id: str) -> SessionMemoryState | None:
        data = self._get_json(self._state_key(session_id))
        if data is None:
            return None
        return SessionMemoryState.from_dict(data)

    def put_state(self, session_id: str, state: SessionMemoryState) -> None:
        self._put_json(self._state_key(session_id), state.to_dict())

    def get_record(self, session_id: str, memory_id: str) -> MemoryRecord:
        data = self._get_json(self._record_key(session_id, memory_id))
        if data is None:
            raise MemoryRecordNotFoundError(session_id, memory_id)
        return MemoryRecord.from_dict(data)

    def put_record(self, session_id: str, record: MemoryRecord) -> None:
        self._put_json(self._record_key(session_id, record.memory_id), record.to_dict())
