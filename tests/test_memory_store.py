from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from cascade_memory.exceptions import MemoryRecordNotFoundError
from cascade_memory.memory_store.filesystem import FilesystemMemoryStore
from cascade_memory.memory_store.s3 import S3MemoryStore
from cascade_memory.schema import MemoryRecord, ProgressState, SessionMemoryState, SummaryBlock


def _sample_state(session_id: str) -> SessionMemoryState:
    return SessionMemoryState(
        session_id=session_id,
        summary_blocks=[
            SummaryBlock(
                memory_id="mem_1",
                session_id=session_id,
                covers_message_range=(0, 2),
                token_count_original=42,
                summary_text="summary",
                progress_state=ProgressState(task="do thing"),
            )
        ],
        preserved_messages=[{"role": "user", "content": "hi"}],
        next_message_index=3,
    )


def _sample_record(session_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id="mem_1",
        session_id=session_id,
        original_messages=[{"role": "user", "content": "evicted"}],
    )


@pytest.fixture
def fs_store(tmp_path: object) -> FilesystemMemoryStore:
    return FilesystemMemoryStore(root=str(tmp_path) + "/memory")


def test_filesystem_state_roundtrip(fs_store: FilesystemMemoryStore) -> None:
    state = _sample_state("sess-1")
    fs_store.put_state("sess-1", state)
    restored = fs_store.get_state("sess-1")
    assert restored == state


def test_filesystem_missing_state_returns_none(fs_store: FilesystemMemoryStore) -> None:
    assert fs_store.get_state("nope") is None


def test_filesystem_record_roundtrip(fs_store: FilesystemMemoryStore) -> None:
    record = _sample_record("sess-1")
    fs_store.put_record("sess-1", record)
    restored = fs_store.get_record("sess-1", "mem_1")
    assert restored == record


def test_filesystem_missing_record_raises(fs_store: FilesystemMemoryStore) -> None:
    with pytest.raises(MemoryRecordNotFoundError):
        fs_store.get_record("sess-1", "mem_missing")


@mock_aws
def test_s3_state_and_record_roundtrip() -> None:
    bucket = "test-bucket"
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
    store = S3MemoryStore(bucket=bucket, client=boto3.client("s3", region_name="us-east-1"))

    state = _sample_state("sess-1")
    store.put_state("sess-1", state)
    assert store.get_state("sess-1") == state

    record = _sample_record("sess-1")
    store.put_record("sess-1", record)
    assert store.get_record("sess-1", "mem_1") == record


@mock_aws
def test_s3_missing_state_returns_none() -> None:
    bucket = "test-bucket"
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
    store = S3MemoryStore(bucket=bucket, client=boto3.client("s3", region_name="us-east-1"))
    assert store.get_state("nope") is None


@mock_aws
def test_s3_missing_record_raises() -> None:
    bucket = "test-bucket"
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
    store = S3MemoryStore(bucket=bucket, client=boto3.client("s3", region_name="us-east-1"))
    with pytest.raises(MemoryRecordNotFoundError):
        store.get_record("sess-1", "mem_missing")
