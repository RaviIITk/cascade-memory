"""cascade-memory: framework-agnostic cascading context/memory manager for LLM agent loops."""

from cascade_memory.config import MemoryConfig
from cascade_memory.exceptions import (
    CascadeMemoryError,
    MemoryRecordNotFoundError,
    SummarizerNotConfiguredError,
)
from cascade_memory.middleware import CascadingMemoryMiddleware
from cascade_memory.model_client import ModelClient
from cascade_memory.schema import MemoryRecord, ProgressState, SessionMemoryState, SummaryBlock
from cascade_memory.tools import LOAD_MEMORY_TOOL_SPEC, LoadMemoryHandler

__version__ = "0.1.0"

__all__ = [
    "CascadingMemoryMiddleware",
    "MemoryConfig",
    "ModelClient",
    "SummaryBlock",
    "MemoryRecord",
    "SessionMemoryState",
    "ProgressState",
    "LOAD_MEMORY_TOOL_SPEC",
    "LoadMemoryHandler",
    "CascadeMemoryError",
    "MemoryRecordNotFoundError",
    "SummarizerNotConfiguredError",
]
