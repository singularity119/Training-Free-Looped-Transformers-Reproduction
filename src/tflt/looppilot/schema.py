"""Versioned, finite-only LoopPilot records and write-once artifact helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DocKey:
    task: str
    task_version: str
    renderer_hash: str
    doc_hash: str
    occurrence_id: str

    def __post_init__(self) -> None:
        for name in ("task", "task_version", "occurrence_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s must be non-empty" % name)
        for name in ("renderer_hash", "doc_hash"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError("%s must be a 64-character SHA-256 hex digest" % name)

    def canonical(self) -> str:
        return "|".join(
            (self.task, self.task_version, self.renderer_hash, self.doc_hash, self.occurrence_id)
        )


@dataclass(frozen=True)
class SignalRecord:
    key: DocKey
    row_id: str
    choice_index: int
    question_token_count: int
    r0_median: float
    r0_p90: float
    r0_max: float
    c0: float
    n0_median: float
    n0_p90: float
    n0_max: float
    q1: Optional[float] = None
    residual_cosine: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.row_id.strip():
            raise ValueError("row_id must be non-empty")
        if self.choice_index < 0:
            raise ValueError("choice_index must be non-negative")
        if self.question_token_count < 1:
            raise ValueError("question_token_count must be positive")
        for name in (
            "r0_median",
            "r0_p90",
            "r0_max",
            "c0",
            "n0_median",
            "n0_p90",
            "n0_max",
            "q1",
            "residual_cosine",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("%s must be finite" % name)


def write_json_once(path: Path, payload: Any) -> None:
    serialized = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text_exclusive(path, serialized)


def write_jsonl_once(path: Path, rows: Iterable[Any]) -> None:
    lines = [json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) for row in rows]
    serialized = "\n".join(lines) + ("\n" if lines else "")
    _write_text_exclusive(path, serialized)


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON artifacts may not contain NaN or Inf")
    return value
