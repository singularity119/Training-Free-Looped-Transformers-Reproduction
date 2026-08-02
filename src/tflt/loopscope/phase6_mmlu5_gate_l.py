"""Gate L import-only boundary prepared by Gate K.

This module intentionally contains no model, CUDA, dataset, or scheduler
imports.  Gate L's executor may extend this path after a planning `PASS`; Gate
K only proves that the future smoke entrypoint can import the frozen card and
trajectory schema without executing a workload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tflt.loopscope.phase6_mmlu5_schema import (
    CARD_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    file_sha256,
    load_json,
    validate_card,
    validate_schema_document,
)


GATE = "L"
SMOKE_PARTITION = "debug"
SMOKE_RECORD_COUNT = 4


def import_only_dry_plan(card_path: Path, trajectory_schema_path: Path) -> Dict[str, Any]:
    card = load_json(card_path)
    schema = load_json(trajectory_schema_path)
    validate_card(card)
    validate_schema_document(schema)
    return {
        "status": "IMPORT_ONLY_DRY_PLAN",
        "gate": GATE,
        "card_schema_version": CARD_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "card_sha256": file_sha256(card_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "partition": SMOKE_PARTITION,
        "records": SMOKE_RECORD_COUNT,
        "native_forward_per_record": 1,
        "model_weights_loaded": False,
        "forward_executed": False,
        "generation_executed": False,
        "loop_executed": False,
        "selector_executed": False,
        "outcome_read": False,
    }


__all__ = ["GATE", "SMOKE_PARTITION", "SMOKE_RECORD_COUNT", "import_only_dry_plan"]
