#!/usr/bin/env python3
"""Build a read-only Phase 9 historical-cell to Phase 8-root map.

The input verification receipts are used only for their score-root paths and
cell metadata.  This command does not read labels, correctness, or aggregate
outcomes.  A map is write-once so an old receipt cannot be silently replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from tflt.loopscope.phase9_accuracy import load_config, logical_panel


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def target_cell_id(cell: Mapping[str, Any]) -> str | None:
    """Translate a frozen Phase 8 score cell into the Phase 9 name."""

    model_index = int(cell.get("model_index", -1))
    prefix = {0: "q4", 1: "q17"}.get(model_index)
    if prefix is None:
        return None
    arm = cell.get("arm")
    if arm == "Native":
        return f"{prefix}-native"
    window = cell.get("window")
    k = cell.get("k")
    if not isinstance(window, list) or len(window) != 2 or k not in (2, 3, 4):
        return None
    if arm == "Loop":
        suffix = "loop"
    elif arm == "Spectral":
        suffix = "shared-t0" if cell.get("fit_t") == 0 else "shared-t1"
    else:
        return None
    return f"{prefix}-w{window[0]}-{window[1]}-k{k}-{suffix}"


def build(receipts: Iterable[Path]) -> Dict[str, Any]:
    config = load_config()
    panel = {cell["cell_id"]: cell for cell in logical_panel(config)}
    historical = {
        cell_id
        for cell_id, cell in panel.items()
        if not cell["new_in_phase9"]
    }
    roots_by_target: Dict[str, set[str]] = {}
    source_ids_by_target: Dict[str, set[str]] = {}
    receipt_paths = []
    for receipt_path in receipts:
        receipt_path = Path(receipt_path)
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "SHARDS_CLOSED", f"historical receipt is not closed: {receipt_path}")
        receipt_paths.append(str(receipt_path))
        for root_text in receipt.get("cell_roots", []):
            root = Path(root_text)
            metadata = read_json(root / "command_args.json")
            cell = metadata.get("cell")
            require(isinstance(cell, Mapping), f"historical root has no cell metadata: {root}")
            target = target_cell_id(cell)
            if target not in historical:
                continue
            roots_by_target.setdefault(target, set()).add(str(root))
            source_ids_by_target.setdefault(target, set()).add(str(cell["cell_id"]))

    require(set(roots_by_target) == historical,
            "historical receipts do not cover exactly the 29 Phase 9 historical cells")
    entries = []
    for cell_id in sorted(historical):
        roots = sorted(roots_by_target[cell_id])
        require(roots, f"historical cell has no score roots: {cell_id}")
        entries.append({
            "cell_id": cell_id,
            "source_cell_ids": sorted(source_ids_by_target[cell_id]),
            "roots": roots,
        })
    return {
        "schema": "loopscope.phase9.history_map.v1",
        "target_gold_loaded": False,
        "source_verifications": sorted(set(receipt_paths)),
        "cell_count": len(entries),
        "cells": entries,
        "closure": "read-only Phase 8 raw-score roots; no labels, correctness, or outcomes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.verification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PHASE9_HISTORY_MAP_READY",
        "cell_count": value["cell_count"],
        "target_gold_loaded": value["target_gold_loaded"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
