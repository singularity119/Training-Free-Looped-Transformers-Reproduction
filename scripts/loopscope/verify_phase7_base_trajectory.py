#!/usr/bin/env python3
"""Fresh-process verifier for sanitized Phase 7 trajectory records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tflt.loopscope.phase7_schema import (
    CATEGORY_COUNT,
    POPULATION_SIZE,
    Phase7ContractError,
    validate_sanitized_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Phase 7 scalar trajectory membership and schema closure.")
    parser.add_argument("--input", required=True, help="Sanitized trajectory JSONL.")
    parser.add_argument("--model-key", choices=("qwen25_3b", "llama32_3b", "gemma2_2b"), required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--expected-subjects", type=int, default=None)
    parser.add_argument("--receipt", default=None, help="Optional write-once JSON verification receipt.")
    return parser


def _load(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Phase7ContractError("invalid JSON at line %d" % line_number) from exc
            if not isinstance(value, dict):
                raise Phase7ContractError("trajectory line %d is not an object" % line_number)
            rows.append(value)
    if not rows:
        raise Phase7ContractError("trajectory JSONL is empty")
    return rows


def verify(args: argparse.Namespace) -> int:
    rows = _load(Path(args.input))
    identities = set()
    subjects = set()
    for row in rows:
        validate_sanitized_record(
            row,
            expected_model_key=args.model_key,
            expected_layer_count=args.layer_count,
        )
        identity = row["canonical_identity"]
        if identity in identities:
            raise Phase7ContractError("duplicate canonical identity")
        identities.add(identity)
        subjects.add(row["subject"])
    expected_count = args.expected_count
    if expected_count is not None and len(rows) != expected_count:
        raise Phase7ContractError("record count differs from expected")
    if args.expected_subjects is not None and len(subjects) != args.expected_subjects:
        raise Phase7ContractError("subject count differs from expected")
    receipt = {
        "status": "PASS",
        "model_key": args.model_key,
        "layer_count": args.layer_count,
        "record_count": len(rows),
        "unique_identity_count": len(identities),
        "subject_count": len(subjects),
        "forward_count": sum(row["forward_count"] for row in rows),
        "loop_insertions": sorted({row["loop_insertions"] for row in rows}),
        "information_barrier": "sanitized scalar schema accepted; forbidden fields absent",
    }
    if args.receipt:
        receipt_path = Path(args.receipt)
        if receipt_path.exists():
            raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % receipt_path)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv=None) -> int:
    return verify(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
