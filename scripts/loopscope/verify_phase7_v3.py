#!/usr/bin/env python3
"""Fresh-process V3 verifier: reload records, recompute, and compare payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tflt.loopscope.phase7_schema import Phase7ContractError
from tflt.loopscope.phase7_v3 import analyze_records, validate_v3_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-process verifier for a Phase 7 V3 analysis.")
    parser.add_argument("--input", required=True, help="Sanitized trajectory JSONL.")
    parser.add_argument("--analysis", required=True, help="V3 analysis JSON to recompute and compare.")
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--receipt", default=None, help="Optional write-once JSON receipt.")
    return parser


def _load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Phase7ContractError("trajectory line is not an object")
                rows.append(value)
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase7ContractError("analysis JSON root is not an object")
    return value


def verify(args: argparse.Namespace) -> int:
    expected = _load_json(Path(args.analysis))
    validate_v3_payload(expected, expected_layer_count=args.layer_count)
    recomputed = analyze_records(
        _load_jsonl(Path(args.input)),
        layer_count=args.layer_count,
        replicates=args.replicates,
        seed=args.seed,
        formal=False,
    )
    if recomputed != expected:
        raise Phase7ContractError("BLOCK_V3_VERIFIER_MISMATCH: fresh recomputation differs")
    receipt = {
        "status": "PASS",
        "layer_count": args.layer_count,
        "replicates": args.replicates,
        "seed": args.seed,
        "selector_decision": expected["selector_decision"],
        "candidate_count": expected["candidate_count"],
        "independent_recomputation": True,
    }
    if args.receipt:
        path = Path(args.receipt)
        if path.exists():
            raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv=None) -> int:
    return verify(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
