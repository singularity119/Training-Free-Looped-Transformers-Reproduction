#!/usr/bin/env python3
"""Run the model-local Phase 7 V3.1 analyzer on sanitized scalar JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase7_v3 import analyze_records
from tflt.loopscope.phase7_schema import (
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_BOOTSTRAP_SEED,
    Phase7ContractError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sanitized Phase 7 H/D trajectories with V3.1.")
    parser.add_argument("--input", required=True, help="Sanitized trajectory JSONL.")
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--output", required=True, help="Write-once V3 analysis JSON.")
    parser.add_argument("--replicates", type=int, default=FORMAL_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=FORMAL_BOOTSTRAP_SEED)
    parser.add_argument("--formal", action="store_true", help="Enforce Gate D's 2,000-replicate/20260801 contract.")
    return parser


def _load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Phase7ContractError("trajectory line is not an object")
                rows.append(value)
    return rows


def run(args: argparse.Namespace) -> int:
    payload = analyze_records(
        _load(Path(args.input)),
        layer_count=args.layer_count,
        replicates=args.replicates,
        seed=args.seed,
        formal=args.formal,
    )
    output = Path(args.output)
    if output.exists():
        raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "selector_decision": payload["selector_decision"], "output": str(output)}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
