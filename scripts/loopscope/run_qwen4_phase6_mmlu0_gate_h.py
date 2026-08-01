#!/usr/bin/env python3
"""Gate H local verifier/dry-run for the MMLU 0-shot prefix contract.

All actions are local and outcome-blind.  This script never loads a model or
dataset, performs a forward/generation call, reads the test split, or runs the
formal selector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase6_mmlu0_verifier import (  # noqa: E402
    dry_run_payload,
    run_self_test,
    verify_gate_h_contracts,
)


DEFAULT_CARD = REPO_ROOT / "configs/loopscope/phase6_mmlu0_prefix_card.json"
DEFAULT_SCHEMA = REPO_ROOT / "configs/loopscope/phase6_mmlu0_trajectory_schema.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the local outcome-blind Phase 6 MMLU 0-shot Gate H contract; "
            "no model/data forward, formal selector, test split, or outcome read."
        )
    )
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true", help="validate and print the no-execution plan")
    actions.add_argument("--self-test", action="store_true", help="run synthetic metrics and negative contract tests")
    actions.add_argument("--verify", action="store_true", help="verify the card and trajectory schema only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        result = dry_run_payload(args.card, args.schema)
    elif args.self_test:
        result = run_self_test(args.card, args.schema)
    else:
        result = verify_gate_h_contracts(args.card, args.schema)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
