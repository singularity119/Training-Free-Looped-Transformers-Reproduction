#!/usr/bin/env python3
"""Independent command-line verifier for Phase 6 Gate A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase6_verifier import run_self_test, verify_gate_a_contracts


DEFAULT_CARD = REPO_ROOT / "configs/loopscope/phase6_pre_answer_v2_card.json"
DEFAULT_TRAJECTORY_SCHEMA = (
    REPO_ROOT / "configs/loopscope/phase6_trajectory_schema.json"
)
DEFAULT_ELIGIBILITY_SCHEMA = (
    REPO_ROOT / "configs/loopscope/phase6_eligibility_schema.json"
)
DEFAULT_SELECTOR_SCHEMA = (
    REPO_ROOT / "configs/loopscope/phase6_selector_freeze_schema.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the local, outcome-blind Phase 6 Gate A contract layer."
    )
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument(
        "--trajectory-schema", type=Path, default=DEFAULT_TRAJECTORY_SCHEMA
    )
    parser.add_argument(
        "--eligibility-schema", type=Path, default=DEFAULT_ELIGIBILITY_SCHEMA
    )
    parser.add_argument("--selector-schema", type=Path, default=DEFAULT_SELECTOR_SCHEMA)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="also verify that critical invalid fixtures fail closed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verifier = run_self_test if args.self_test else verify_gate_a_contracts
    result = verifier(
        args.card,
        args.trajectory_schema,
        args.eligibility_schema,
        args.selector_schema,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
