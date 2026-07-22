#!/usr/bin/env python3
"""CLI for the authorized one-shot LoopScope Phase 5 Gate D analysis."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from tflt.loopscope.phase5_outcome_analysis import (
    git_provenance,
    run_one_shot,
    validate_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--expected-commit", required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--expected-commit", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        git = git_provenance(args.expected_commit, remote_required=True)
        value = validate_inputs(hash_results=True)
        result = {
            "gate": "D", "status": "PRE_UNSEAL_CLOSURE_PASS",
            "expected_commit": args.expected_commit,
            "git": git,
            "cell_count": len(value["completion"]["cells"]),
            "record_count_per_cell": value["completion"]["identity_closure"]["record_count_per_cell"],
            "window_decision": value["selector_freeze"]["window_decision"],
            "outcome_payload_parsed": False,
        }
    else:
        result = run_one_shot(expected_commit=args.expected_commit, argv=list(argv or ()))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
