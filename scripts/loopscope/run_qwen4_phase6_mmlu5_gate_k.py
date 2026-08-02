#!/usr/bin/env python3
"""Run Gate K's local checks or outcome-safe CPU prompt projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase6_mmlu5_gate_k import (  # noqa: E402
    build_cpu_projection,
    dry_run,
    run_self_test,
    verify_cpu_projection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate/close the LoopScope Phase 6 MMLU 5-shot Gate K contract; "
            "no model forward, GPU, Slurm, selector, or outcome read."
        )
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true", help="validate the contract and Gate L import-only plan")
    actions.add_argument("--self-test", action="store_true", help="run pure contract negative tests")
    actions.add_argument("--cpu-project", action="store_true", help="render the fresh validation-1531 CPU projection")
    actions.add_argument("--verify-projection", action="store_true", help="verify a completed CPU projection root")
    parser.add_argument("--run-root", type=Path, help="fresh or completed Gate K CPU run root")
    parser.add_argument("--expected-commit", help="exact clean loopscope commit for remote provenance")
    parser.add_argument("--seed", type=int, default=20260801, help="frozen dev demonstration sampling seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        result = dry_run()
    elif args.self_test:
        result = run_self_test()
    elif args.cpu_project:
        if args.run_root is None or not args.expected_commit:
            raise SystemExit("--cpu-project requires --run-root and --expected-commit")
        result = build_cpu_projection(run_root=args.run_root, expected_commit=args.expected_commit, seed=args.seed)
    else:
        if args.run_root is None or not args.expected_commit:
            raise SystemExit("--verify-projection requires --run-root and --expected-commit")
        result = verify_cpu_projection(run_root=args.run_root, expected_commit=args.expected_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
