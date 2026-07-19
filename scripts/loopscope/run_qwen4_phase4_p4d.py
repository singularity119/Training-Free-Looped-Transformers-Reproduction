#!/usr/bin/env python3
"""CLI for the exact one-shot LoopScope Phase 4 P4-D analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tflt.loopscope.phase4_outcome_analysis import (
    AUTHORIZED_OUTPUT_ROOT,
    run_analysis,
    run_verifier,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="unseal and analyze the frozen eight cells once")
    analyze.add_argument("--output-root", type=Path, default=AUTHORIZED_OUTPUT_ROOT)
    analyze.add_argument("--expected-commit", required=True)
    verify = subparsers.add_parser("verify", help="verify compact aggregate artifacts and close receipts")
    verify.add_argument("--output-root", type=Path, default=AUTHORIZED_OUTPUT_ROOT)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--repair-count", type=int, default=0)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    invocation = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    if arguments.command == "analyze":
        run_analysis(
            output_root=arguments.output_root,
            expected_commit=arguments.expected_commit,
            argv=invocation,
        )
    else:
        run_verifier(
            output_root=arguments.output_root,
            expected_commit=arguments.expected_commit,
            argv=invocation,
            repair_count=arguments.repair_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
