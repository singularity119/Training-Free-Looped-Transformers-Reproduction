#!/usr/bin/env python3
"""Operate the exact LoopScope Phase 5 Gate C sealed eight-cell panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from tflt.loopscope.phase5_outcome import (
    build_launch_manifest,
    freeze_formal,
    prepare_smoke,
    record_submission,
    run_cell,
    seal_stage,
)


def _scheduler(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", required=True)
    parser.add_argument("--gres", required=True)
    parser.add_argument("--qos")
    parser.add_argument("--account")
    parser.add_argument("--array-throttle", required=True, type=int)
    parser.add_argument("--time-limit", required=True)
    parser.add_argument("--batch-size", required=True)


def _root_commit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--mode", choices=("smoke", "formal"), required=True)
    _root_commit(dry)
    _scheduler(dry)
    prepare = sub.add_parser("prepare-smoke")
    _root_commit(prepare)
    _scheduler(prepare)
    formal = sub.add_parser("freeze-formal")
    _root_commit(formal)
    _scheduler(formal)
    cell = sub.add_parser("run-cell")
    cell.add_argument("--mode", choices=("smoke", "formal"), required=True)
    _root_commit(cell)
    cell.add_argument("--cell-index", required=True, type=int)
    submit = sub.add_parser("record-submission")
    submit.add_argument("--mode", choices=("smoke", "formal"), required=True)
    submit.add_argument("--run-root", required=True, type=Path)
    submit.add_argument("--job-id", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--mode", choices=("smoke", "formal"), required=True)
    _root_commit(seal)
    seal.add_argument("--job-id", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dry-run":
        result = build_launch_manifest(
            mode=args.mode, run_root=args.run_root, expected_commit=args.expected_commit,
            partition=args.partition, gres=args.gres, qos=args.qos, account=args.account,
            array_throttle=args.array_throttle, time_limit=args.time_limit,
            batch_size=args.batch_size, created_at_utc="2026-07-22T00:00:00Z",
        )
    elif args.command == "prepare-smoke":
        result = prepare_smoke(
            run_root=args.run_root, expected_commit=args.expected_commit,
            partition=args.partition, gres=args.gres, qos=args.qos, account=args.account,
            array_throttle=args.array_throttle, time_limit=args.time_limit,
            batch_size=args.batch_size,
        )
    elif args.command == "freeze-formal":
        result = freeze_formal(
            run_root=args.run_root, expected_commit=args.expected_commit,
            partition=args.partition, gres=args.gres, qos=args.qos, account=args.account,
            array_throttle=args.array_throttle, time_limit=args.time_limit,
            batch_size=args.batch_size,
        )
    elif args.command == "run-cell":
        run_cell(
            mode=args.mode, run_root=args.run_root, cell_index=args.cell_index,
            expected_commit=args.expected_commit,
        )
        result = {"mode": args.mode, "cell_index": args.cell_index, "status": "PRODUCER_COMPLETED"}
    elif args.command == "record-submission":
        result = record_submission(mode=args.mode, run_root=args.run_root, job_id=args.job_id)
    else:
        result = seal_stage(
            mode=args.mode, run_root=args.run_root, expected_commit=args.expected_commit,
            job_id=args.job_id,
        )
    # CLI output is contract/identity metadata only. Never print evaluator payloads.
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
