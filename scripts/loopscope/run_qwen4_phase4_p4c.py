#!/usr/bin/env python3
"""Operate the exact LoopScope Phase 4 P4-C sealed eight-cell acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from tflt.loopscope.phase4_outcome import (
    AUTHORIZED_RUN_ROOT,
    build_launch_manifest,
    freeze_full,
    prepare_smoke,
    record_submission,
    run_cell,
    seal_stage,
)


def _scheduler(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", required=True)
    parser.add_argument("--gres", required=True)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--array-throttle", required=True, type=int)
    parser.add_argument("--time-limit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4-C frozen full-decode acquisition")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="validate a launch manifest without writing")
    dry.add_argument("--mode", choices=("smoke", "full"), required=True)
    dry.add_argument("--expected-commit", required=True)
    _scheduler(dry)
    prepare = sub.add_parser("prepare-smoke", help="create the exact write-once root")
    prepare.add_argument("--card", required=True, type=Path)
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--expected-commit", required=True)
    _scheduler(prepare)
    full = sub.add_parser("freeze-full", help="freeze full launch after all smokes pass")
    full.add_argument("--run-root", required=True, type=Path)
    full.add_argument("--expected-commit", required=True)
    _scheduler(full)
    cell = sub.add_parser("run-cell", help="run one exact smoke/full producer cell")
    cell.add_argument("--mode", choices=("smoke", "full"), required=True)
    cell.add_argument("--run-root", required=True, type=Path)
    cell.add_argument("--cell-index", required=True, type=int)
    cell.add_argument("--expected-commit", required=True)
    cell.add_argument("--batch-size", required=True)
    submit = sub.add_parser("record-submit", help="write one submission receipt")
    submit.add_argument("--mode", choices=("smoke", "full"), required=True)
    submit.add_argument("--run-root", required=True, type=Path)
    submit.add_argument("--expected-commit", required=True)
    submit.add_argument("--job-id", required=True)
    seal = sub.add_parser("seal", help="close scheduler/stat/SHA evidence without parsing results")
    seal.add_argument("--mode", choices=("smoke", "full"), required=True)
    seal.add_argument("--run-root", required=True, type=Path)
    seal.add_argument("--expected-commit", required=True)
    seal.add_argument("--job-id", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = None
    if args.command == "dry-run":
        result = build_launch_manifest(
            mode=args.mode, run_root=AUTHORIZED_RUN_ROOT,
            expected_commit=args.expected_commit, partition=args.partition,
            gres=args.gres, qos=args.qos, account=args.account,
            array_throttle=args.array_throttle, time_limit=args.time_limit,
            created_at_utc="DRY_RUN_METADATA_ONLY",
        )
    elif args.command == "prepare-smoke":
        result = prepare_smoke(
            card_path=args.card, run_root=args.run_root,
            expected_commit=args.expected_commit, partition=args.partition,
            gres=args.gres, qos=args.qos, account=args.account,
            array_throttle=args.array_throttle, time_limit=args.time_limit,
        )
    elif args.command == "freeze-full":
        result = freeze_full(
            run_root=args.run_root, expected_commit=args.expected_commit,
            partition=args.partition, gres=args.gres, qos=args.qos,
            account=args.account, array_throttle=args.array_throttle,
            time_limit=args.time_limit,
        )
    elif args.command == "run-cell":
        run_cell(
            mode=args.mode, run_root=args.run_root, cell_index=args.cell_index,
            expected_commit=args.expected_commit, batch_size=args.batch_size,
        )
    elif args.command == "record-submit":
        result = record_submission(
            mode=args.mode, run_root=args.run_root,
            expected_commit=args.expected_commit, job_id=args.job_id,
        )
    elif args.command == "seal":
        result = seal_stage(
            mode=args.mode, run_root=args.run_root,
            expected_commit=args.expected_commit, job_id=args.job_id,
        )
    if result is not None:
        # CLI output is identity metadata only; producer commands print no result values.
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
