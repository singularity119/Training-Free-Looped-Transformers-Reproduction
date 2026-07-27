#!/usr/bin/env python3
"""Execute the exact write-once LoopScope Phase 5 Gate I workflow."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("Gate I requires a fresh isolated Python process")
    stage_root = Path(__file__).resolve().parents[2]
    source_root = Path(
        "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/src/tflt"
    )
    local_root = stage_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(local_root), str(source_root)]  # type: ignore[attr-defined]
    package.__file__ = str(local_root)
    sys.modules["tflt"] = package
    loopscope = types.ModuleType("tflt.loopscope")
    loopscope.__package__ = "tflt.loopscope"
    loopscope.__path__ = [  # type: ignore[attr-defined]
        str(local_root / "loopscope"),
        str(source_root / "loopscope"),
    ]
    loopscope.__file__ = str(local_root / "loopscope")
    sys.modules["tflt.loopscope"] = loopscope


_install_isolated_tflt_namespace()

from tflt.loopscope.phase5_angular_trajectory import (
    _assert_no_loop_modules_loaded,
    acquire,
    analyze_data,
    dry_run,
    finalize,
    freeze_membership,
    initialize_run_root,
    plot_figures,
    record_scheduler,
    seal_formal,
    seal_smoke,
    verify_data,
    verify_figure,
    write_launchers,
)

_assert_no_loop_modules_loaded()


def _card(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)


def _root(parser: argparse.ArgumentParser) -> None:
    _card(parser)
    parser.add_argument("--run-root", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind Gate I only: exact raw FinalNorm-input adjacent angular "
            "distance for Qwen3-1.7B/4B validation-1531."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("dry-run")
    _card(command)

    command = sub.add_parser("admission")
    _root(command)
    command.add_argument("--expected-commit", required=True)

    command = sub.add_parser("freeze-smoke")
    _root(command)
    command.add_argument("--model-key", required=True, choices=("qwen3_1p7b", "qwen3_4b"))

    command = sub.add_parser("acquire")
    _root(command)
    command.add_argument("--model-key", required=True, choices=("qwen3_1p7b", "qwen3_4b"))
    command.add_argument("--mode", required=True, choices=("smoke", "formal"))
    command.add_argument("--shard-id", type=int, default=None)
    command.add_argument("--smoke-attempt", type=int, default=1)

    command = sub.add_parser("seal-smoke")
    _root(command)
    command.add_argument("--model-key", required=True, choices=("qwen3_1p7b", "qwen3_4b"))
    command.add_argument("--attempt", type=int, default=1)

    command = sub.add_parser("freeze-formal")
    _root(command)
    command.add_argument("--model-key", required=True, choices=("qwen3_1p7b", "qwen3_4b"))
    command.add_argument("--shard-count", type=int, default=4)

    command = sub.add_parser("seal-formal")
    _root(command)
    command.add_argument("--model-key", required=True, choices=("qwen3_1p7b", "qwen3_4b"))

    command = sub.add_parser("write-launchers")
    _root(command)
    command.add_argument("--stage-root", required=True, type=Path)
    command.add_argument("--implementation-commit", required=True)
    command.add_argument("--partition", required=True)

    command = sub.add_parser("record-scheduler")
    command.add_argument("--run-root", required=True, type=Path)
    command.add_argument("--model-key", required=True, choices=("qwen3_1p7b", "qwen3_4b"))
    command.add_argument("--stage", required=True, choices=("smoke", "formal"))
    command.add_argument("--job-id", required=True)
    command.add_argument("--state", default=None)
    command.add_argument("--resources", default=None)

    command = sub.add_parser("analyze-data")
    _root(command)

    command = sub.add_parser("verify-data")
    _root(command)

    command = sub.add_parser("plot")
    _root(command)

    command = sub.add_parser("verify-figure")
    _root(command)

    command = sub.add_parser("finalize")
    _root(command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dry-run":
        result = dry_run(args.card)
    elif args.command == "admission":
        result = initialize_run_root(args.card, args.run_root, args.expected_commit)
    elif args.command == "freeze-smoke":
        result = freeze_membership(args.card, args.run_root, args.model_key, "smoke")
    elif args.command == "acquire":
        result = acquire(
            args.card,
            args.run_root,
            args.model_key,
            args.mode,
            shard_id=args.shard_id,
            smoke_attempt=args.smoke_attempt,
        )
    elif args.command == "seal-smoke":
        result = seal_smoke(args.card, args.run_root, args.model_key, args.attempt)
    elif args.command == "freeze-formal":
        result = freeze_membership(
            args.card,
            args.run_root,
            args.model_key,
            "formal",
            shard_count=args.shard_count,
        )
    elif args.command == "seal-formal":
        result = seal_formal(args.card, args.run_root, args.model_key)
    elif args.command == "write-launchers":
        result = write_launchers(
            args.card,
            args.run_root,
            args.stage_root,
            args.implementation_commit,
            args.partition,
        )
    elif args.command == "record-scheduler":
        result = record_scheduler(
            args.run_root,
            args.model_key,
            args.stage,
            args.job_id,
            state=args.state,
            resources=args.resources,
        )
    elif args.command == "analyze-data":
        result = analyze_data(args.card, args.run_root)
    elif args.command == "verify-data":
        result = verify_data(args.card, args.run_root)
    elif args.command == "plot":
        result = plot_figures(args.card, args.run_root)
    elif args.command == "verify-figure":
        result = verify_figure(args.card, args.run_root)
    elif args.command == "finalize":
        result = finalize(args.card, args.run_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
