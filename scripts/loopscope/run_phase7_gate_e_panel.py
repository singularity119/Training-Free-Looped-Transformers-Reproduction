#!/usr/bin/env python3
"""Launch only the frozen LoopScope Phase 7 Gate E outcome panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase7_outcome import (  # noqa: E402
    Phase7OutcomeError,
    analyze_once,
    build_launch,
    prepare_run,
    resource_summary,
    run_cell,
    run_pool,
    submit_run,
)


def _indices(text: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("indices must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7 Gate E frozen outcome panel launcher.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Freeze one fresh static run root before model forwards.")
    prepare.add_argument("--mode", choices=("debug", "formal"), required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--expected-commit", required=True)
    prepare.add_argument("--cache-dir", default=None)
    prepare.add_argument("--formal-identity-manifest", default=None)
    prepare.add_argument("--debug-indices", type=_indices, default=None)
    prepare.add_argument("--card", default=None)

    launch = sub.add_parser("build-launch", help="Freeze a bounded Slurm worker-pool launch plan.")
    launch.add_argument("--run-root", required=True)
    launch.add_argument("--parent-count", type=int, required=True)
    launch.add_argument("--concurrency", type=int, required=True)
    launch.add_argument("--partition", required=True)
    launch.add_argument("--gpu-type", required=True)
    launch.add_argument("--time", dest="time_limit", required=True)
    launch.add_argument("--cpus", dest="cpus_per_task", type=int, required=True)
    launch.add_argument("--mem", dest="memory", required=True)
    launch.add_argument("--child-cpu-threads", type=int, required=True)
    launch.add_argument("--qos", default=None)

    submit = sub.add_parser("submit", help="Submit a frozen Slurm worker-pool launcher once.")
    submit.add_argument("--run-root", required=True)

    cell = sub.add_parser("run-cell", help="Run exactly one frozen cell; used only by worker pools.")
    cell.add_argument("--run-root", required=True)
    cell.add_argument("--cell-index", type=int, required=True)

    pool = sub.add_parser("run-pool", help="Run one bounded same-GPU worker pool.")
    pool.add_argument("--run-root", required=True)
    pool.add_argument("--pool-index", type=int, required=True)

    summary = sub.add_parser("resource-summary", help="Report resource-only packing evidence.")
    summary.add_argument("--run-root", required=True)

    analyze = sub.add_parser("analyze", help="Run the one authorized combined analysis after completeness closure.")
    analyze.add_argument("--run-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_run(
                mode=args.mode,
                run_root=Path(args.run_root),
                expected_commit=args.expected_commit,
                cache_dir=args.cache_dir,
                formal_identity_manifest=(Path(args.formal_identity_manifest) if args.formal_identity_manifest else None),
                debug_indices=args.debug_indices,
                card_path=(Path(args.card) if args.card else None),
            )
        elif args.command == "build-launch":
            result = build_launch(
                run_root=Path(args.run_root),
                parent_count=args.parent_count,
                concurrency=args.concurrency,
                partition=args.partition,
                gpu_type=args.gpu_type,
                time_limit=args.time_limit,
                cpus_per_task=args.cpus_per_task,
                memory=args.memory,
                child_cpu_threads=args.child_cpu_threads,
                qos=args.qos,
            )
        elif args.command == "submit":
            result = submit_run(Path(args.run_root))
        elif args.command == "run-cell":
            result = run_cell(run_root=Path(args.run_root), cell_index=args.cell_index)
        elif args.command == "run-pool":
            result = run_pool(run_root=Path(args.run_root), pool_index=args.pool_index)
        elif args.command == "resource-summary":
            result = resource_summary(Path(args.run_root))
        elif args.command == "analyze":
            result = analyze_once(Path(args.run_root))
        else:  # pragma: no cover - argparse makes this unreachable.
            raise Phase7OutcomeError("unsupported Gate E command")
    except Phase7OutcomeError as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
