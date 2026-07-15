#!/usr/bin/env python3
"""Execute the exact write-once LoopScope Phase 3 Gate P3-B workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    """Bypass the convenience package initializer for the no-loop producer.

    ``tflt.__init__`` exports the legacy loop wrapper and therefore imports the
    wrapper, cache, and strategy modules as a package side effect.  P3-B is a
    native-forward producer, so its executable entry point installs a minimal
    namespace package rooted at ``src/tflt`` before importing any project
    module.  Normal library/test imports remain unchanged.
    """

    if "tflt" in sys.modules:
        raise RuntimeError("P3-B producer requires a fresh isolated Python process")
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__file__ = str(package_root)
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase3_acquisition import (
    _assert_no_loop_modules_loaded,
    acquire_shard,
    acquire_smoke,
    freeze_shards,
    freeze_smoke,
    materialize_b0_b1,
    merge_primary,
    record_monitor_lifecycle,
    verify_primary,
    write_resource_accounting,
    write_slurm_launchers,
)

_assert_no_loop_modules_loaded()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind P3-B acquisition only: B0/B1 source closure, B2 native "
            "no-loop smoke, B3 validation-1531 native no-loop shards, merge, and verify."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b0 = sub.add_parser("b0-b1", help="Materialize exact gold-free source and pool")
    _common(b0)
    b0.add_argument("--projection", required=True, type=Path)
    b0.add_argument("--renderer-manifest", required=True, type=Path)

    smoke_freeze = sub.add_parser(
        "freeze-smoke", help="Freeze the first four canonical smoke identities"
    )
    _common(smoke_freeze)

    smoke = sub.add_parser(
        "acquire-smoke", help="Run one native no-loop forward for four smoke identities"
    )
    _common(smoke)

    shards = sub.add_parser(
        "freeze-shards", help="Freeze an exact disjoint validation-1531 shard manifest"
    )
    _common(shards)
    shards.add_argument("--shard-count", required=True, type=int)

    launchers = sub.add_parser(
        "write-launchers", help="Write one exact offline Slurm runner script"
    )
    _common(launchers)
    launchers.add_argument("--stage", required=True, choices=("smoke", "primary"))
    launchers.add_argument("--partition", required=True)
    launchers.add_argument("--qos", default=None)
    launchers.add_argument("--account", default=None)
    launchers.add_argument("--time-limit", required=True)
    launchers.add_argument("--cpus-per-task", required=True, type=int)
    launchers.add_argument("--memory", required=True)
    launchers.add_argument("--gres", required=True)
    launchers.add_argument("--array-throttle", type=int, default=None)

    monitor = sub.add_parser(
        "record-monitor", help="Seal one deleted executor-owned heartbeat lifecycle"
    )
    _common(monitor)
    monitor.add_argument("--stage", required=True, choices=("smoke", "primary"))
    monitor.add_argument("--job-id", required=True)
    monitor.add_argument("--automation-id", required=True)
    monitor.add_argument("--automation-name", required=True)
    monitor.add_argument("--cadence-minutes", required=True, type=int)
    monitor.add_argument("--created-at-utc", required=True)
    monitor.add_argument("--terminal-wakeup-at-utc", required=True)
    monitor.add_argument("--disabled-at-utc", required=True)
    monitor.add_argument("--terminal-state", required=True)
    monitor.add_argument("--disable-evidence", required=True)

    acquire = sub.add_parser(
        "acquire-shard", help="Acquire one frozen primary no-loop shard"
    )
    _common(acquire)
    acquire.add_argument("--shard-id", type=int, default=None)

    merge = sub.add_parser("merge", help="Merge all completed primary shards")
    _common(merge)

    verify = sub.add_parser(
        "verify", help="Independently recompute full source/pool/shard/trajectory closure"
    )
    _common(verify)

    resources = sub.add_parser(
        "resource-receipt", help="Seal terminal Slurm and GPU accounting evidence"
    )
    _common(resources)
    resources.add_argument("--smoke-job-id", required=True)
    resources.add_argument("--primary-job-id", required=True)
    resources.add_argument("--initial-throttle", required=True, type=int)
    return parser


def _argv() -> Sequence[str]:
    return [str(Path(__file__).resolve())] + list(sys.argv[1:])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provenance_argv = _argv() if argv is None else [str(Path(__file__).resolve())] + list(argv)
    common = {
        "card_path": args.card,
        "run_root": args.run_root,
        "expected_commit": args.expected_commit,
    }
    if args.command == "b0-b1":
        result = materialize_b0_b1(
            projection_path=args.projection,
            renderer_manifest_path=args.renderer_manifest,
            argv=provenance_argv,
            **common,
        )
    elif args.command == "freeze-smoke":
        result = freeze_smoke(argv=provenance_argv, **common)
    elif args.command == "acquire-smoke":
        result = acquire_smoke(argv=provenance_argv, **common)
    elif args.command == "freeze-shards":
        result = freeze_shards(
            shard_count=args.shard_count, argv=provenance_argv, **common
        )
    elif args.command == "write-launchers":
        result = write_slurm_launchers(
            stage=args.stage,
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            gres=args.gres,
            array_throttle=args.array_throttle,
            **common,
        )
    elif args.command == "record-monitor":
        result = record_monitor_lifecycle(
            stage=args.stage,
            job_id=args.job_id,
            automation_id=args.automation_id,
            automation_name=args.automation_name,
            cadence_minutes=args.cadence_minutes,
            created_at_utc=args.created_at_utc,
            terminal_wakeup_at_utc=args.terminal_wakeup_at_utc,
            disabled_at_utc=args.disabled_at_utc,
            terminal_state=args.terminal_state,
            disable_evidence=args.disable_evidence,
            argv=provenance_argv,
            **common,
        )
    elif args.command == "acquire-shard":
        shard_id = args.shard_id
        if shard_id is None:
            raw = os.environ.get("SLURM_ARRAY_TASK_ID")
            if raw is None:
                raise SystemExit("--shard-id or SLURM_ARRAY_TASK_ID is required")
            shard_id = int(raw)
        result = acquire_shard(
            shard_id=shard_id, argv=provenance_argv, **common
        )
    elif args.command == "merge":
        result = merge_primary(argv=provenance_argv, **common)
    elif args.command == "verify":
        result = verify_primary(argv=provenance_argv, **common)
    elif args.command == "resource-receipt":
        result = write_resource_accounting(
            smoke_job_id=args.smoke_job_id,
            primary_job_id=args.primary_job_id,
            initial_throttle=args.initial_throttle,
            argv=provenance_argv,
            **common,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
