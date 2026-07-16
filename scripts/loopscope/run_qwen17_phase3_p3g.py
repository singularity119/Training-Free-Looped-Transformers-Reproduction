#!/usr/bin/env python3
"""Run the exact write-once LoopScope Phase 3 Gate P3-G stages."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("P3-G requires a fresh isolated Python process")
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__file__ = str(package_root)
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase3_p3g import (  # noqa: E402
    AUTHORIZED_RUN_ROOT,
    analyze_full,
    build_manifest,
    prepare_run,
    verify_full,
    verify_smoke,
)


def _roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--phase3-card", required=True, type=Path)
    parser.add_argument("--p3c-root", required=True, type=Path)
    parser.add_argument("--p3g-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def _scheduler(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", required=True)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--array-throttle", required=True, type=int)
    parser.add_argument("--smoke-time-limit", required=True)
    parser.add_argument("--full-time-limit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P3-G exact four-cell structural smoke, sealed targeted full, "
            "one analysis, and one verifier"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="Build and validate metadata only; write nothing")
    dry.add_argument("--expected-commit", required=True)
    _scheduler(dry)

    prepare = sub.add_parser(
        "prepare", help="Create the exact write-once root, manifest, and launchers"
    )
    _roots(prepare)
    _scheduler(prepare)

    smoke = sub.add_parser(
        "verify-smoke",
        help="Inspect only terminal state, recipe, structure, and loop effect",
    )
    _roots(smoke)
    smoke.add_argument("--job-id", required=True)

    analyze = sub.add_parser(
        "analyze",
        help="Require exact 4/4 full seal, then run the one authorized analysis",
    )
    _roots(analyze)
    analyze.add_argument("--job-id", required=True)

    verify = sub.add_parser(
        "verify", help="Independently reload exact sources and write the one verifier"
    )
    _roots(verify)
    verify.add_argument("--job-id", required=True)
    return parser


def _provenance_argv(argv: Optional[Sequence[str]]) -> Sequence[str]:
    suffix = list(sys.argv[1:] if argv is None else argv)
    return [str(Path(__file__).resolve())] + suffix


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provenance_argv = _provenance_argv(argv)
    if args.command == "dry-run":
        result = build_manifest(
            run_root=AUTHORIZED_RUN_ROOT,
            expected_commit=args.expected_commit,
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            smoke_time_limit=args.smoke_time_limit,
            full_time_limit=args.full_time_limit,
            created_at_utc="DRY_RUN_METADATA_ONLY",
        )
    elif args.command == "prepare":
        result = prepare_run(
            targeted_card_path=args.card,
            phase3_card_path=args.phase3_card,
            p3c_root=args.p3c_root,
            p3g_root=args.p3g_root,
            expected_commit=args.expected_commit,
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            smoke_time_limit=args.smoke_time_limit,
            full_time_limit=args.full_time_limit,
            argv=provenance_argv,
        )
    elif args.command == "verify-smoke":
        result = verify_smoke(
            targeted_card_path=args.card,
            phase3_card_path=args.phase3_card,
            p3c_root=args.p3c_root,
            p3g_root=args.p3g_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
        )
    elif args.command == "analyze":
        result = analyze_full(
            targeted_card_path=args.card,
            phase3_card_path=args.phase3_card,
            p3c_root=args.p3c_root,
            p3g_root=args.p3g_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            argv=provenance_argv,
        )
    elif args.command == "verify":
        result = verify_full(
            targeted_card_path=args.card,
            phase3_card_path=args.phase3_card,
            p3c_root=args.p3c_root,
            p3g_root=args.p3g_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            argv=provenance_argv,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
