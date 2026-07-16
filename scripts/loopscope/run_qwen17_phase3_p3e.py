#!/usr/bin/env python3
"""Run the exact write-once LoopScope Phase 3 Gate P3-E stages."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("P3-E requires a fresh isolated Python process")
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__file__ = str(package_root)
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase3_p3e import (  # noqa: E402
    AUTHORIZED_RUN_ROOT,
    build_full_manifest,
    prepare_run,
    record_submission,
    seal_completed_sources,
    unseal_analyze_verify,
)


def _roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--p3c-root", required=True, type=Path)
    parser.add_argument("--p3d-root", required=True, type=Path)
    parser.add_argument("--p3e-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def _scheduler(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", required=True)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--array-throttle", required=True, type=int)
    parser.add_argument("--time-limit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3-E exact missing-12 full, sealed admission, and one-shot all-25 analysis"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="Build and validate metadata only; write nothing")
    dry.add_argument("--expected-commit", required=True)
    _scheduler(dry)
    prepare = sub.add_parser("prepare", help="Create the exact write-once run root and launchers")
    _roots(prepare)
    _scheduler(prepare)
    submit = sub.add_parser("record-submit", help="Write the one allowed submission receipt")
    submit.add_argument("--p3e-root", required=True, type=Path)
    submit.add_argument("--expected-commit", required=True)
    submit.add_argument("--job-id", required=True)
    seal = sub.add_parser("seal-check", help="Admit unseal only after exact 12/12 closure")
    _roots(seal)
    seal.add_argument("--job-id", required=True)
    unseal = sub.add_parser("unseal", help="Run the analyzer and verifier exactly once")
    _roots(unseal)
    return parser


def _provenance_argv(argv: Optional[Sequence[str]]) -> Sequence[str]:
    suffix = list(sys.argv[1:] if argv is None else argv)
    return [str(Path(__file__).resolve())] + suffix


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provenance_argv = _provenance_argv(argv)
    if args.command == "dry-run":
        result = build_full_manifest(
            run_root=AUTHORIZED_RUN_ROOT,
            expected_commit=args.expected_commit,
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            time_limit=args.time_limit,
            created_at_utc="DRY_RUN_METADATA_ONLY",
        )
    elif args.command == "prepare":
        result = prepare_run(
            card_path=args.card,
            p3c_root=args.p3c_root,
            p3d_root=args.p3d_root,
            p3e_root=args.p3e_root,
            expected_commit=args.expected_commit,
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            time_limit=args.time_limit,
            argv=provenance_argv,
        )
    elif args.command == "record-submit":
        result = record_submission(
            p3e_root=args.p3e_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            argv=provenance_argv,
        )
    elif args.command == "seal-check":
        result = seal_completed_sources(
            card_path=args.card,
            p3c_root=args.p3c_root,
            p3d_root=args.p3d_root,
            p3e_root=args.p3e_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            argv=provenance_argv,
        )
    elif args.command == "unseal":
        result = unseal_analyze_verify(
            card_path=args.card,
            p3c_root=args.p3c_root,
            p3d_root=args.p3d_root,
            p3e_root=args.p3e_root,
            expected_commit=args.expected_commit,
            argv=provenance_argv,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
