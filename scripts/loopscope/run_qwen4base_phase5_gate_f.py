#!/usr/bin/env python3
"""Operate Gate F's frozen original four cells plus the authorized 15:19 supplement."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("Gate F launcher requires a fresh Python process")
    root = Path(__file__).resolve().parents[2]
    tflt_paths = [root / "src/tflt"]
    for entry in sys.path:
        candidate = Path(entry or ".").resolve() / "tflt"
        if candidate.is_dir() and candidate not in tflt_paths:
            tflt_paths.append(candidate)
    package = types.ModuleType("tflt")
    package.__path__ = [str(path) for path in tflt_paths]
    package.__package__ = "tflt"
    sys.modules["tflt"] = package
    loopscope = types.ModuleType("tflt.loopscope")
    loopscope.__path__ = [
        str(path / "loopscope") for path in tflt_paths if (path / "loopscope").is_dir()
    ]
    loopscope.__package__ = "tflt.loopscope"
    sys.modules["tflt.loopscope"] = loopscope


_install_isolated_tflt_namespace()

from tflt.loopscope.phase5_variable_width_outcome import (  # noqa: E402
    build_launch_manifest,
    build_supplement_launch_manifest,
    freeze_formal,
    implementation_hashes,
    prepare_smoke,
    prepare_supplement_formal,
    record_submission,
    record_supplement_submission,
    run_cell,
    run_combined_one_shot_analysis,
    run_one_shot_analysis,
    run_supplement_cell,
    seal_stage,
    seal_supplement,
)
from tflt.loopscope.phase5_variable_width_outcome import (  # noqa: E402
    _load_card,
    _strict_json,
)


def _scheduler(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", required=True)
    parser.add_argument("--gres", required=True)
    parser.add_argument("--qos")
    parser.add_argument("--account")
    parser.add_argument("--array-throttle", required=True, type=int)
    parser.add_argument("--time-limit", required=True)
    parser.add_argument("--batch-size", required=True)


def _root_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)


def _implementation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--implementation-commit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run")
    dry.add_argument("--mode", choices=("smoke", "formal"), required=True)
    _root_source(dry)
    _implementation(dry)
    _scheduler(dry)

    prepare = sub.add_parser("prepare-smoke")
    _root_source(prepare)
    _implementation(prepare)
    _scheduler(prepare)

    formal = sub.add_parser("freeze-formal")
    _root_source(formal)
    _implementation(formal)
    _scheduler(formal)

    supplement_dry = sub.add_parser("supplement-dry-run")
    _root_source(supplement_dry)
    _implementation(supplement_dry)
    _scheduler(supplement_dry)

    supplement_prepare = sub.add_parser("prepare-supplement-formal")
    _root_source(supplement_prepare)
    _implementation(supplement_prepare)
    _scheduler(supplement_prepare)

    supplement_cell = sub.add_parser("run-supplement-cell")
    _root_source(supplement_cell)

    supplement_submit = sub.add_parser("record-supplement-submission")
    supplement_submit.add_argument("--run-root", required=True, type=Path)
    supplement_submit.add_argument("--job-id", required=True)

    supplement_seal = sub.add_parser("seal-supplement")
    _root_source(supplement_seal)
    supplement_seal.add_argument("--job-id", required=True)

    cell = sub.add_parser("run-cell")
    cell.add_argument("--mode", choices=("smoke", "formal"), required=True)
    _root_source(cell)
    cell.add_argument("--cell-index", required=True, type=int)

    submit = sub.add_parser("record-submission")
    submit.add_argument("--mode", choices=("smoke", "formal"), required=True)
    submit.add_argument("--run-root", required=True, type=Path)
    submit.add_argument("--job-id", required=True)

    seal = sub.add_parser("seal")
    seal.add_argument("--mode", choices=("smoke", "formal"), required=True)
    _root_source(seal)
    seal.add_argument("--job-id", required=True)

    analyze = sub.add_parser("analyze-once")
    _root_source(analyze)
    _implementation(analyze)

    combined = sub.add_parser("analyze-combined")
    combined.add_argument("--original-run-root", required=True, type=Path)
    combined.add_argument("--supplement-run-root", required=True, type=Path)
    combined.add_argument("--combined-run-root", required=True, type=Path)
    combined.add_argument("--expected-source-commit", required=True)
    _implementation(combined)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.command == "dry-run":
        card = _load_card()
        result = build_launch_manifest(
            mode=args.mode,
            run_root=args.run_root,
            implementation_commit=args.implementation_commit,
            expected_source_commit=args.expected_source_commit,
            partition=args.partition,
            gres=args.gres,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            time_limit=args.time_limit,
            batch_size=args.batch_size,
            created_at_utc="2026-07-23T00:00:00Z",
            card=card,
            score_payload=_strict_json(Path(card["source"]["gate_e_score_path"])),
            implementation_sha256=implementation_hashes(),
        )
    elif args.command == "prepare-smoke":
        result = prepare_smoke(
            run_root=args.run_root,
            implementation_commit=args.implementation_commit,
            expected_source_commit=args.expected_source_commit,
            partition=args.partition,
            gres=args.gres,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            time_limit=args.time_limit,
            batch_size=args.batch_size,
        )
    elif args.command == "freeze-formal":
        result = freeze_formal(
            run_root=args.run_root,
            implementation_commit=args.implementation_commit,
            expected_source_commit=args.expected_source_commit,
            partition=args.partition,
            gres=args.gres,
            qos=args.qos,
            account=args.account,
            array_throttle=args.array_throttle,
            time_limit=args.time_limit,
            batch_size=args.batch_size,
        )
    elif args.command == "supplement-dry-run":
        card = _load_card()
        result = build_supplement_launch_manifest(
            run_root=args.run_root,
            implementation_commit=args.implementation_commit,
            expected_source_commit=args.expected_source_commit,
            partition=args.partition,
            gres=args.gres,
            qos=args.qos,
            account=args.account,
            time_limit=args.time_limit,
            batch_size=args.batch_size,
            created_at_utc="2026-07-23T00:00:00Z",
            card=card,
            score_payload=_strict_json(Path(card["source"]["gate_e_score_path"])),
            implementation_sha256=implementation_hashes(),
        )
    elif args.command == "prepare-supplement-formal":
        result = prepare_supplement_formal(
            run_root=args.run_root,
            implementation_commit=args.implementation_commit,
            expected_source_commit=args.expected_source_commit,
            partition=args.partition,
            gres=args.gres,
            qos=args.qos,
            account=args.account,
            time_limit=args.time_limit,
            batch_size=args.batch_size,
        )
    elif args.command == "run-supplement-cell":
        run_supplement_cell(
            run_root=args.run_root,
            expected_source_commit=args.expected_source_commit,
        )
        result = {"status": "PRODUCER_COMPLETED", "cell": "15:19"}
    elif args.command == "record-supplement-submission":
        result = record_supplement_submission(
            run_root=args.run_root, job_id=args.job_id
        )
    elif args.command == "seal-supplement":
        result = seal_supplement(
            run_root=args.run_root,
            expected_source_commit=args.expected_source_commit,
            job_id=args.job_id,
        )
    elif args.command == "run-cell":
        run_cell(
            mode=args.mode,
            run_root=args.run_root,
            cell_index=args.cell_index,
            expected_source_commit=args.expected_source_commit,
        )
        result = {
            "mode": args.mode,
            "cell_index": args.cell_index,
            "status": "PRODUCER_COMPLETED",
        }
    elif args.command == "record-submission":
        result = record_submission(
            mode=args.mode, run_root=args.run_root, job_id=args.job_id
        )
    elif args.command == "seal":
        result = seal_stage(
            mode=args.mode,
            run_root=args.run_root,
            expected_source_commit=args.expected_source_commit,
            job_id=args.job_id,
        )
    elif args.command == "analyze-once":
        result = run_one_shot_analysis(
            run_root=args.run_root,
            expected_source_commit=args.expected_source_commit,
            implementation_commit=args.implementation_commit,
            argv=raw_argv,
        )
    else:
        result = run_combined_one_shot_analysis(
            original_run_root=args.original_run_root,
            supplement_run_root=args.supplement_run_root,
            combined_run_root=args.combined_run_root,
            expected_source_commit=args.expected_source_commit,
            implementation_commit=args.implementation_commit,
            argv=raw_argv,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
