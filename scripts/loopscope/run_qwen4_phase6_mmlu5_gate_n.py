#!/usr/bin/env python3
"""Execute the synchronous CPU-only LoopScope Phase 6 Gate N V3 path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase6_mmlu5_v3_selector import (  # noqa: E402
    EXPECTED_COMMIT,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    GateNV3Error,
    _jsonl_bytes,
    analyze_source,
    write_new_bytes,
    write_new_json,
)
from tflt.loopscope.phase6_mmlu5_v3_verifier import verify_freeze  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate N CPU-only post-hoc V3 rescore")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="run the one authorized V3 analyze")
    analyze.add_argument("--source-root", required=True)
    analyze.add_argument("--output-root", required=True)
    analyze.add_argument("--expected-commit", required=True)
    verify = subparsers.add_parser("verify", help="run the one fresh-process V3 verifier")
    verify.add_argument("--source-root", required=True)
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--expected-commit", required=True)
    return parser


def run_analyze(args: argparse.Namespace) -> int:
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if current_commit != args.expected_commit:
        raise GateNV3Error("BLOCK_REPOSITORY_ADMISSION_MISMATCH: repository HEAD differs")
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise GateNV3Error("BLOCK_WRITE_ONCE_VIOLATION: output root already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "analysis").mkdir(exist_ok=False)
    (output_root / "verifier").mkdir(exist_ok=False)
    write_new_json(
        output_root / "command_args.json",
        {
            "gate": "N",
            "command": "analyze",
            "source_root": str(Path(args.source_root).resolve()),
            "output_root": str(output_root),
            "expected_commit": args.expected_commit,
            "replicates": FORMAL_REPLICATES,
            "seed": FORMAL_SEED,
            "cpu_only": True,
            "model_loaded": False,
            "forward_executed": False,
            "gpu_slurm_executed": False,
            "outcome_read": False,
        },
    )
    payload, projected, _closure = analyze_source(
        Path(args.source_root),
        expected_commit=args.expected_commit,
        replicates=FORMAL_REPLICATES,
        seed=FORMAL_SEED,
    )
    projection_sha = write_new_bytes(
        output_root / "analysis/selector_projection.jsonl", _jsonl_bytes(projected)
    )
    if payload["selector_projection_file_sha256"] != projection_sha:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: projection hash differs before freeze")
    freeze_sha = write_new_json(output_root / "analysis/selector_freeze.json", payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "ANALYZE",
                "selector_freeze_file_sha256": freeze_sha,
                "selector_projection_file_sha256": projection_sha,
                "selector_decision": payload["selector_decision"],
                "selected_key": payload["selected_key"],
                "candidate_count": payload["candidate_count"],
                "record_count": payload["record_count"],
                "category_count": payload["category_count"],
                "bootstrap_draw_index_sha256": payload["bootstrap_draw_index_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_verify(args: argparse.Namespace) -> int:
    result = verify_freeze(
        source_root=Path(args.source_root),
        run_root=Path(args.run_root),
        expected_commit=args.expected_commit,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "analyze":
            return run_analyze(args)
        if args.command == "verify":
            return run_verify(args)
        raise GateNV3Error("BLOCK_COMMAND: unsupported Gate N command")
    except GateNV3Error as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
