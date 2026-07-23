#!/usr/bin/env python3
"""Run or verify LoopScope Gate G from two frozen scalar trajectory files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("Gate G launcher requires a fresh Python process")
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

from tflt.loopscope.phase5_relative_biphasic import (  # noqa: E402
    analyze_model,
    build_primary_payloads,
    extract_scalar_samples,
    file_sha256,
    load_strict_json,
    load_strict_jsonl,
    validate_card,
    verify_legacy_compatibility,
    verify_primary_outputs,
    write_primary_outputs,
)


EXPECTED_INPUT_KEYS = {
    "qwen17": {
        "trajectory": "trajectory_sha256",
        "trajectory_manifest": "trajectory_manifest_sha256",
        "legacy_table": "legacy_table_sha256",
    },
    "qwen4": {
        "trajectory": "trajectory_sha256",
        "trajectory_manifest": "trajectory_manifest_sha256",
        "legacy_table": "legacy_table_sha256",
    },
}


def _git_provenance(source_checkout: Path, expected_commit: str) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source_checkout), *args], text=True
        ).strip()

    top = run("rev-parse", "--show-toplevel")
    commit = run("rev-parse", "HEAD")
    branch = run("symbolic-ref", "--short", "HEAD")
    dirty = run("status", "--porcelain")
    if Path(top).resolve() != source_checkout.resolve():
        raise RuntimeError("runtime source checkout top-level differs")
    if commit != expected_commit or branch != "loopscope" or dirty:
        raise RuntimeError("runtime source checkout provenance differs")
    return {
        "source_checkout": str(source_checkout),
        "branch": branch,
        "commit": commit,
        "dirty": False,
    }


def _input_provenance(
    args: argparse.Namespace, card: Mapping[str, Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for model_key in ("qwen17", "qwen4"):
        expected = card["models"][model_key]["inputs"]
        observed: Dict[str, Any] = {}
        for argument_name, card_key in EXPECTED_INPUT_KEYS[model_key].items():
            path = getattr(args, "%s_%s" % (model_key, argument_name))
            digest = file_sha256(path)
            if digest != expected[card_key]:
                raise RuntimeError("%s %s SHA256 differs" % (model_key, argument_name))
            observed["%s_sha256" % argument_name] = digest
            observed["%s_path" % argument_name] = str(path)
        manifest = load_strict_json(
            getattr(args, "%s_trajectory_manifest" % model_key)
        )
        internal = manifest.get("trajectory_file_sha256")
        if internal is not None and internal != observed["trajectory_sha256"]:
            raise RuntimeError("%s trajectory manifest binding differs" % model_key)
        observed["trajectory_manifest_internal_sha256"] = manifest.get(
            "manifest_sha256"
        )
        result[model_key] = observed
    return result


def _implementation_provenance(
    args: argparse.Namespace,
    card_sha256: str,
    runtime_git: Mapping[str, Any],
) -> Dict[str, Any]:
    stage_root = Path(__file__).resolve().parents[2]
    files = [
        {
            "path": "configs/loopscope/phase5_relative_biphasic_reversal_card.json",
            "sha256": card_sha256,
        },
        {
            "path": "src/tflt/loopscope/phase5_relative_biphasic.py",
            "sha256": file_sha256(
                stage_root / "src/tflt/loopscope/phase5_relative_biphasic.py"
            ),
        },
        {
            "path": "scripts/loopscope/run_qwen_phase5_gate_g.py",
            "sha256": file_sha256(Path(__file__)),
        },
        {
            "path": "tests/test_loopscope_phase5_relative_biphasic.py",
            "sha256": file_sha256(
                stage_root / "tests/test_loopscope_phase5_relative_biphasic.py"
            ),
        },
    ]
    return {
        "authorized_local_implementation_commit": args.implementation_commit,
        "runtime_reference_git": dict(runtime_git),
        "staged_files": files,
        "python": sys.version,
    }


def _compute(args: argparse.Namespace):
    card = load_strict_json(args.card)
    validate_card(card)
    runtime_git = _git_provenance(args.source_checkout, args.expected_runtime_commit)
    inputs = _input_provenance(args, card)
    q17_records = load_strict_jsonl(args.qwen17_trajectory)
    q4_records = load_strict_jsonl(args.qwen4_trajectory)
    q17_samples = extract_scalar_samples(q17_records, "qwen17", enforce_population=True)
    q4_samples = extract_scalar_samples(q4_records, "qwen4", enforce_population=True)
    q17 = analyze_model(q17_samples, "qwen17", enforce_population=True)
    q4 = analyze_model(
        q4_samples,
        "qwen4",
        enforce_population=True,
        additional_windows=((4, 4), (4, 5), (5, 13), (4, 15), (5, 15)),
    )
    q17_legacy = load_strict_json(args.qwen17_legacy_table)
    q4_legacy = load_strict_json(args.qwen4_legacy_table)
    compatibility = {
        "qwen17": verify_legacy_compatibility(q17, q17_legacy, tolerance=1e-10),
        "qwen4": verify_legacy_compatibility(q4, q4_legacy, tolerance=1e-10),
    }
    card_sha = file_sha256(args.card)
    implementation = _implementation_provenance(
        args, card_sha, runtime_git
    )
    payloads = build_primary_payloads(
        q17,
        q4,
        card_sha256=card_sha,
        input_provenance=inputs,
        implementation_provenance=implementation,
        compatibility=compatibility,
    )
    return q17, q4, inputs, implementation, compatibility, payloads


def run_analyze(args: argparse.Namespace) -> Dict[str, Any]:
    q17, q4, _inputs, _implementation, compatibility, payloads = _compute(args)
    hashes = write_primary_outputs(args.output_root, payloads)
    return {
        "command": "analyze",
        "method": "RELATIVE_BIPHASIC_REVERSAL_V1",
        "output_root": str(args.output_root),
        "candidate_counts": {
            "qwen17": q17["candidate_count"],
            "qwen4": q4["candidate_count"],
        },
        "decisions": {
            "qwen17": q17["summary"]["decision"],
            "qwen4": q4["summary"]["decision"],
        },
        "new_eligible_counts": {
            "qwen17": q17["summary"]["new_eligible_count"],
            "qwen4": q4["summary"]["new_eligible_count"],
        },
        "legacy_compatibility": compatibility,
        "primary_output_sha256": hashes,
        "information_boundary_preserved": True,
    }


def run_verify(args: argparse.Namespace) -> Dict[str, Any]:
    _q17, _q4, inputs, implementation, _compatibility, payloads = _compute(args)
    manifest = verify_primary_outputs(
        args.output_root,
        payloads,
        input_provenance=inputs,
        implementation_provenance=implementation,
    )
    return {
        "command": "verify",
        "method": "RELATIVE_BIPHASIC_REVERSAL_V1",
        "output_root": str(args.output_root),
        "result": manifest["result"],
        "manifest_sha256": manifest["manifest_sha256"],
        "information_boundary_preserved": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--card", required=True, type=Path)
        sub.add_argument("--qwen17-trajectory", required=True, type=Path)
        sub.add_argument("--qwen17-trajectory-manifest", required=True, type=Path)
        sub.add_argument("--qwen17-legacy-table", required=True, type=Path)
        sub.add_argument("--qwen4-trajectory", required=True, type=Path)
        sub.add_argument("--qwen4-trajectory-manifest", required=True, type=Path)
        sub.add_argument("--qwen4-legacy-table", required=True, type=Path)
        sub.add_argument("--output-root", required=True, type=Path)
        sub.add_argument("--source-checkout", required=True, type=Path)
        sub.add_argument("--expected-runtime-commit", required=True)
        sub.add_argument("--implementation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analyze(args) if args.command == "analyze" else run_verify(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
