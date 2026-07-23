#!/usr/bin/env python3
"""Run or verify LoopScope Gate H from two frozen scalar trajectory files."""

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
        raise RuntimeError("Gate H launcher requires a fresh Python process")
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

from tflt.loopscope.phase5_relative_biphasic_v2_absolute_rate import (  # noqa: E402
    METHOD,
    analyze_v2_model,
    build_primary_payloads,
    validate_card,
    verify_primary_outputs,
    write_primary_outputs,
)
from tflt.loopscope.phase5_relative_biphasic import (  # noqa: E402
    extract_scalar_samples,
    file_sha256,
    load_strict_json,
    load_strict_jsonl,
)


INPUT_ARGUMENTS = {
    "trajectory": "trajectory_sha256",
    "trajectory_manifest": "trajectory_manifest_sha256",
    "legacy_table": "legacy_table_sha256",
    "v1_candidates": "v1_candidates_sha256",
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
        for argument_name, card_key in INPUT_ARGUMENTS.items():
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
    stage_root = args.v1_stage_root
    v1_files = (
        (
            "configs/loopscope/phase5_relative_biphasic_reversal_card.json",
            "card_sha256",
        ),
        (
            "src/tflt/loopscope/phase5_relative_biphasic.py",
            "module_sha256",
        ),
        (
            "scripts/loopscope/run_qwen_phase5_gate_g.py",
            "launcher_sha256",
        ),
        (
            "tests/test_loopscope_phase5_relative_biphasic.py",
            "focused_tests_sha256",
        ),
    )
    verified_v1_files = []
    for relative_path, card_key in v1_files:
        path = stage_root / relative_path
        digest = file_sha256(path)
        if digest != card["v1_implementation"][card_key]:
            raise RuntimeError("V1 implementation dependency SHA256 differs: %s" % relative_path)
        verified_v1_files.append({"path": str(path), "sha256": digest})
    result["v1_implementation"] = {
        "stage_root": str(stage_root),
        "files": verified_v1_files,
    }
    return result


def _implementation_provenance(
    args: argparse.Namespace,
    card_sha256: str,
    runtime_git: Mapping[str, Any],
) -> Dict[str, Any]:
    stage_root = Path(__file__).resolve().parents[2]
    files = [
        {
            "path": "configs/loopscope/phase5_relative_biphasic_v2_absolute_rate_card.json",
            "sha256": card_sha256,
        },
        {
            "path": "src/tflt/loopscope/phase5_relative_biphasic_v2_absolute_rate.py",
            "sha256": file_sha256(
                stage_root
                / "src/tflt/loopscope/phase5_relative_biphasic_v2_absolute_rate.py"
            ),
        },
        {
            "path": "scripts/loopscope/run_qwen_phase5_gate_h.py",
            "sha256": file_sha256(Path(__file__)),
        },
        {
            "path": "tests/test_loopscope_phase5_relative_biphasic_v2_absolute_rate.py",
            "sha256": file_sha256(
                stage_root
                / "tests/test_loopscope_phase5_relative_biphasic_v2_absolute_rate.py"
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
    q17_samples = extract_scalar_samples(
        load_strict_jsonl(args.qwen17_trajectory),
        "qwen17",
        enforce_population=True,
    )
    q4_samples = extract_scalar_samples(
        load_strict_jsonl(args.qwen4_trajectory),
        "qwen4",
        enforce_population=True,
    )
    q17 = analyze_v2_model(
        q17_samples,
        "qwen17",
        load_strict_json(args.qwen17_v1_candidates),
        enforce_population=True,
    )
    q4 = analyze_v2_model(
        q4_samples,
        "qwen4",
        load_strict_json(args.qwen4_v1_candidates),
        enforce_population=True,
    )
    card_sha = file_sha256(args.card)
    implementation = _implementation_provenance(args, card_sha, runtime_git)
    payloads = build_primary_payloads(
        q17,
        q4,
        card_sha256=card_sha,
        input_provenance=inputs,
        implementation_provenance=implementation,
    )
    return q17, q4, inputs, implementation, payloads


def run_analyze(args: argparse.Namespace) -> Dict[str, Any]:
    q17, q4, _inputs, _implementation, payloads = _compute(args)
    hashes = write_primary_outputs(args.output_root, payloads)
    return {
        "command": "analyze",
        "method": METHOD,
        "output_root": str(args.output_root),
        "candidate_counts": {
            "qwen17": q17["candidate_count"],
            "qwen4": q4["candidate_count"],
        },
        "decisions": {
            "qwen17": q17["summary"]["decision"],
            "qwen4": q4["summary"]["decision"],
        },
        "ranking_candidate_counts": {
            "qwen17": q17["summary"]["ranking_candidate_count"],
            "qwen4": q4["summary"]["ranking_candidate_count"],
        },
        "primary_output_sha256": hashes,
        "outcome_values_consumed": False,
        "model_or_dataset_loaded": False,
        "forward": False,
        "gpu_cuda_slurm": False,
    }


def run_verify(args: argparse.Namespace) -> Dict[str, Any]:
    _q17, _q4, inputs, implementation, payloads = _compute(args)
    manifest = verify_primary_outputs(
        args.output_root,
        payloads,
        input_provenance=inputs,
        implementation_provenance=implementation,
    )
    return {
        "command": "verify",
        "method": METHOD,
        "output_root": str(args.output_root),
        "result": manifest["result"],
        "manifest_sha256": manifest["manifest_sha256"],
        "outcome_values_consumed": False,
        "model_or_dataset_loaded": False,
        "forward": False,
        "gpu_cuda_slurm": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--card", required=True, type=Path)
    for command in ("analyze", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--card", required=True, type=Path)
        for model_key in ("qwen17", "qwen4"):
            sub.add_argument("--%s-trajectory" % model_key, required=True, type=Path)
            sub.add_argument(
                "--%s-trajectory-manifest" % model_key, required=True, type=Path
            )
            sub.add_argument("--%s-legacy-table" % model_key, required=True, type=Path)
            sub.add_argument("--%s-v1-candidates" % model_key, required=True, type=Path)
        sub.add_argument("--output-root", required=True, type=Path)
        sub.add_argument("--source-checkout", required=True, type=Path)
        sub.add_argument("--v1-stage-root", required=True, type=Path)
        sub.add_argument("--expected-runtime-commit", required=True)
        sub.add_argument("--implementation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dry-run":
        validate_card(load_strict_json(args.card))
        result = {"command": "dry-run", "method": METHOD, "result": "PASS"}
    elif args.command == "analyze":
        result = run_analyze(args)
    else:
        result = run_verify(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
