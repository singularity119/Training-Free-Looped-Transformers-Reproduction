#!/usr/bin/env python3
"""Run LoopScope Phase 5 Extension Gate E on existing validation trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Sequence


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("Gate E launcher requires a fresh Python process")
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

from tflt.loopscope.phase5_variable_width import (  # noqa: E402
    analyze_boundary_samples,
    build_output_payloads,
    extract_boundary_samples,
    file_sha256,
    load_strict_json,
    load_strict_jsonl,
    validate_extension_card,
    verify_and_close_outputs,
    verify_width4_backward_compatibility,
    write_primary_outputs,
)


def _git_provenance(source_checkout: Path, expected_commit: str) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source_checkout), *args], text=True
        ).strip()

    commit = run("rev-parse", "HEAD")
    branch = run("symbolic-ref", "--short", "HEAD")
    dirty = run("status", "--porcelain")
    if commit != expected_commit or branch != "loopscope" or dirty:
        raise RuntimeError("HPC2 source checkout provenance differs")
    return {
        "source_checkout": str(source_checkout),
        "branch": branch,
        "commit": commit,
        "dirty": False,
    }


def _input_provenance(args: argparse.Namespace, card: Dict[str, Any]) -> Dict[str, Any]:
    expected = card["parent"]
    observed = {
        "trajectory_file_sha256": file_sha256(args.trajectory),
        "trajectory_manifest_file_sha256": file_sha256(args.trajectory_manifest),
        "width4_selector_report_file_sha256": file_sha256(args.selector_report),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError("%s differs from frozen Gate E card" % key)
    manifest = load_strict_json(args.trajectory_manifest)
    if manifest.get("trajectory_file_sha256") != observed["trajectory_file_sha256"]:
        raise RuntimeError("trajectory manifest binding differs")
    return {
        **observed,
        "trajectory_path": str(args.trajectory),
        "trajectory_manifest_path": str(args.trajectory_manifest),
        "width4_selector_report_path": str(args.selector_report),
        "trajectory_manifest_internal_sha256": manifest.get("manifest_sha256"),
    }


def _implementation_provenance(
    args: argparse.Namespace, card_sha256: str, git: Dict[str, Any]
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = {
        "configs/loopscope/phase5_multiwidth_mid40_card.json": card_sha256,
        "src/tflt/loopscope/phase5_variable_width.py": file_sha256(
            root / "src/tflt/loopscope/phase5_variable_width.py"
        ),
        "scripts/loopscope/run_qwen4base_phase5_gate_e.py": file_sha256(Path(__file__)),
    }
    return {"git": git, "staged_file_sha256": files, "python": sys.version}


def _compute(args: argparse.Namespace):
    card = load_strict_json(args.card)
    validate_extension_card(card)
    git = _git_provenance(args.source_checkout, args.expected_source_commit)
    input_provenance = _input_provenance(args, card)
    records = load_strict_jsonl(args.trajectory)
    samples = extract_boundary_samples(records, enforce_population=True)
    analysis = analyze_boundary_samples(samples, enforce_population=True)
    selector_report = load_strict_json(args.selector_report)
    compatibility = verify_width4_backward_compatibility(
        analysis["rows"],
        selector_report,
        tolerance=float(card["backward_compatibility"]["absolute_tolerance"]),
    )
    if (
        analysis["bootstrap"]["draw_index_sha256"]
        != selector_report["bootstrap"]["draw_index_sha256"]
    ):
        raise RuntimeError("Gate E bootstrap draw stream differs from width4 selector")
    card_sha256 = file_sha256(args.card)
    implementation = _implementation_provenance(args, card_sha256, git)
    payloads = build_output_payloads(
        analysis,
        card_file_sha256=card_sha256,
        input_provenance=input_provenance,
        implementation_provenance=implementation,
        width4_compatibility=compatibility,
    )
    return analysis, selector_report, input_provenance, implementation, payloads


def run_analyze(args: argparse.Namespace) -> Dict[str, Any]:
    analysis, _selector, _inputs, _implementation, payloads = _compute(args)
    hashes = write_primary_outputs(args.output_root, payloads)
    return {
        "command": "analyze",
        "output_root": str(args.output_root),
        "candidate_count": analysis["candidate_count"],
        "eligible_count": sum(bool(row["point_eligible"]) for row in analysis["rows"]),
        "primary_output_sha256": hashes,
        "outcome_values_consumed": False,
    }


def run_verify(args: argparse.Namespace) -> Dict[str, Any]:
    analysis, selector, inputs, implementation, payloads = _compute(args)
    manifest = verify_and_close_outputs(
        args.output_root,
        payloads,
        input_provenance=inputs,
        implementation_provenance=implementation,
        bootstrap_draw_digest_matches_width4_selector=(
            analysis["bootstrap"]["draw_index_sha256"]
            == selector["bootstrap"]["draw_index_sha256"]
        ),
    )
    return {
        "command": "verify",
        "output_root": str(args.output_root),
        "result": manifest["result"],
        "manifest_sha256": manifest["manifest_sha256"],
        "outcome_values_consumed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--card", required=True, type=Path)
        sub.add_argument("--trajectory", required=True, type=Path)
        sub.add_argument("--trajectory-manifest", required=True, type=Path)
        sub.add_argument("--selector-report", required=True, type=Path)
        sub.add_argument("--output-root", required=True, type=Path)
        sub.add_argument("--source-checkout", required=True, type=Path)
        sub.add_argument("--expected-source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analyze(args) if args.command == "analyze" else run_verify(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
