#!/usr/bin/env python3
"""Run and independently verify the CPU-only Phase 3 Gate F dual ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from tflt.loopscope.phase3_schema import load_phase3_card
from tflt.loopscope.phase3_variable_width import (
    DUAL_CARD_SHA256,
    EDGE_CSV,
    EDGE_JSON,
    RANKINGS_JSON,
    STRICT_CSV,
    STRICT_JSON,
    VERIFIER_JSON,
    analyze_boundary_metric_records,
    build_artifact_payloads,
    extract_boundary_metric_records,
    file_sha256,
    load_dual_card,
    load_strict_json,
    load_strict_jsonl,
    verify_recomputed_artifacts,
    verify_width4_backward_compatibility,
    write_new_json,
    write_new_text,
)
from tflt.loopscope.schema import SchemaError, attach_manifest_sha256, verify_manifest_sha256


P3C_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3c-20260716T064601Z"
)
OLD_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3f-20260716T115623Z"
)
PARENT_CARD_NAME = "phase3_card.json"
SELECTOR_NAME = "selector_freeze.json"
TRAJECTORY_MANIFEST_NAME = "validation1531_no_loop_trajectory_manifest.json"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository_root()), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise SchemaError("git command failed: %s" % process.stderr.strip())
    return process.stdout.strip()


def verify_git(expected_commit: str) -> Dict[str, Any]:
    if len(expected_commit) != 40 or any(ch not in "0123456789abcdef" for ch in expected_commit):
        raise SchemaError("expected commit must be lowercase 40-hex")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    head = _git_output("rev-parse", "HEAD")
    origin = _git_output("rev-parse", "origin/loopscope")
    dirty = bool(_git_output("status", "--porcelain"))
    if branch != "loopscope" or head != expected_commit or origin != expected_commit or dirty:
        raise SchemaError("Git branch/commit/clean closure differs")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root()),
            "merge-base",
            "--is-ancestor",
            "4f59bd93eca4da3cbf458a93508f91c5b23912bc",
            "HEAD",
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise SchemaError("fixed ancestor closure differs")
    return {"branch": branch, "commit": head, "origin_loopscope": origin, "dirty": False}


def assert_paths(
    card_path: Path, p3b_root: Path, p3c_root: Path, p3f_root: Path, card: Mapping[str, Any]
) -> None:
    repo = repository_root().resolve()
    if card_path.resolve() != repo / "configs/loopscope/phase3_variable_width_dual_card.json":
        raise SchemaError("dual card path differs from the exact authorization")
    if p3b_root.resolve() != Path(card["trajectory_source"]["run_root"]):
        raise SchemaError("P3-B root differs from the dual card")
    if p3c_root.resolve() != P3C_ROOT:
        raise SchemaError("P3-C selector root differs from authorization")
    if p3f_root.resolve() != Path(card["outputs"]["write_once_run_root"]):
        raise SchemaError("P3-F dual output root differs from authorization")
    if OLD_ROOT.exists():
        raise SchemaError("superseded P3-F root unexpectedly exists")


def load_inputs(
    card_path: Path, p3b_root: Path, p3c_root: Path, p3f_root: Path
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    card = load_dual_card(card_path)
    assert_paths(card_path, p3b_root, p3c_root, p3f_root, card)
    parent_path = card_path.parent / PARENT_CARD_NAME
    if file_sha256(parent_path) != card["parent"]["phase3_card_sha256"]:
        raise SchemaError("parent card SHA256 mismatch")
    parent_card = load_phase3_card(parent_path)
    records_path = p3b_root / card["trajectory_source"]["records"]
    manifest_path = p3b_root / TRAJECTORY_MANIFEST_NAME
    selector_path = p3c_root / SELECTOR_NAME
    if file_sha256(records_path) != card["trajectory_source"]["records_sha256"]:
        raise SchemaError("P3-B trajectory SHA256 mismatch")
    if file_sha256(manifest_path) != card["trajectory_source"]["manifest_sha256"]:
        raise SchemaError("P3-B trajectory manifest SHA256 mismatch")
    if file_sha256(selector_path) != card["parent"]["width4_selector_freeze_sha256"]:
        raise SchemaError("P3-C selector-freeze SHA256 mismatch")
    manifest = load_strict_json(manifest_path)
    selector = load_strict_json(selector_path)
    verify_manifest_sha256(manifest)
    verify_manifest_sha256(selector)
    if selector.get("process_isolation", {}).get("outcome_fields_consumed") is not False:
        raise SchemaError("selector freeze does not preserve the outcome boundary")
    trajectories = load_strict_jsonl(records_path)
    metrics = extract_boundary_metric_records(trajectories, parent_card, enforce_population=True)
    provenance = {
        "dual_card": {"path": str(card_path), "file_sha256": DUAL_CARD_SHA256},
        "parent_card": {"path": str(parent_path), "file_sha256": file_sha256(parent_path)},
        "trajectory": {"path": str(records_path), "file_sha256": file_sha256(records_path)},
        "trajectory_manifest": {
            "path": str(manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "internal_sha256": manifest["manifest_sha256"],
        },
        "selector_freeze": {
            "path": str(selector_path),
            "file_sha256": file_sha256(selector_path),
            "internal_sha256": selector["manifest_sha256"],
            "usage": "width4_backward_compatibility_only",
        },
    }
    return card, metrics, selector, provenance, parent_card


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    git = verify_git(args.expected_commit)
    card, metrics, selector, provenance, _parent = load_inputs(
        args.card, args.p3b_root, args.p3c_root, args.p3f_root
    )
    if args.p3f_root.exists():
        raise FileExistsError("refusing to reuse P3-F dual run root: %s" % args.p3f_root)
    analysis = analyze_boundary_metric_records(metrics, card, enforce_population=True)
    compatibility = verify_width4_backward_compatibility(
        analysis["strict_rows"], analysis["width4_summary"], selector, card
    )
    strict, strict_csv, edge, edge_csv, rankings = build_artifact_payloads(
        analysis,
        card,
        git_commit=git["commit"],
        input_provenance=provenance,
        backward_compatibility=compatibility,
    )
    args.p3f_root.mkdir(parents=True, exist_ok=False)
    hashes = {
        STRICT_JSON: write_new_json(args.p3f_root / STRICT_JSON, strict),
        STRICT_CSV: write_new_text(args.p3f_root / STRICT_CSV, strict_csv),
        EDGE_JSON: write_new_json(args.p3f_root / EDGE_JSON, edge),
        EDGE_CSV: write_new_text(args.p3f_root / EDGE_CSV, edge_csv),
        RANKINGS_JSON: write_new_json(args.p3f_root / RANKINGS_JSON, rankings),
    }
    return {
        "status": "P3F_DUAL_ANALYSIS_MATERIALIZED",
        "artifact_hashes": hashes,
        "strict": analysis["strict_summary"],
        "edge_aware": analysis["edge_summary"],
        "width4_backward_compatibility": compatibility,
        "bootstrap": analysis["bootstrap"],
        "outcome_values_consumed": False,
    }


def run_verify(args: argparse.Namespace) -> Dict[str, Any]:
    git = verify_git(args.expected_commit)
    card, metrics, selector, provenance, _parent = load_inputs(
        args.card, args.p3b_root, args.p3c_root, args.p3f_root
    )
    expected_before = {STRICT_JSON, STRICT_CSV, EDGE_JSON, EDGE_CSV, RANKINGS_JSON}
    actual_before = {path.name for path in args.p3f_root.iterdir() if path.is_file()}
    if actual_before != expected_before:
        raise SchemaError("P3-F dual root does not contain exactly five pre-verifier artifacts")
    if (args.p3f_root / VERIFIER_JSON).exists():
        raise FileExistsError("refusing to replace verifier receipt")
    strict = load_strict_json(args.p3f_root / STRICT_JSON)
    edge = load_strict_json(args.p3f_root / EDGE_JSON)
    rankings = load_strict_json(args.p3f_root / RANKINGS_JSON)
    verify_manifest_sha256(strict)
    verify_manifest_sha256(edge)
    verify_manifest_sha256(rankings)
    strict_csv = (args.p3f_root / STRICT_CSV).read_text(encoding="utf-8")
    edge_csv = (args.p3f_root / EDGE_CSV).read_text(encoding="utf-8")

    # Fresh raw recomputation: this is intentionally not a hash-only verifier.
    analysis = analyze_boundary_metric_records(metrics, card, enforce_population=True)
    compatibility = verify_recomputed_artifacts(
        analysis,
        card,
        selector,
        git_commit=git["commit"],
        input_provenance=provenance,
        strict_payload=strict,
        strict_csv=strict_csv,
        edge_payload=edge,
        edge_csv=edge_csv,
        rankings_payload=rankings,
    )
    artifact_hashes = {
        name: file_sha256(args.p3f_root / name) for name in sorted(expected_before)
    }
    receipt: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.dual_variable_width_verifier_receipt.v1",
        "artifact_role": "independent_raw_trajectory_dual_ranking_recompute_receipt",
        "status": "PASS",
        "card_sha256": DUAL_CARD_SHA256,
        "git": git,
        "inputs": provenance,
        "verified_artifact_sha256": artifact_hashes,
        "implementation_sha256": {
            "analysis": file_sha256(
                repository_root() / "src/tflt/loopscope/phase3_variable_width.py"
            ),
            "launcher": file_sha256(
                repository_root() / "scripts/loopscope/run_qwen17_phase3_p3f.py"
            ),
        },
        "verification": {
            "raw_trajectory_reloaded": True,
            "raw_metrics_recomputed": True,
            "strict_row_count": len(analysis["strict_rows"]),
            "edge_aware_row_count": len(analysis["edge_rows"]),
            "strict_scores_finite": all(math_isfinite(row["score"]) for row in analysis["strict_rows"]),
            "edge_aware_scores_finite": all(math_isfinite(row["score"]) for row in analysis["edge_rows"]),
            "edge_reference_count_min": min(row["reference_count"] for row in analysis["edge_rows"]),
            "edge_reference_count_max": max(row["reference_count"] for row in analysis["edge_rows"]),
            "bootstrap_index_stream_sha256": analysis["bootstrap"]["index_stream_sha256"],
            "width4_backward_compatibility": compatibility,
            "strict_ranking_recomputed": True,
            "edge_aware_ranking_recomputed": True,
        },
        "outcome_values_consumed": False,
    }
    attach_manifest_sha256(receipt)
    receipt_hash = write_new_json(args.p3f_root / VERIFIER_JSON, receipt)
    final_names = {path.name for path in args.p3f_root.iterdir() if path.is_file()}
    if final_names != expected_before | {VERIFIER_JSON}:
        raise SchemaError("P3-F dual root does not contain exactly six artifacts")
    return {
        "status": "P3F_DUAL_INDEPENDENT_RECOMPUTE_PASS",
        "receipt_file_sha256": receipt_hash,
        "receipt_internal_sha256": receipt["manifest_sha256"],
        "strict": analysis["strict_summary"],
        "edge_aware": analysis["edge_summary"],
        "width4_backward_compatibility": compatibility,
        "outcome_values_consumed": False,
    }


def math_isfinite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("analyze", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--card", type=Path, required=True)
        child.add_argument("--p3b-root", type=Path, required=True)
        child.add_argument("--p3c-root", type=Path, required=True)
        child.add_argument("--p3f-root", type=Path, required=True)
        child.add_argument("--expected-commit", required=True)
    return result


def main(argv: Sequence[str] = ()) -> int:
    args = parser().parse_args(list(argv) if argv else None)
    payload = run_analysis(args) if args.command == "analyze" else run_verify(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
