#!/usr/bin/env python3
"""Verify Gate E final artifacts and seal the analysis directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional

from tflt.looppilot.full import terminal_recommendation


REQUIRED = (
    "selected_policy.json", "pre_holdout_freeze_proof.json", "candidate_grid.json",
    "paired_samples.jsonl", "coverage_curves.json", "bootstrap.json", "random_null.json",
    "comparators.json", "oracle.json", "strata.json", "decision_trace.json",
    "gate_e_summary.json", "gate_e_report.md",
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.analysis_dir).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Gate E final artifacts missing: %s" % missing)
    selected_hash = _sha(root / "selected_policy.json")
    proof = _json(root / "pre_holdout_freeze_proof.json")
    if proof.get("selected_policy_sha256") != selected_hash or proof.get("holdout_labels_read") is not False or proof.get("exclusive_create") is not True:
        raise RuntimeError("Gate E selected-policy pre-holdout freeze proof failed")
    summary = _json(root / "gate_e_summary.json")
    trace = _json(root / "decision_trace.json")
    if summary.get("selected_policy_sha256") != selected_hash:
        raise RuntimeError("Gate E summary selected-policy hash mismatch")
    if trace.get("terminal_recommendation") != terminal_recommendation(trace):
        raise RuntimeError("Gate E terminal routing differs from frozen order")
    if summary.get("document_count", 0) < 1 or len((root / "paired_samples.jsonl").read_text().splitlines()) != summary["document_count"]:
        raise RuntimeError("Gate E paired sample count mismatch")
    manifest = root / "final_sha256.txt"
    complete = root / "analysis_complete.txt"
    if manifest.exists() or complete.exists():
        raise FileExistsError("Gate E final seal already exists")
    with manifest.open("x") as handle:
        for name in REQUIRED:
            handle.write("%s  %s\n" % (_sha(root / name), name))
    with complete.open("x") as handle:
        handle.write("complete\n")
    print(json.dumps({"final_manifest_sha256": _sha(manifest), "recommendation": trace["terminal_recommendation"]}, sort_keys=True))
    return 0


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
