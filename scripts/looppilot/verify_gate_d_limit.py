#!/usr/bin/env python3
"""Verify and close the frozen Gate D baseline/loop/signal/decision join."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional

from tflt.looppilot.limit import validate_gate_d_config, verify_gate_d_records
from tflt.looppilot.schema import write_json_once


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_d_limit.json")
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    preflight_path = Path(args.preflight_manifest).resolve()
    results_dir = Path(args.results_dir).resolve()
    config = _load_json(config_path)
    validate_gate_d_config(config)
    preflight = _load_json(preflight_path)
    if preflight.get("config_hash") != _sha256_file(config_path):
        raise RuntimeError("Gate D verifier preflight config hash mismatch")
    if preflight.get("document_count") != 20:
        raise RuntimeError("Gate D verifier requires 20 frozen preflight docs")

    required = (
        "baseline/lm_eval_results.json",
        "loop/lm_eval_results.json",
        "baseline_samples.jsonl",
        "loop_samples.jsonl",
        "signal_records.jsonl",
        "decisions.jsonl",
        "revision_closure.json",
        "resolved_model.json",
        "resolved_tokenizer.json",
        "renderer_manifest.json",
        "runtime.json",
    )
    missing = [name for name in required if not (results_dir / name).is_file()]
    if missing:
        raise RuntimeError("Gate D terminal artifacts are missing: %s" % missing)

    closure = _load_json(results_dir / "revision_closure.json")
    if closure.get("model_commit") != config["model_revision"]:
        raise RuntimeError("Gate D model revision closure mismatch")
    if closure.get("tokenizer_commit") != config["tokenizer_revision"]:
        raise RuntimeError("Gate D tokenizer revision closure mismatch")
    if closure.get("lm_eval_version") != config["lm_eval_version"]:
        raise RuntimeError("Gate D lm-eval revision closure mismatch")
    if _load_json(results_dir / "renderer_manifest.json") != preflight.get("renderer_manifest"):
        raise RuntimeError("Gate D renderer manifest differs from preflight")

    baseline = _read_jsonl(results_dir / "baseline_samples.jsonl")
    loop = _read_jsonl(results_dir / "loop_samples.jsonl")
    signals = _read_jsonl(results_dir / "signal_records.jsonl")
    decisions = _read_jsonl(results_dir / "decisions.jsonl")
    report, summary = verify_gate_d_records(baseline, loop, signals, decisions)

    frozen_keys = {row["doc_key"] for row in preflight["documents"]}
    if {row["doc_key"] for row in baseline} != frozen_keys:
        raise RuntimeError("Gate D joined keys differ from frozen preflight docs")
    runtime = _load_json(results_dir / "runtime.json")
    for flag in ("threshold_selected", "ranking_computed", "limit_for_scientific_claim"):
        if runtime.get(flag) is not False or config.get(flag) is not False:
            raise RuntimeError("Gate D non-scientific-use flag mismatch: %s" % flag)
    gate_c_link = preflight.get("gate_c_link", {})
    if gate_c_link.get("results_manifest_sha256") != config[
        "gate_c_results_manifest_sha256"
    ]:
        raise RuntimeError("Gate D accepted Gate C linkage mismatch")
    if gate_c_link.get("always_loop_legacy_equivalence_accepted") is not True:
        raise RuntimeError("Gate D lacks accepted AlwaysLoop/legacy equivalence linkage")

    report.update(
        {
            "config_sha256": _sha256_file(config_path),
            "preflight_manifest_sha256": _sha256_file(preflight_path),
            "revision_closure_id": closure.get("closure_id"),
            "frozen_preflight_keys_match": True,
            "gate_c_results_manifest_sha256": gate_c_link["results_manifest_sha256"],
        }
    )
    summary.update(
        {
            "config_sha256": _sha256_file(config_path),
            "preflight_manifest_sha256": _sha256_file(preflight_path),
            "join_schema_sha256": preflight.get("join_schema_hash"),
            "mask_schema_sha256": preflight.get("mask_schema_hash"),
            "criterion_sha256": preflight.get("criterion_hash"),
            "revision_closure_id": closure.get("closure_id"),
            "gate_c_results_manifest_sha256": gate_c_link["results_manifest_sha256"],
            "baseline_raw_results_present": True,
            "loop_raw_results_present": True,
        }
    )
    write_json_once(results_dir / "join_report.json", report)
    write_json_once(results_dir / "run_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
