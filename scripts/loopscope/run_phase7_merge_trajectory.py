#!/usr/bin/env python3
"""Merge and seal one model's Phase 7 trajectory shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tflt.loopscope.phase7_manifest import validate_manifest_rows
from tflt.loopscope.phase7_schema import Phase7ContractError, validate_sanitized_record


MODEL_KEYS = ("qwen25_3b", "llama32_3b", "gemma2_2b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and verify Phase 7 trajectory shards.")
    parser.add_argument("--manifest", required=True, help="Common safe validation manifest JSON.")
    parser.add_argument("--shard", action="append", required=True, help="Fresh sanitized JSONL shard; repeat per shard.")
    parser.add_argument("--output", required=True, help="Fresh merged sanitized JSONL.")
    parser.add_argument("--receipt", required=True, help="Fresh merge receipt JSON.")
    parser.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--expected-count", type=int, default=1531)
    parser.add_argument("--expected-subjects", type=int, default=57)
    return parser


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Phase7ContractError("invalid JSON: %s" % path) from exc


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Phase7ContractError("invalid JSON at %s:%d" % (path, line_number)) from exc
            if not isinstance(value, dict):
                raise Phase7ContractError("trajectory shard row is not an object: %s:%d" % (path, line_number))
            rows.append(value)
    if not rows:
        raise Phase7ContractError("trajectory shard is empty: %s" % path)
    return rows


def _write_once(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest_value = _load_json(Path(args.manifest))
    if not isinstance(manifest_value, list):
        raise Phase7ContractError("common manifest must be a JSON list")
    manifest = validate_manifest_rows(
        manifest_value,
        expected_count=args.expected_count,
        expected_subjects=args.expected_subjects,
    )
    expected = {row["canonical_identity"]: row["subject"] for row in manifest}
    records: Dict[str, Dict[str, Any]] = {}
    shard_counts: List[int] = []
    for shard_name in args.shard:
        rows = _load_jsonl(Path(shard_name))
        shard_counts.append(len(rows))
        for row in rows:
            validate_sanitized_record(
                row,
                expected_model_key=args.model_key,
                expected_layer_count=args.layer_count,
            )
            if row["model_revision"] != args.model_revision:
                raise Phase7ContractError("trajectory shard model revision differs")
            if row["tokenizer_revision"] != args.tokenizer_revision:
                raise Phase7ContractError("trajectory shard tokenizer revision differs")
            identity = row["canonical_identity"]
            if identity not in expected:
                raise Phase7ContractError("trajectory shard contains an extra identity")
            if row["subject"] != expected[identity]:
                raise Phase7ContractError("trajectory subject differs from common manifest")
            if identity in records:
                raise Phase7ContractError("trajectory shards contain a duplicate identity")
            records[identity] = row
    missing = [identity for identity in expected if identity not in records]
    if missing:
        raise Phase7ContractError("trajectory shards are incomplete: missing=%d" % len(missing))
    if len(records) != len(expected):
        raise Phase7ContractError("trajectory membership is not an exact common-manifest closure")
    ordered = [records[row["canonical_identity"]] for row in manifest]
    output_text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in ordered)
    _write_once(Path(args.output), output_text)
    receipt = {
        "status": "PASS",
        "model_key": args.model_key,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "layer_count": args.layer_count,
        "shard_count": len(args.shard),
        "shard_record_counts": shard_counts,
        "record_count": len(ordered),
        "unique_identity_count": len(records),
        "subject_count": len({row["subject"] for row in ordered}),
        "forward_count": sum(row["forward_count"] for row in ordered),
        "loop_insertions": sorted({row["loop_insertions"] for row in ordered}),
        "common_manifest_membership": "exact",
        "information_barrier": "sanitized scalar schema accepted; forbidden fields absent",
    }
    _write_once(Path(args.receipt), json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
