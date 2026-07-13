#!/usr/bin/env python3
"""Verify all eight write-once Gate E shard artifacts and joins."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional

from tflt.looppilot.full import verify_three_arm_rows
from tflt.looppilot.schema import write_json_once


REQUIRED = (
    "baseline_a_samples.jsonl", "baseline_b_samples.jsonl", "always_loop_samples.jsonl",
    "signal_records.jsonl", "decisions.jsonl", "join_report.json", "runtime.json", "results_manifest.txt",
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.run_root).resolve()
    reports = []
    all_keys = set()
    for shard_id in range(8):
        shard = root / ("shards/shard-%d" % shard_id)
        if not shard.is_dir():
            raise RuntimeError("missing Gate E shard %d" % shard_id)
        missing = [name for name in REQUIRED if not (shard / name).is_file()]
        if missing:
            raise RuntimeError("Gate E shard %d missing files: %s" % (shard_id, missing))
        _verify_manifest(shard / "results_manifest.txt", shard)
        report = verify_three_arm_rows(
            _jsonl(shard / "baseline_a_samples.jsonl"),
            _jsonl(shard / "baseline_b_samples.jsonl"),
            _jsonl(shard / "always_loop_samples.jsonl"),
            _jsonl(shard / "signal_records.jsonl"),
            _jsonl(shard / "decisions.jsonl"),
        )
        keys = {row["doc_key"] for row in _jsonl(shard / "baseline_a_samples.jsonl")}
        overlap = all_keys.intersection(keys)
        if overlap:
            raise RuntimeError("cross-shard duplicate docs: %s" % sorted(overlap)[:3])
        all_keys.update(keys)
        reports.append({"shard_id": shard_id, "manifest_sha256": _sha(shard / "results_manifest.txt"), **report})
    expected = len(_jsonl(root / "preflight/document_manifest.jsonl"))
    if len(all_keys) != expected:
        raise RuntimeError("Gate E aggregate doc count mismatch")
    write_json_once(Path(args.output), {"schema_version": 1, "shard_count": 8, "document_count": expected, "cross_shard_unique": True, "reports": reports})
    print(json.dumps({"shards": 8, "documents": expected}, sort_keys=True))
    return 0


def _verify_manifest(path: Path, cwd: Path) -> None:
    for line in path.read_text().splitlines():
        digest, name = line.split(None, 1)
        target = Path(name.strip())
        if not target.is_absolute():
            target = cwd / target
        if _sha(target) != digest:
            raise RuntimeError("shard manifest mismatch: %s" % target)


def _jsonl(path: Path) -> List[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
