#!/usr/bin/env python3
"""Build and shard the common safe Phase 7 validation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase7_manifest import (
    build_canonical_validation_manifest,
    split_manifest_rows,
)
from tflt.loopscope.phase7_schema import Phase7ContractError


MODEL_KEYS = ("qwen25_3b", "llama32_3b", "gemma2_2b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the safe common Phase 7 validation manifest.")
    parser.add_argument("--cache-dir", required=True, help="Audited HF datasets cache root.")
    parser.add_argument("--output", required=True, help="Fresh common manifest JSON path.")
    parser.add_argument("--shard-dir", required=True, help="Fresh directory for model-scoped shard manifests.")
    parser.add_argument("--shards", type=int, default=2, help="Number of contiguous shards per model.")
    parser.add_argument(
        "--model-key",
        action="append",
        choices=MODEL_KEYS,
        dest="model_keys",
        help="Model namespace(s) for shard paths; defaults to all frozen models.",
    )
    return parser


def _write_once(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_canonical_validation_manifest(cache_dir=args.cache_dir)
    shards = split_manifest_rows(rows, args.shards)
    model_keys = tuple(args.model_keys or MODEL_KEYS)
    output = Path(args.output)
    shard_dir = Path(args.shard_dir)
    if shard_dir.exists() and any(shard_dir.iterdir()):
        raise Phase7ContractError("shard directory is not fresh: %s" % shard_dir)
    _write_once(output, rows)
    shard_paths = []
    for model_key in model_keys:
        model_dir = shard_dir / model_key
        for shard_index, shard in enumerate(shards):
            path = model_dir / ("shard-%02d.json" % shard_index)
            _write_once(path, shard)
            shard_paths.append(str(path))
    print(
        json.dumps(
            {
                "status": "PASS",
                "record_count": len(rows),
                "subject_count": len({row["subject"] for row in rows}),
                "shard_count": len(shards),
                "model_keys": list(model_keys),
                "output": str(output),
                "shards": shard_paths,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
