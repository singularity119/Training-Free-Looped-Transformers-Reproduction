#!/usr/bin/env python3
"""Prepare native ARC25-shot gold-free inputs from existing offline CPU cache."""
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase10_data import build_arc_inputs, write_arc_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--data-dir", help="Existing local train/test/validation*.parquet only")
    parser.add_argument("--source-manifest", help="JSON split -> revision-bound source URL list; required with --data-dir")
    parser.add_argument("--output-dir", required=True, help="Fresh, write-once dedicated input root")
    args = parser.parse_args()
    if Path(args.output_dir).exists():
        raise FileExistsError("output root already exists")
    if args.data_dir and not args.source_manifest:
        parser.error("--data-dir requires --source-manifest")
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    sources = json.loads(Path(args.source_manifest).read_text(encoding="utf-8")) if args.source_manifest else None
    bundle = build_arc_inputs(config, args.dataset_revision, args.cache_dir, args.max_length,
                              data_dir=args.data_dir, source_manifest=sources)
    write_arc_inputs(bundle, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "target_gold_loaded": False,
                      "counts": {k: len(v) for k, v in bundle["rows_by_split"].items()},
                      "length_statistics": bundle["length_statistics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
