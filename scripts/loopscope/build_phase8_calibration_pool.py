#!/usr/bin/env python3
"""Export the frozen Gate B validation512 using the offline safe renderer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tflt.loopscope.phase8_pool import (
    DATASET_REPO, DATASET_REVISION, SEED,
    build_calibration_pool, write_calibration_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/loopscope/phase8_model_windows.json")
    parser.add_argument("--gate-b-identities", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE"))
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.model_config).read_text())
    if args.dry_run:
        print(json.dumps({"dataset": DATASET_REPO, "revision": DATASET_REVISION,
                          "seed": SEED, "calibration_identity_count": 512,
                          "gate_b_identities": args.gate_b_identities,
                          "splits": ["validation", "dev"], "offline": True,
                          "model_forward": False, "status": "FORMAL_CALIBRATION",
                          "tokenizers": [{"model": model["model"], "revision": model["observed_revision"]}
                                         for model in config["models"]]}, indent=2))
        return
    if not args.cache_dir or not args.output_dir:
        parser.error("--cache-dir (or HF_DATASETS_CACHE) and --output-dir are required")
    identities = json.loads(Path(args.gate_b_identities).read_text())
    bundle = build_calibration_pool(config, args.cache_dir, identities)
    bundle["source_evidence"]["gate_b_identities_path"] = str(Path(args.gate_b_identities))
    bundle["source_evidence"]["gate_b_identity_order_match"] = True
    write_calibration_pool(bundle, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "calibration_count": len(bundle["rows"])}))


if __name__ == "__main__":
    main()
