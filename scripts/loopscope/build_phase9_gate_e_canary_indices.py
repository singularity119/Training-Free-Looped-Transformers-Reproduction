#!/usr/bin/env python3
"""Build the label-free representative Gate E formal canary bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase8_test_pool import validate_test_pool
from tflt.loopscope.phase9_gate_e_accuracy import (
    canary_bundle,
    load_config,
    validate_canary_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    validate_test_pool(pool)
    config = load_config(args.config)
    value = canary_bundle(pool, config, args.count)
    validate_canary_bundle(value, pool, config, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PHASE9_GATE_E_CANARY_INDICES_READY",
        "count": value["count"],
        "models": {key: len(indices) for key, indices in value["model_indices"].items()},
        "target_gold_loaded": value["target_gold_loaded"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
