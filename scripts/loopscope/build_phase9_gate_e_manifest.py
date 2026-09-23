#!/usr/bin/env python3
"""Build the write-once Phase 9 Gate E Lag1 score manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase9_gate_e_accuracy import load_config, score_manifest, validate_score_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    value = score_manifest(config, args.pool, args.scope)
    validate_score_manifest(value, config, args.pool, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PHASE9_GATE_E_SCORE_MANIFEST_READY",
        "cell_count": value["cell_count"],
        "sample_count_per_cell": value["sample_count_per_cell"],
        "new_configuration_sample_records": value["new_configuration_sample_records"],
        "direction_policies": value["direction_policies"],
        "intervention_modes": value["intervention_modes"],
        "target_gold_loaded": value["target_gold_loaded"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
