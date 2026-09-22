#!/usr/bin/env python3
"""Build the Phase 9 47-cell logical manifest without scores or outcomes."""

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase9_accuracy import load_config, manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="frozen Phase 9 model/window config")
    parser.add_argument("--output", type=Path, required=True, help="write-once manifest path")
    args = parser.parse_args()
    config = load_config(args.config)
    value = manifest(config, args.config)
    write_manifest(args.output, value)
    print(json.dumps({key: value[key] for key in (
        "schema", "cell_count", "historical_cell_count", "new_cell_count",
        "native_cell_count", "target_gold_loaded",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
