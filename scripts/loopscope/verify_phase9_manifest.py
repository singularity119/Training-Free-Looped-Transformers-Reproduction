#!/usr/bin/env python3
"""Verify Phase 9 logical panel membership without reading scores or labels."""

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase9_accuracy import load_config, validate_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    with args.manifest.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema") != "loopscope.phase9.panel.v2":
        raise ValueError("unexpected Phase 9 manifest schema")
    if value.get("sample_count_per_cell") != 14042:
        raise ValueError("manifest sample count differs from the frozen test panel")
    counts = validate_panel(value.get("cells", []), config)
    expected_summary = {
        "cell_count": counts["cell_count"],
        "historical_cell_count": counts["historical_count"],
        "new_cell_count": counts["new_count"],
        "native_cell_count": counts["native_count"],
        "total_configuration_sample_records": counts["cell_count"] * 14042,
        "new_configuration_sample_records": counts["new_count"] * 14042,
    }
    if any(value.get(key) != expected for key, expected in expected_summary.items()):
        raise ValueError("manifest summary differs from frozen panel")
    if value.get("target_gold_loaded") is not False:
        raise ValueError("manifest summary differs from frozen panel")
    print(json.dumps({"status": "MANIFEST_VALID", **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
