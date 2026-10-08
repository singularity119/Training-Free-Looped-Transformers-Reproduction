#!/usr/bin/env python3
"""Write a fresh Phase 11 outcome-free panel or one task score manifest."""
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase11_panel import DATASETS, SCOPES, build_all_panels, load_config, score_manifest, write_json_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--scope", choices=SCOPES, default="FORMAL_TEST")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    value = score_manifest(args.dataset, args.pool, args.scope, config) if args.dataset else build_all_panels(config)
    if args.dry_run:
        print(json.dumps({"schema": value["schema"], "independent_cell_count": value["independent_cell_count"],
                          "display_row_count": value["display_row_count"], "alias_count": value["alias_count"],
                          "target_gold_loaded": False}, sort_keys=True))
    else:
        parser.error("--output is required unless --dry-run") if args.output is None else write_json_once(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
