#!/usr/bin/env python3
"""Close a complete Phase11 task's 18 raw cells without reading gold."""
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase11_analysis import close_panel, verify_attempt
from tflt.loopscope.phase11_panel import validate_score_manifest, write_json_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cell-roots", help="JSON cell_id -> run-root list; formal shards require exact nonoverlapping indices")
    parser.add_argument("--attempt", help="One synthetic preflight run root; never closes a task panel")
    parser.add_argument("--pool", help="Explicit pool path; must equal manifest.pool")
    parser.add_argument("--scope", choices=("FORMAL_TEST", "PREFLIGHT_ONLY"), default="FORMAL_TEST")
    parser.add_argument("--approved-source-commit", action="append",
                        help="Externally audited compatible source commit; repeat for every allowed revision. Default: one shared revision")
    parser.add_argument("--output-dir", required=True, help="Fresh verification directory")
    parser.add_argument("--dry-run", action="store_true", help="Inspect manifest only; do not read raw records or gold")
    args = parser.parse_args()
    if args.scope == "PREFLIGHT_ONLY":
        if not args.attempt or args.cell_roots:
            parser.error("PREFLIGHT_ONLY requires --attempt and excludes --cell-roots")
    elif not args.cell_roots or args.attempt:
        parser.error("FORMAL_TEST requires --cell-roots and excludes --attempt")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("verification output root already exists")
    panel = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_score_manifest(panel, scope=args.scope)
    if args.pool and args.pool != panel["pool"]:
        parser.error("--pool differs from manifest.pool")
    if args.dry_run:
        print(json.dumps({"dataset": panel["dataset"], "independent_cells": 18,
                          "full_sample_count": panel["sample_count_per_cell"], "scope": args.scope,
                          "target_gold_loaded": False}))
        return
    pool = json.loads(Path(panel["pool"]).read_text(encoding="utf-8"))
    if args.scope == "PREFLIGHT_ONLY":
        closure = verify_attempt(panel, pool, args.attempt, args.scope,
                                 approved_source_commits=args.approved_source_commit)
    else:
        roots = json.loads(Path(args.cell_roots).read_text(encoding="utf-8"))
        closure, _ = close_panel(panel, pool, roots, approved_source_commits=args.approved_source_commit)
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / ("attempt_verification.json" if args.attempt else "closure.json"), closure)
    print(json.dumps({"status": closure["status"], "dataset": closure["dataset"],
                      "sample_count": closure.get("sample_count", closure.get("records_count")),
                      "output_dir": str(output), "target_gold_loaded": False}))


if __name__ == "__main__":
    main()
