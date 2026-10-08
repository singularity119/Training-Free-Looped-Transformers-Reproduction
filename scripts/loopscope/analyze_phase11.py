#!/usr/bin/env python3
"""Re-close a complete Phase11 task, then read its sealed gold and analyze."""
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase11_analysis import analyze_closed_panel
from tflt.loopscope.phase11_panel import validate_score_manifest, write_json_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cell-roots", required=True, help="JSON cell_id -> run-root list")
    parser.add_argument("--gold", required=True, help="Sealed JSON rows identity/label_index/category; GPQA gold_letter also accepted")
    parser.add_argument("--output-dir", required=True, help="Fresh analysis directory")
    parser.add_argument("--dry-run", action="store_true", help="Read manifest only, keeping raw records and gold unread")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("analysis output root already exists")
    panel = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_score_manifest(panel, scope="FORMAL_TEST")
    if args.dry_run:
        print(json.dumps({"dataset": panel["dataset"], "families": [30, 12, 2],
                          "target_gold_loaded": False, "output_dir": str(output)}))
        return
    pool = json.loads(Path(panel["pool"]).read_text(encoding="utf-8"))
    roots = json.loads(Path(args.cell_roots).read_text(encoding="utf-8"))
    report = analyze_closed_panel(panel, pool, roots, args.gold)
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / "analysis.json", report)
    print(json.dumps({"status": report["status"], "dataset": report["dataset"],
                      "sample_count": report["sample_count"], "output_dir": str(output)}))


if __name__ == "__main__":
    main()
