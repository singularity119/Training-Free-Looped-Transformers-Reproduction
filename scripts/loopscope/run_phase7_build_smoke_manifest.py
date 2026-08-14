#!/usr/bin/env python3
"""Build the common four-identity, outcome-blind Phase 7 smoke manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase7_renderer import build_canonical_smoke_rows
from tflt.loopscope.phase7_schema import Phase7ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the common four-row safe MMLU validation manifest for Phase 7 Gate B."
    )
    parser.add_argument("--output", required=True, help="Fresh JSON manifest path.")
    parser.add_argument("--cache-dir", default=None, help="Audited HF datasets cache root.")
    return parser


def _write_once(path: Path, rows: object) -> None:
    if path.exists():
        raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_canonical_smoke_rows(cache_dir=args.cache_dir)
    _write_once(Path(args.output), rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "record_count": len(rows),
                "subjects": sorted({row["subject"] for row in rows}),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
