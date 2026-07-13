#!/usr/bin/env python3
"""Prepare a write-once Phase 2 H1 V2 manifest without running an experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tflt.loopscope.phase2_reuse import build_phase2_freeze_candidate
from tflt.loopscope.phase2_schema import atomic_write_new_json
from tflt.loopscope.schema import ensure_new_directory


def build_freeze_candidate(card_path: Path) -> Dict[str, Any]:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    return build_phase2_freeze_candidate(card, Path(__file__).resolve().parents[2])


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output_dir = ensure_new_directory(Path(args.output_dir))
    manifest = build_freeze_candidate(Path(args.card))
    atomic_write_new_json(output_dir / "phase2_freeze_candidate.json", manifest)
    print(str(output_dir / "phase2_freeze_candidate.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
