#!/usr/bin/env python3
"""Fresh-process verifiers for the Phase 7 Gate E outcome barrier and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase7_outcome import (  # noqa: E402
    Phase7OutcomeError,
    verify_analysis,
    verify_preoutcome,
    verify_static,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7 Gate E fresh-process verifier.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("static", "Verify frozen card, static cell panel, and canonical membership."),
        ("preoutcome", "Verify all completed cells without computing accuracy."),
        ("analysis", "Independently recompute the one combined paired analysis."),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--run-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "static":
            result = verify_static(Path(args.run_root))
        elif args.command == "preoutcome":
            result = verify_preoutcome(Path(args.run_root))
        else:
            result = verify_analysis(Path(args.run_root))
    except Phase7OutcomeError as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
