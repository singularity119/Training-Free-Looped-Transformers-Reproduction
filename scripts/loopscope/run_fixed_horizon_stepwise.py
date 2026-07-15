#!/usr/bin/env python3
"""Standalone entrypoint for LoopScope Phase 2 Gate D."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="family", required=True)
    probe = sub.add_parser("probe", help="D1/D2 producer and noninterference commands")
    probe.add_argument("args", nargs=argparse.REMAINDER)
    analysis = sub.add_parser("analysis", help="D3 source closure and D4 atlas/verifier commands")
    analysis.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    if parsed.family == "probe":
        from tflt.loopscope.phase2_fixed_horizon import main as probe_main

        return probe_main(parsed.args)
    from tflt.loopscope.phase2_fixed_horizon_analysis import main as analysis_main

    return analysis_main(parsed.args)


if __name__ == "__main__":
    raise SystemExit(main())

