#!/usr/bin/env python3
"""Run the authorized LoopScope Phase 6 Gate M MMLU five-shot path."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase6_mmlu5_gate_m import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
