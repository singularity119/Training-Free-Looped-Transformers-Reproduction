#!/usr/bin/env python3
"""Launch only the frozen LoopScope Phase 7 Gate F outcome panel."""

from __future__ import annotations

import os
os.environ["LOOPSCOPE_PHASE7_OUTCOME_PROFILE"] = "gate_f"

from run_phase7_gate_e_panel import main


if __name__ == "__main__":
    raise SystemExit(main())
