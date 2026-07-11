#!/usr/bin/env python3
"""Validate the frozen Gate A config without torch or transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


EXPECTED = {
    "model": "Qwen/Qwen3-1.7B-Base",
    "task": "mmlu",
    "num_fewshot": 5,
    "dtype": "float16",
    "window": [12, 15],
    "k": 2,
    "iteration_mode": "block",
    "strategy": "damped_euler",
    "alpha": 1.0,
    "beta": 0.0,
    "cache_strategy": "last",
    "decode_mode": "bypass",
    "actions": ["BASELINE", "LOOP_K2"],
    "batch_size": 1,
}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/looppilot/qwen17_mmlu_phase1.json")
    args = parser.parse_args(argv)
    path = Path(args.config)
    payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in EXPECTED.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError("phase-one config mismatch: %s" % json.dumps(mismatches, sort_keys=True))
    for key in ("model_revision", "tokenizer_revision"):
        revision = payload.get(key)
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("%s must be a resolved 40-character commit" % key)
    if payload.get("batch_size") != 1:
        raise ValueError("batch_size must be frozen to 1 for Gate C")
    print(hashlib.sha256(path.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
