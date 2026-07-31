#!/usr/bin/env python3
"""Gate A-only prepare skeleton for the frozen Phase 6 two-pass producer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tflt.loopscope.phase6_schema import file_sha256, load_json, validate_card


DEFAULT_CARD = REPO_ROOT / "configs/loopscope/phase6_pre_answer_v2_card.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and print the Phase 6 two-pass producer contract. "
            "Gate A does not load a model or dataset."
        )
    )
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the frozen plan without generation, replay, or selector execution",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.dry_run:
        raise SystemExit("Gate A skeleton requires --dry-run")
    card = load_json(args.card)
    validate_card(card)
    payload = {
        "status": "DRY_RUN_VALID",
        "card": str(args.card.resolve()),
        "card_sha256": file_sha256(args.card),
        "pass_1": "native_no_loop_deterministic_full_cot_generation",
        "anchor": "literal_regex_match_ordinal_zero_then_preceding_generated_token",
        "eligibility": (
            "zero_match_or_no_preceding_generated_token_is_anchor_not_expressed"
        ),
        "eligibility_mask": "canonical_test_12032_frozen_before_selector",
        "overall_coverage_floor": card["anchor_eligibility"][
            "overall_coverage_floor"
        ],
        "category_coverage_floor": card["anchor_eligibility"][
            "per_category_coverage_floor"
        ],
        "pass_2": (
            "eligible_only_exact_prefix_replay_no_cache_hidden_states_"
            "zero_loop_insertions"
        ),
        "model_or_data_loaded": False,
        "generation_or_replay_executed": False,
        "selector_executed": False,
        "outcomes_read": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
