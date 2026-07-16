#!/usr/bin/env python3
"""Verify the P4-A contract; never load a model, dataset, network, or outcome."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from tflt.loopscope.phase4_schema import Phase4ContractError, load_phase4_card
from tflt.loopscope.phase4_verifier import verify_committed_receipt


RECEIPT_RELATIVE = "configs/loopscope/phase4_card_verifier_receipt.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the local-only LoopScope Phase 4 card and receipt"
    )
    parser.add_argument("mode", choices=("verify",))
    parser.add_argument("--card", required=True)
    parser.add_argument("--allow-unresolved-provenance", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    card_path = Path(args.card).resolve()
    root = card_path.parents[2]
    receipt = verify_committed_receipt(card_path, root / RECEIPT_RELATIVE)
    load_phase4_card(card_path, require_provenance_closed=not args.allow_unresolved_provenance)
    if not receipt["card"]["provenance_closed"] and not args.allow_unresolved_provenance:
        raise Phase4ContractError("Phase 4 exact provenance is not closed")
    print(receipt["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
