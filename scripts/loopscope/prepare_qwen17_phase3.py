#!/usr/bin/env python3
"""Materialize or verify the local-only Phase 3 P3-A producer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from tflt.loopscope.schema import SchemaError, canonical_json_bytes, write_new_json
from tflt.loopscope.phase3_pool import make_source_contract
from tflt.loopscope.phase3_schema import (
    load_phase3_card,
    pool_record_json_schema,
    trajectory_record_json_schema,
)
from tflt.loopscope.phase3_verifier import (
    POOL_SCHEMA_RELATIVE,
    SOURCE_CONTRACT_RELATIVE,
    TRAJECTORY_SCHEMA_RELATIVE,
    verify_committed_receipt,
    write_new_verifier_receipt,
)


RECEIPT_RELATIVE = "configs/loopscope/phase3_card_verifier_receipt.json"


def expected_contract_artifacts(card_path: Path) -> Dict[str, Mapping[str, Any]]:
    card = load_phase3_card(Path(card_path))
    return {
        SOURCE_CONTRACT_RELATIVE: make_source_contract(card),
        POOL_SCHEMA_RELATIVE: pool_record_json_schema(),
        TRAJECTORY_SCHEMA_RELATIVE: trajectory_record_json_schema(),
    }


def write_contract(card_path: Path) -> Dict[str, Any]:
    card_path = Path(card_path).resolve()
    root = card_path.parents[2]
    artifacts = expected_contract_artifacts(card_path)
    for relative, payload in artifacts.items():
        write_new_json(root / relative, payload)
    return write_new_verifier_receipt(card_path, root / RECEIPT_RELATIVE)


def verify_contract(card_path: Path) -> Dict[str, Any]:
    card_path = Path(card_path).resolve()
    root = card_path.parents[2]
    for relative, expected in expected_contract_artifacts(card_path).items():
        observed = _load_json(root / relative)
        if canonical_json_bytes(observed) != canonical_json_bytes(expected):
            raise SchemaError("committed Phase 3 contract artifact differs: %s" % relative)
    return verify_committed_receipt(card_path, root / RECEIPT_RELATIVE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write once or verify the pure-Python LoopScope Phase 3 P3-A contract; "
            "no dataset/model/network action is performed."
        )
    )
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--card", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "write":
        receipt = write_contract(Path(args.card))
    else:
        receipt = verify_contract(Path(args.card))
    print(receipt["manifest_sha256"])
    return 0


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise SchemaError("contract artifact must contain an object: %s" % path)
    return value


def _bad_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


if __name__ == "__main__":
    raise SystemExit(main())
