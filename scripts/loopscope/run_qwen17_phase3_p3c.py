#!/usr/bin/env python3
"""Execute the isolated write-once LoopScope Phase 3 Gate P3-C stages."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    """Avoid importing loop runtime modules as a package side effect."""

    if "tflt" in sys.modules:
        raise RuntimeError("P3-C requires a fresh isolated Python process")
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__file__ = str(package_root)
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase3_p3c import (
    materialize_c0,
    materialize_c1,
    materialize_c2,
    verify_c1_freeze,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--p3b-root", required=True, type=Path)
    parser.add_argument("--p3c-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P3-C only: C0 metadata-safe closure, C1 outcome-blind selector "
            "freeze, independent C1 verification, and C2 historical unseal."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    c0 = sub.add_parser("c0", help="Materialize canonical outcome-safe metadata closure")
    _common(c0)
    c0.add_argument("--phase1-root", required=True, type=Path)
    c0.add_argument("--blind-search-root", required=True, type=Path)

    c1 = sub.add_parser("c1", help="Freeze all outcome-blind selector ranks")
    _common(c1)

    verify = sub.add_parser(
        "verify-c1", help="Independently recompute and verify the frozen selector"
    )
    _common(verify)

    c2 = sub.add_parser(
        "c2", help="Unseal only manifest-bound baseline and historical-13 outcomes"
    )
    _common(c2)
    c2.add_argument("--phase1-root", required=True, type=Path)
    return parser


def _provenance_argv(argv: Optional[Sequence[str]]) -> Sequence[str]:
    suffix = list(sys.argv[1:] if argv is None else argv)
    return [str(Path(__file__).resolve())] + suffix


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "card_path": args.card,
        "p3b_root": args.p3b_root,
        "p3c_root": args.p3c_root,
        "expected_commit": args.expected_commit,
    }
    provenance_argv = _provenance_argv(argv)
    if args.command == "c0":
        result = materialize_c0(
            phase1_root=args.phase1_root,
            blind_search_root=args.blind_search_root,
            argv=provenance_argv,
            **common,
        )
    elif args.command == "c1":
        result = materialize_c1(argv=provenance_argv, **common)
    elif args.command == "verify-c1":
        result = verify_c1_freeze(**common)
    elif args.command == "c2":
        result = materialize_c2(
            phase1_root=args.phase1_root,
            argv=provenance_argv,
            **common,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
