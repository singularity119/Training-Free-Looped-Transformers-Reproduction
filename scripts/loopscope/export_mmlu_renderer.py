#!/usr/bin/env python3
"""Export an independently reproducible lm-eval 0.4.11 MMLU projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from tflt.loopscope.mmlu_renderer import (
    PHASE1_TARGET_SPLIT,
    RendererVerificationError,
    create_renderer_bundle,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the actual lm-eval==0.4.11 MMLU tasks at an exact dataset commit, "
            "render fixed five-shot contexts, and write a self-verifying projection."
        )
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--target-split", default=PHASE1_TARGET_SPLIT)
    parser.add_argument("--fewshot-split", default="dev")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--task-group", default="mmlu")
    parser.add_argument("--task-name", action="append")
    parser.add_argument("--max-targets-per-task", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, backend: Optional[Any] = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output_jsonl).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if output_path == manifest_path:
        raise RendererVerificationError("projection and manifest paths must differ")
    if not args.dry_run:
        for path in (output_path, manifest_path):
            if path.exists():
                raise FileExistsError("refusing to overwrite renderer artifact: %s" % path)
    bundle = create_renderer_bundle(
        dataset_revision=args.dataset_revision,
        target_split=args.target_split,
        fewshot_split=args.fewshot_split,
        seed=args.seed,
        task_group=args.task_group,
        task_names=args.task_name,
        max_targets_per_task=args.max_targets_per_task,
        backend=backend,
    )
    if args.dry_run:
        print(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new_bytes(output_path, bundle["projection_bytes"])
    try:
        manifest_bytes = (
            json.dumps(
                bundle["manifest"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_new_bytes(manifest_path, manifest_bytes)
    except Exception:
        print(
            "projection was written but manifest creation failed; preserving it for audit: %s"
            % output_path,
            file=sys.stderr,
        )
        raise
    print(str(output_path))
    print(str(manifest_path))
    print(bundle["manifest"]["manifest_sha256"])
    return 0


def _write_new_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RendererVerificationError, FileExistsError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
