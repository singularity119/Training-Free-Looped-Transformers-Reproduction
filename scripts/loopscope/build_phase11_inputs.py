#!/usr/bin/env python3
"""Build Phase11 inputs on remote CPU from existing revision-bound parquet."""
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase11_data import COUNTS, build_phase11_inputs, write_phase11_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-file", required=True, help="Question-free JSON source/tokenizer binding")
    parser.add_argument("--output-dir", required=True, help="Fresh dedicated remote input directory")
    parser.add_argument("--task", choices=["all", *COUNTS], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Inspect metadata only; no dataset/tokenizer access")
    args = parser.parse_args()
    if Path(args.output_dir).exists():
        raise FileExistsError("output root already exists")
    binding = json.loads(Path(args.binding_file).read_text(encoding="utf-8"))
    tasks = list(COUNTS) if args.task == "all" else [args.task]
    if args.dry_run:
        print(json.dumps({"tasks": tasks, "counts": {task: binding["tasks"][task]["count"] for task in tasks},
                          "max_length": binding["max_length"], "output_dir": args.output_dir,
                          "dataset_access": False, "tokenizer_access": False}))
        return
    bundle = build_phase11_inputs(binding, tasks)
    write_phase11_inputs(bundle, args.output_dir)
    # Print only metadata; never print original questions, CoT, or gold mappings.
    print(json.dumps({"output_dir": args.output_dir, "counts": {task: len(rows) for task, rows in bundle["rows_by_task"].items()},
                      "max_prompt_tokens": {task: facts.get("max_prompt_tokens", facts.get("max_context_tokens"))
                                            for task, facts in bundle["task_facts"].items()}}))


if __name__ == "__main__":
    main()
