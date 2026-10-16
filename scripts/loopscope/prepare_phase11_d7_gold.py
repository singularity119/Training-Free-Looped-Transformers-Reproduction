#!/usr/bin/env python3
"""Project and seal the frozen MMLU-Pro gold only after complete D7 raw/S2 closure.

The projection read in this script is the first target-answer access. The later
analysis JSONL read is not its replacement. No target answer is printed.
"""
import argparse
from pathlib import Path

from analyze_phase11_d7 import COUNT, load_before_gold, now, read, require, write
from tflt.loopscope.phase11_mmlupro_s2 import PARSER_VERSION

REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SOURCE = Path("/hpc2hdd/home/xhuang225/shared/hf_home/hub/"
              "datasets--TIGER-Lab--MMLU-Pro/snapshots/" + REVISION +
              "/data/test-00000-of-00001.parquet")
PROJECTION = ("question_id", "answer_index", "category")


def read_projection(path):
    """Lazy CPU parquet access with only the three authorized columns."""
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    require(all(column in source.schema_arrow.names for column in PROJECTION),
            "D7 original parquet schema differs; report to planning")
    require(source.metadata.num_rows == COUNT, "D7 original parquet population differs")
    return source.read(columns=list(PROJECTION)).to_pylist()


def join_gold(projected, pool_rows, identities):
    """Reconstruct Gate A source-order identities, preserving original options."""
    require(len(projected) == len(pool_rows) == len(identities) == COUNT,
            "D7 projected/canonical gold population differs")
    require(len(set(identities)) == COUNT and
            [r["identity"] for r in pool_rows] == identities, "D7 canonical gold identities differ")
    canonical = {r["identity"]: r for r in pool_rows}
    gold = {}
    for source_index, row in enumerate(projected):
        require(set(row) == set(PROJECTION), "D7 gold projection columns differ")
        identity = f"TIGER-Lab/MMLU-Pro@{REVISION}:test:{source_index}:{row['question_id']}"
        require(identity in canonical and identity not in gold,
                "D7 gold source-row identity missing, unknown or duplicated")
        original = canonical[identity]
        require(row["category"] == original["subject"], "D7 gold category differs from canonical input")
        labels, label = original["choice_labels"], row["answer_index"]
        require(labels == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(original["choices"])]) and
                type(label) is int and 0 <= label < len(labels), "D7 gold outside original option range")
        gold[identity] = {"identity": identity, "label_index": label, "category": row["category"]}
    require(set(gold) == set(identities), "D7 gold exact identity join incomplete")
    return [gold[identity] for identity in identities]


def run(args):
    source = Path(args.source)
    require(source == SOURCE, "D7 gold source path/revision differs from the exact handoff")
    closure = read(args.closure)
    # This must finish before importing the parquet reader or reading any answer.
    data, pool_rows, identities = load_before_gold(closure)
    require(len(identities) == COUNT and closure["record_count"] == COUNT * 5,
            "D7 requires all 60160 raw/S2 predictions before first answer access")
    del data
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = {"schema": "loopscope.phase11.d7.gold_access_receipt.v1",
        "source_closure": str(args.closure), "closure_status": closure["status"],
        "prior_record_count": closure["record_count"], "prior_cell_count": closure["cell_count"],
        "prior_prediction_closed": closure["prediction_closed"], "prior_target_gold_loaded": False,
        "closure_closed_at_utc": closure["closed_at_utc"], "parser_version": PARSER_VERSION,
        "source_revision": REVISION, "source_path": str(source), "projection_columns": list(PROJECTION),
        "source_row_index": "zero-based original parquet row order, never filtered or sorted",
        "producer_sources": {c["cell_key"]: c["source_commit"] for c in closure["cells"]}}
    started = now()
    pre = {**common, "status": "CLOSURE_VERIFIED_BEFORE_FIRST_GOLD_ACCESS",
           "timestamp_utc": started, "target_gold_read": False}
    write(args.output_dir / "pre_gold_access.json", pre)
    try:
        projected = read_projection(source)
        completed = now()
        # Persist actual source-read timing before join/scoring/JSONL access.
        write(args.output_dir / "post_gold_access.json", {**common,
            "status": "FIRST_GOLD_PROJECTION_READ", "target_gold_read": True,
            "projection_started_after_utc": started, "projection_completed_at_utc": completed,
            "projected_row_count": len(projected)})
        gold = join_gold(projected, pool_rows, identities)
        sealed = args.output_dir / "sealed"
        sealed.mkdir(exist_ok=False)
        path = sealed / "gold.jsonl"
        import json
        with path.open("x", encoding="utf-8") as handle:
            for row in gold:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write(args.output_dir / "gold_adapter_receipt.json", {**common,
            "status": "GOLD_PROJECTED_EXACT_JOIN_SEALED", "target_gold_read": True,
            "projection_started_after_utc": started, "projection_completed_at_utc": completed,
            "join_completed_at_utc": now(), "projected_row_count": len(projected),
            "joined_count": len(gold), "missing_count": 0, "duplicate_count": 0,
            "category_match": True, "option_range_valid": True, "gold_path": str(path),
            "gold_fields": ["identity", "label_index", "category"]})
    except Exception as error:
        # Preserve access attempt evidence without exposing labels or row contents.
        write(args.output_dir / "gold_access_failure.json", {**common,
            "status": "GOLD_ADAPTER_FAILED_PRESERVE_ATTEMPT", "timestamp_utc": now(),
            "first_gold_read_attempted": True, "error_type": type(error).__name__})
        raise
    print('{"status":"GOLD_PROJECTED_EXACT_JOIN_SEALED","joined_count":' + str(len(gold)) + '}')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
