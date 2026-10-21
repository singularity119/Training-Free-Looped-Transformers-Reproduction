#!/usr/bin/env python3
"""Cloud-only D9 GPQA preparation from the original fixed author archive.

Read the original CSV member anonymously in the cloud, reuse the frozen GPQA
builder and D8 thinking renderer, and seal answer-role information separately.
No model forward runs here. Source-contract reuse is not HPC artifact readback.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

from tflt.loopscope.phase11_d8 import COUNT, MAX_LENGTH, engineering_pool, prepare_pool
from tflt.loopscope.phase11_data import (
    GPQA_COLUMNS, GPQA_GITHUB_COMMIT, GPQA_GITHUB_REPO, GPQA_ARCHIVE_MEMBER,
    GPQA_ARCHIVE_URL, GPQA_SEED, SCHEMA, gpqa_source_provenance, prepare_gpqa_inputs,
)
from tflt.loopscope.phase11_panel import require

SOURCE = {"source_kind": "author_github_archive", "github_repo": GPQA_GITHUB_REPO,
    "github_commit": GPQA_GITHUB_COMMIT, "archive_member": GPQA_ARCHIVE_MEMBER,
    "revision": GPQA_GITHUB_COMMIT, "split": "main_csv", "format": "csv", "name": "gpqa_main"}


class AuthorRedirect(urllib.request.HTTPRedirectHandler):
    """Keep anonymous acquisition at the fixed author distribution host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        require(target.scheme == "https" and target.netloc == "raw.githubusercontent.com",
                "GPQA author archive redirected outside the authorized host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def project_archive(payload):
    """Read just the accepted member and five current construction fields."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read(GPQA_ARCHIVE_MEMBER).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    require(reader.fieldnames is not None and set(GPQA_COLUMNS).issubset(reader.fieldnames),
            "GPQA author CSV lacks the five original construction columns")
    rows = [{key: row[key] for key in GPQA_COLUMNS} for row in reader]
    require(len(rows) == COUNT, "D9 requires all original 448 GPQA rows")
    require(all(isinstance(value, str) and value for row in rows for value in row.values()),
            "GPQA original construction fields are incomplete")
    return rows


def fetch_author_rows():
    # No HF client, token, cookie jar, or authorization header participates.
    opener = urllib.request.build_opener(AuthorRedirect())
    with opener.open(GPQA_ARCHIVE_URL, timeout=60) as response:
        return project_archive(response.read())


def build_inputs(raw_rows, tokenizer):
    require(len(raw_rows) == COUNT, "D9 requires all original 448 GPQA rows")
    require(all(set(row) == set(GPQA_COLUMNS) and
                all(isinstance(value, str) and value for value in row.values()) for row in raw_rows),
            "GPQA builder requires only the five complete original construction fields")
    # The existing producer owns the one canonical-order RNG stream and gold.
    rows, sealed = prepare_gpqa_inputs(raw_rows, tokenizer, GPQA_GITHUB_COMMIT,
                                      "main_csv", MAX_LENGTH, source=SOURCE)
    for row in rows:
        for field in ("prompt", "input_ids", "prompt_token_length"):
            del row[field]
    source_alignment = {"canonical_source_reused": True,
        "basis": "same_fixed_author_archive_and_frozen_builder",
        "github_commit": GPQA_GITHUB_COMMIT, "member": GPQA_ARCHIVE_MEMBER,
        "hpc_artifact_readback_performed": False}
    pool = {"schema_version": SCHEMA, "task": "gpqa_main",
        "dataset_revision": GPQA_GITHUB_COMMIT, "split": "main_csv", "scope": "FORMAL_TEST",
        "engineering_synthetic": False, "target_gold_loaded": False, "rows": rows,
        "test_identities": [row["identity"] for row in rows],
        **gpqa_source_provenance(SOURCE), "source_format": "csv",
        "d9_source_alignment": source_alignment}
    thinking = prepare_pool(pool, tokenizer)
    receipt = {"schema": "loopscope.phase11.d9.input_alignment.v1",
        "status": "SOURCE_CONTRACT_REUSED", "source_url": GPQA_ARCHIVE_URL,
        "source_kind": "author_github_archive", "github_repo": GPQA_GITHUB_REPO,
        "github_commit": GPQA_GITHUB_COMMIT, "archive_member": GPQA_ARCHIVE_MEMBER,
        "input_seed": GPQA_SEED, "count": COUNT, "canonical_indices": "0..447",
        "canonical_order": "fixed author archive member CSV row order",
        "option_builder": "tflt.loopscope.phase11_data.prepare_gpqa_inputs",
        "thinking_builder": "tflt.loopscope.phase11_d8.prepare_pool",
        "hpc_artifact_readback_performed": False, "model_inputs_contain_answer_roles": False,
        "gold_location": "sealed/gpqa_main-gold.jsonl"}
    return thinking, sealed, receipt


def build_synthetic_inputs(tokenizer):
    original = engineering_pool({"scope": "PREFLIGHT_ONLY", "engineering_synthetic": True,
                                "task": "gpqa_main", "target_gold_loaded": False})
    pool = prepare_pool(original, tokenizer)
    receipt = {"schema": "loopscope.phase11.d9.input_alignment.v1",
        "status": "ENGINEERING_SYNTHETIC", "count": 1, "scope": "PREFLIGHT_ONLY",
        "source_dataset_read": False, "gold_created": False,
        "builder": "tflt.loopscope.phase11_d8.engineering_pool + prepare_pool"}
    return pool, None, receipt


def write_outputs(root, pool, gold, receipt):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    with (root / "thinking-pool.json").open("x", encoding="utf-8") as handle:
        json.dump(pool, handle, ensure_ascii=False)
        handle.write("\n")
    if gold is not None:
        (root / "sealed").mkdir(mode=0o700)
        with (root / "sealed/gpqa_main-gold.jsonl").open("x", encoding="utf-8") as handle:
            for row in gold:
                handle.write(json.dumps(row) + "\n")
    with (root / "alignment.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Pre-admitted local Qwen3-4B checkpoint/tokenizer directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Fresh cloud-only input root")
    parser.add_argument("--synthetic", action="store_true",
                        help="Original D8 engineering fixture only; no dataset access or gold")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, token=False)
    pool, gold, receipt = (build_synthetic_inputs(tokenizer) if args.synthetic else
                           build_inputs(fetch_author_rows(), tokenizer))
    receipt.update(model_dir=str(args.model_dir))
    write_outputs(args.output_dir, pool, gold, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
