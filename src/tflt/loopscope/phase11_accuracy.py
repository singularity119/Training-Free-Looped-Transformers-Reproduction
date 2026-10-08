"""Gold-free Phase 11 inputs, strict extraction, and raw-record completeness."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .phase11_panel import COUNTS, DATASETS, MODEL, REVISION, SCOPES, load_config, require, write_json_once

RECORD_SCHEMA = "loopscope.phase11.raw_record.v1"
SAFE_INPUT_FIELDS = {"identity", "sample_id", "task", "split", "source_index", "source_id", "subject",
    "doc_index", "question", "choices", "choice_labels", "fewshot_sample_ids", "prompt", "input_ids",
    "prompt_token_length", "prompt_token_lengths", "tokenization", "candidate_text_lengths"}


def extract_final_answer(text: str, labels: Sequence[str]) -> dict[str, Any]:
    """Accept exactly one whole valid line, with only outer whitespace allowed."""
    require(isinstance(text, str), "generated response must be text")
    labels = list(labels)
    require(labels and len(set(labels)) == len(labels) and
            all(isinstance(label, str) and re.fullmatch("[A-Z]", label) for label in labels),
            "generation labels must be unique uppercase letters")
    matches = []
    for line in text.splitlines():
        match = re.fullmatch(r"Final answer: \(([A-Z])\)", line.strip())
        if match and match.group(1) in labels:
            matches.append(match.group(1))
    unique = len(matches) == 1
    return {"extraction_status": "unique_valid" if unique else
            ("no_valid_final_line" if not matches else "multiple_valid_final_lines"),
            "valid_final_line_count": len(matches), "predicted_label": matches[0] if unique else None,
            "prediction_index": labels.index(matches[0]) if unique else None}


def termination(generated_ids: Sequence[int], eos_ids: Sequence[int], max_new_tokens: int) -> dict[str, Any]:
    ids = list(generated_ids)
    require(type(max_new_tokens) is int and max_new_tokens > 0 and 0 < len(ids) <= max_new_tokens,
            "generation length must be within its declared positive bound")
    require(all(type(token) is int and token >= 0 for token in ids), "generated token IDs are invalid")
    eos = set(eos_ids)
    require(eos and all(type(token) is int and token >= 0 for token in eos), "native EOS/EOT IDs are required")
    ended = ids[-1] in eos
    require(ended or len(ids) == max_new_tokens, "generation stopped before native EOS or the length limit")
    return {"end_reason": "eos" if ended else "length_limit", "truncated": not ended,
            "eos_reached": ended, "generated_token_count": len(ids), "last_token_id": ids[-1],
            "max_new_tokens": max_new_tokens, "eos_token_ids": sorted(eos)}


def _base_record(row: Mapping[str, Any], canonical_index: int) -> dict[str, Any]:
    return {"schema": RECORD_SCHEMA, "identity": row["identity"], "canonical_index": canonical_index,
            "category": row["subject"], "candidate_labels": list(row["choice_labels"]), "status": "OK"}


def generation_record(row: Mapping[str, Any], text: str, generated_ids: Sequence[int], eos_ids: Sequence[int],
                      max_new_tokens: int, canonical_index: int) -> dict[str, Any]:
    value = {**_base_record(row, canonical_index), "kind": "generation", "generated_text": text,
             "generated_token_ids": list(generated_ids), **termination(generated_ids, eos_ids, max_new_tokens),
             **extract_final_answer(text, row["choice_labels"])}
    validate_record(value, row["task"])
    return value


def arc_record(row: Mapping[str, Any], scores: Sequence[Any], positions: Sequence[Mapping[str, Any]],
               canonical_index: int) -> dict[str, Any]:
    raw = [float(value[0] if isinstance(value, (tuple, list)) else value) for value in scores]
    require(len(raw) == len(row["choices"]) == len(positions), "ARC score or candidate metadata is incomplete")
    require(all(math.isfinite(value) for value in raw), "ARC requires all finite candidate loglikelihoods")
    value = {**_base_record(row, canonical_index), "kind": "candidate_loglikelihood", "scores": raw,
        "choice_text_lengths": [len(choice) for choice in row["choices"]],
        "continuation_token_lengths": [int(item["continuation_length"]) for item in positions]}
    validate_record(value, "arc_challenge")
    return value


def failure_record(row: Mapping[str, Any], canonical_index: int, error: Exception) -> dict[str, Any]:
    """Preserve a failed attempt; this record cannot close a scientific panel."""
    return {**_base_record(row, canonical_index), "status": "RUNTIME_FAILED", "error_type": type(error).__name__}


def validate_record(record: Mapping[str, Any], dataset: str) -> None:
    require(dataset in DATASETS, "unknown raw-record task")
    require(record.get("schema") == RECORD_SCHEMA and record.get("status") == "OK",
            "raw record is absent, failed, or uses an unexpected schema")
    require(isinstance(record.get("identity"), str) and bool(record["identity"]) and
            type(record.get("canonical_index")) is int and record["canonical_index"] >= 0 and
            isinstance(record.get("category"), str) and bool(record["category"]),
            "raw sample identity/index/category is invalid")
    labels = record.get("candidate_labels", [])
    require(2 <= len(labels) <= 26 and len(set(labels)) == len(labels), "raw candidate labels do not close")
    if dataset == "arc_challenge":
        require(record.get("kind") == "candidate_loglikelihood", "ARC must use full candidate scoring")
        scores, lengths = record.get("scores", []), record.get("choice_text_lengths", [])
        tokens = record.get("continuation_token_lengths", [])
        require(len(scores) == len(lengths) == len(tokens) == len(labels) and
                all(math.isfinite(float(v)) for v in scores) and
                all(type(v) is int and v > 0 for v in lengths + tokens),
                "ARC raw scores or original character/token lengths are incomplete")
    else:
        require(record.get("kind") == "generation", "MMLU-Pro and GPQA must generate answers")
        parsed = extract_final_answer(record.get("generated_text"), labels)
        require(all(record.get(key) == value for key, value in parsed.items()),
                "saved extraction differs from strict whole-response extraction")
        stop = termination(record.get("generated_token_ids", []), record.get("eos_token_ids", []),
                           record.get("max_new_tokens"))
        require(all(record.get(key) == value for key, value in stop.items()),
                "saved EOS/length classification differs from generated tokens")


def validate_cell_records(dataset: str, records: Sequence[Mapping[str, Any]],
                          identities: Sequence[str]) -> list[dict[str, Any]]:
    identities = list(identities)
    require(identities and len(set(identities)) == len(identities), "canonical identities must be nonempty and unique")
    require(len(records) == len(identities), "cell is missing canonical sample records")
    by_index = {}
    for record in records:
        validate_record(record, dataset)
        index = record["canonical_index"]
        require(index < len(identities) and index not in by_index and record["identity"] == identities[index],
                "cell sample identity is missing, duplicated, or outside canonical order")
        by_index[index] = dict(record)
    require(set(by_index) == set(range(len(identities))), "cell does not cover the canonical population")
    return [by_index[index] for index in range(len(identities))]


def load_pool(path: Path, dataset: str, scope: str, config=None) -> dict[str, Any]:
    require(dataset in DATASETS and scope in SCOPES, "unknown task or acquisition scope")
    config = load_config() if config is None else config
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    require(pool.get("schema_version") == "loopscope.phase11.inputs.v1" and pool.get("task") == dataset and
            pool.get("model") == MODEL and pool.get("model_revision") == REVISION and
            pool.get("tokenizer_revision") == REVISION and pool.get("target_gold_loaded") is False,
            "Phase 11 pool model, tokenizer, task, or gold-free projection differs")
    rows = list(pool.get("rows", []))
    require(rows and all(set(row).issubset(SAFE_INPUT_FIELDS) for row in rows),
            "input pool is empty or contains fields outside the permitted projection")
    identities = [row["identity"] for row in rows]
    require(pool.get("test_identities") == identities and len(set(identities)) == len(identities),
            "input identities differ or repeat")
    for row in rows:
        require(row.get("task") == dataset and isinstance(row.get("prompt"), str) and row["prompt"] and
                isinstance(row.get("subject"), str) and row["subject"] and
                len(row.get("choice_labels", [])) == len(row.get("choices", [])) >= 2,
                "input prompt, category, or choices are incomplete")
        if dataset != "arc_challenge":
            ids = row.get("input_ids", [])
            require(ids and all(type(token) is int and token >= 0 for token in ids) and
                    row.get("prompt_token_length") == len(ids) and len(ids) + 2048 <= config["max_length"],
                    "complete chat prompt tokenization or context capacity differs")
    if scope == "FORMAL_TEST":
        recipe = config["datasets"][dataset]
        require(recipe.get("revision") and recipe.get("split"), "formal task source revision/split has not been bound")
        require(pool.get("scope") == "FORMAL_TEST" and pool.get("engineering_synthetic") is False and
                pool.get("dataset_revision") == recipe["revision"] and pool.get("split") == recipe["split"] and
                len(rows) == COUNTS[dataset] and all(row["split"] == recipe["split"] for row in rows),
                "formal inputs differ from the complete bound population")
        if dataset == "gpqa_main":
            require(all(pool.get(key) == recipe[key] for key in ("source_kind", "github_repo", "github_commit", "archive_member")),
                    "GPQA model pool lacks the distinct accepted author source provenance")
        shots = {"arc_challenge": 25, "mmlu_pro": 5, "gpqa_main": 0}[dataset]
        require(all(len(row["fewshot_sample_ids"]) == shots for row in rows), "formal demonstration identities differ")
    else:
        require(pool.get("scope") == "PREFLIGHT_ONLY" and pool.get("engineering_synthetic") is True and
                all(row["split"] == "synthetic" for row in rows),
                "Phase 11 preflight must use declared synthetic engineering inputs")
    return pool


def select_indices(rows, scope, explicit=None, shard=None):
    require(scope in SCOPES, "unknown selection scope")
    indices = list(range(len(rows))) if shard is None else list(shard)
    if explicit is not None:
        require(scope == "PREFLIGHT_ONLY" and shard is None, "formal selection cannot reduce the frozen population")
        indices = list(explicit)
    require(indices and indices == sorted(set(indices)) and
            all(type(i) is int and 0 <= i < len(rows) for i in indices), "selected canonical indices are invalid")
    return indices


def engineering_cell(cell, check, scope):
    result = copy.deepcopy(cell)
    if check is None:
        return result
    require(scope == "PREFLIGHT_ONLY" and cell["arm"] == "Online", "engineering controls are synthetic Online preflight only")
    require(check in ("zero-strength", "k2-policy"), "unknown engineering control")
    if check == "zero-strength":
        result["strength"] = 0.0
    else:
        require(cell["k"] == 2 and cell["direction_policy"] == "fixed_t0", "K2 alias control needs a canonical fixed_t0 K2 cell")
        result["direction_policy"] = "lag1"
    result["cell_id"] += "-engineering-" + check
    return result
