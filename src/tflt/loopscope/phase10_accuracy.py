"""Gold-free input, selection, and complete-candidate score semantics."""
import json
import math
from pathlib import Path


SCOPES = ("PREFLIGHT_ONLY", "FORMAL_TEST")


def engineering_cell(cell, check, scope):
    """Explicit preflight controls, never configurations eligible for formal closure."""
    result = dict(cell)
    if check is None:
        return result
    require(scope == "PREFLIGHT_ONLY" and cell["arm"] == "Online",
            "engineering controls are Online preflight only")
    require(check in ("zero-strength", "k2-policy"), "unknown engineering control")
    if check == "zero-strength":
        result["strength"] = 0.0
    else:
        require(cell["k"] == 2 and cell["direction_policy"] == "fixed_t0",
                "K2 policy equivalence control requires a canonical fixed_t0 K2 cell")
        result["direction_policy"] = "lag1"
    result["cell_id"] += "-engineering-" + check
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json_once(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def row_identity(row, dataset):
    return row["identity"] if dataset == "mmlu" else row["sample_id"]


def choices_for_row(row, dataset):
    return list("ABCD") if dataset == "mmlu" else list(row["choices"])


def validate_mmlu_preflight_pool(pool):
    revision = "c30699e8356da336a370243923dbaf21066bb9fe"
    require(pool.get("schema_version") == "loopscope.phase10.mmlu_preflight_inputs.v1" and
            pool.get("status") == "PREFLIGHT_ONLY_GOLD_FREE" and
            pool.get("dataset") == {"repo": "cais/mmlu", "revision": revision, "split": "validation"},
            "unexpected MMLU preflight validation scope")
    evidence = pool.get("source_evidence", {})
    counts = evidence.get("subject_counts", {})
    rows = pool.get("rows", [])
    require(evidence.get("target_gold_loaded") is False and
            evidence.get("loaded_splits") == ["validation", "dev"] and
            len(counts) == 57 and sum(counts.values()) == len(rows) == 1531,
            "MMLU validation source/count does not close")
    require([(r["subject"], r["doc_index"]) for r in rows] ==
            [(s, i) for s in sorted(counts) for i in range(counts[s])],
            "MMLU validation canonical order differs")
    fields = {"identity", "subject", "doc_index", "split", "question", "choices", "prompt", "prompt_token_lengths"}
    models = {"Qwen/Qwen3-4B-Base", "Qwen/Qwen3-1.7B-Base"}
    for row in rows:
        require(set(row) == fields and row["split"] == "validation" and
                row["identity"] == f"cais/mmlu@{revision}:validation:{row['subject']}:{row['doc_index']}" and
                len(row["choices"]) == 4 and set(row["prompt_token_lengths"]) == models and
                all(type(n) is int and n > 0 for n in row["prompt_token_lengths"].values()),
                "MMLU validation identity/safe fields/tokenization differs")
    require(pool.get("validation_identities") == [row["identity"] for row in rows],
            "MMLU validation identity list differs")


def load_pool(path, dataset, scope):
    if dataset == "mmlu":
        from .phase8_test_pool import validate_test_pool
        pool = json.loads(Path(path).read_text(encoding="utf-8"))
        if pool.get("schema_version") == "loopscope.phase10.mmlu_preflight_inputs.v1":
            require(scope == "PREFLIGHT_ONLY", "validation inputs cannot enter formal scoring")
            validate_mmlu_preflight_pool(pool)
        else:
            validate_test_pool(pool)
        rows = list(pool["rows"])
    elif dataset == "arc_challenge":
        contents = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix == ".json":
            pool = json.loads(contents)
            require(pool.get("schema_version") == "loopscope.phase10.arc_inputs.v1" and
                    pool.get("status") == "ARC_TARGET_GOLD_FREE", "unexpected ARC pool schema")
            require(pool.get("seed") == 20261002 and pool.get("num_fewshot") == 25,
                    "ARC sampler recipe differs from frozen contract")
            require(pool.get("source_evidence", {}).get("target_gold_loaded") is False,
                    "ARC pool is not gold-free")
            rows = list(pool["rows"])
        else:
            rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
        allowed = {"sample_id", "task", "split", "source_index", "question", "choices",
                   "choice_labels", "prompt", "fewshot_sample_ids", "tokenization", "identity",
                   "subject", "doc_index", "source_id", "candidate_text_lengths", "prompt_token_lengths"}
        for row in rows:
            require(set(row).issubset(allowed), "ARC input contains fields outside the gold-free projection")
            require(row.get("task") == "arc_challenge" and row.get("split") in ("test", "validation"), "unexpected ARC task/split")
            require(scope == "PREFLIGHT_ONLY" or row["split"] == "test", "formal ARC scoring requires test")
            require(isinstance(row.get("choices"), list) and len(row["choices"]) >= 2 and
                    all(isinstance(choice, str) and choice for choice in row["choices"]), "ARC requires complete choice texts")
            require(len(row.get("choice_labels", [])) == len(row["choices"]), "ARC choice labels do not close")
            require(len(row.get("fewshot_sample_ids", [])) == 25, "ARC requires frozen 25-shot sample identities")
            require(isinstance(row.get("tokenization"), dict) and row["tokenization"], "ARC requires frozen tokenization")
            require(all(item.get("candidate_prefix_consistent") is True for item in row["tokenization"].values()), "ARC candidate-specific truncation requires planning decision")
    else:
        raise ValueError("Phase 10 dataset must be mmlu or arc_challenge")
    require(rows and all(isinstance(row.get("prompt"), str) and row["prompt"] for row in rows), "empty/missing rendered prompts")
    keys = [json.dumps(row_identity(row, dataset), sort_keys=True) for row in rows]
    require(len(set(keys)) == len(keys), "duplicate canonical input identities")
    if scope == "FORMAL_TEST":
        require(len(rows) == {"mmlu": 14042, "arc_challenge": 1172}[dataset], "formal pool count differs from frozen full test")
    return rows


def select_indices(rows, cell, scope, explicit=None):
    require(scope in SCOPES, "invalid score scope")
    if explicit is not None:
        indices = list(explicit)
        require(indices and indices == sorted(set(indices)) and
                all(type(index) is int and 0 <= index < len(rows) for index in indices), "selection indices must be sorted unique canonical indices")
        require(scope == "PREFLIGHT_ONLY", "formal producer must score the full frozen test")
        return indices
    if scope == "FORMAL_TEST":
        return list(range(len(rows)))
    def length(index):
        row = rows[index]
        if cell["dataset"] == "mmlu":
            return row["prompt_token_lengths"][cell["model"]]
        metadata = row["tokenization"]
        item = metadata.get(cell["model"])
        if item is None:
            prefix = "q4" if "4B" in cell["model"] else "q17"
            item = metadata[prefix]
        return item["context_length"]
    chosen = set(range(min(2, len(rows))))
    for index in sorted(range(len(rows)), key=lambda index: (-length(index), index)):
        chosen.add(index)
        if len(chosen) >= min(6, len(rows)):
            break
    return sorted(chosen)


def finite_scores(values, candidate_count):
    scores = [float(value[0] if isinstance(value, (list, tuple)) else value) for value in values]
    require(len(scores) == candidate_count and all(math.isfinite(score) for score in scores), "score record must contain every finite raw candidate score")
    return scores


def score_record(row, dataset, values, positions):
    choices = choices_for_row(row, dataset)
    require(len(positions) == len(choices), "candidate token metadata is incomplete")
    lengths = [int(item["continuation_length"]) for item in positions]
    require(all(length > 0 for length in lengths), "empty continuation tokenization")
    result = {"identity": row_identity(row, dataset),
              "scores": finite_scores(values, len(choices)),
              "choice_text_lengths": [len(choice) for choice in choices],
              "continuation_token_lengths": lengths}
    if dataset == "mmlu":
        result.update(subject=row["subject"], doc_index=row["doc_index"])
    return result


def check_position_metadata(positions, tokenizer):
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    if getattr(tokenizer, "pad_token_id", None) is not None:
        special.add(tokenizer.pad_token_id)
    prefixes = set()
    for item in positions:
        p, actual = item["position"], tuple(item["actual_tokens"])
        valid = tuple(item["valid_positions"])
        effective = tuple(item.get("effective_valid_positions", valid))
        require(valid and valid == tuple(sorted(set(valid))) and p in valid, "invalid pre-answer mask")
        require(effective and p in effective and set(effective).issubset(valid), "invalid effective prefix mask")
        require(all(0 <= index <= p and actual[index] not in special for index in valid), "special/padding/continuation token in SVD mask")
        require(len(actual) == item["input_length"] == p + item["continuation_length"], "inconsistent answer boundary")
        require(tuple(item["prefix_tokens"]) == actual[:p + 1], "prefix metadata differs from actual input")
        prefixes.add(actual[:p + 1])
    require(positions and len(prefixes) == 1, "candidate-specific retained prefix requires planning decision")
    return {"request_count": len(positions), "max_input_length": max(item["input_length"] for item in positions),
            "left_truncated_requests": sum(item["left_truncated_tokens"] > 0 for item in positions),
            "multitoken_requests": sum(item["continuation_length"] > 1 for item in positions)}
