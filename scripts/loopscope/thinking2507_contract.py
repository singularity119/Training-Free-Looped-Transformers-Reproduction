"""Thinking-2507 record metadata and raw-token lifecycle contract; no runtime imports."""
import json
from pathlib import Path
import random

from tflt.loopscope.phase11_accuracy import validate_record
from tflt.loopscope.phase11_panel import require
from tflt.loopscope.phase11_thinking2507 import (
    MODEL, REVISION, MAX_LENGTH, MAX_NEW_TOKENS, SAMPLING, PARSER, SCORING_REGION,
    actual_seed, answer_span,
)


def metadata(job, cell, pool):
    return {"schema": "loopscope.phase11.thinking2507.job.v1", "cell": cell,
        **{k: job[k] for k in ("cell_key", "cell_id", "seed", "source_commit", "prompt_mode", "damping_scope",
                              "pool", "run_root", "indices", "synthetic", "engineering")},
        "model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
        "max_length": MAX_LENGTH, "max_new_tokens": MAX_NEW_TOKENS, "sampling": SAMPLING,
        "seed_rule": "20261018+canonical_index", "input_seed": 20261008,
        "think_open_id": pool["think_open_id"], "think_close_id": pool["think_close_id"],
        "target_gold_loaded": False, "parser": PARSER, "scoring_region": SCORING_REGION}


def validate_bundle(index, record, telemetry, meta, pool, tokenizer):
    require(meta.get("schema") == "loopscope.phase11.thinking2507.job.v1" and
            meta.get("model") == MODEL and meta.get("revision") == REVISION and
            meta.get("tokenizer_revision") == REVISION and
            meta.get("max_length") == MAX_LENGTH and meta.get("max_new_tokens") == MAX_NEW_TOKENS and
            meta.get("sampling") == SAMPLING and meta.get("parser") == PARSER and
            meta.get("scoring_region") == SCORING_REGION and meta.get("target_gold_loaded") is False,
            "Thinking2507 model/greedy/context/scoring metadata differs")
    validate_record(record, "gpqa_main")
    row, cell = pool["rows"][index], meta["cell"]
    require(record["canonical_index"] == index and record["identity"] == row["identity"] and
            record["candidate_labels"] == row["choice_labels"] and record["category"] == row["subject"],
            "Thinking2507 raw identity/order differs")
    require(record.get("thinking2507_binding") == meta and telemetry.get("thinking2507_binding") == meta and
            record.get("actual_seed") == actual_seed(index) and telemetry.get("actual_seed") == actual_seed(index),
            "Thinking2507 record seed/source/cell differs")
    require(record["generated_text"] == tokenizer.decode(record["generated_token_ids"], skip_special_tokens=False),
            "Thinking2507 complete decoded response differs from token IDs")
    require(record["final_span"] == answer_span(record["generated_token_ids"], row["input_ids"], tokenizer,
            record["eos_token_ids"]), "Thinking2507 final-span extraction differs")
    n = record["generated_token_count"]
    require(record["max_new_tokens"] == MAX_NEW_TOKENS and telemetry.get("prefill_count") == 1 and
            telemetry.get("decode_count") == n - 1 and telemetry.get("prefill_input_tokens") == len(row["input_ids"])
            and telemetry.get("generated_token_count") == n and telemetry.get("cache_growth_closed") is True,
            "Thinking2507 generation/cache lifecycle differs")
    if cell["arm"] == "Online":
        online = telemetry.get("online", {})
        applied = {str(t): int(cell["direction_policy"] == "current_t" or t > 0) for t in range(cell["k"])}
        require(online.get("decode_fit_count") == 0 and online.get("directions_released") is True and
                online.get("forward_count") == n and online.get("callback_count") == n * cell["k"] and
                online.get("callbacks_by_t") == {str(t): n for t in range(cell["k"])} and
                online.get("applied_by_t") == {t: a * n for t, a in applied.items()} and
                online.get("prefill_applied_by_t") == applied and
                online.get("decode_applied_by_t") == {t: a * (n - 1) for t, a in applied.items()} and
                online.get("damping_scope") == "full", "Thinking2507 loop/direction lifecycle differs")
    else:
        require("online" not in telemetry and cell["window"] is None, "Thinking2507 Native contains loop runtime")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_compute(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {k: seed for k in ("python", "numpy", "torch", "cuda_all_devices")}
