"""Thinking-2507 frozen greedy recipe and whole-output S2; no gold access."""
import copy

from .phase11_panel import build_panel, require
from .phase11_s2 import extract_s2

MODEL = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "UNVERIFIED_OSS_COPY"
SEED = 20261018
COUNT = 448
MAX_NEW_TOKENS = 8192
MAX_LENGTH = 262144
CELLS = ("native", "current_t_k2_l05", "fixed_t0_k3_l09")
SAMPLING = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}
PARSER = "D4-whole-output-GPQA-S2-v1"
SCORING_REGION = "whole_generated_output"


def engineering_pool(original):
    require(original.get("engineering_synthetic") is True and original.get("scope") == "PREFLIGHT_ONLY",
            "Thinking2507 engineering fixture cannot replace a formal population")
    identity = "phase11-thinking2507-synthetic:gpqa_main:0"
    row = {"identity": identity, "sample_id": identity, "task": "gpqa_main", "split": "synthetic",
        "source_index": 0, "source_id": 0, "subject": "synthetic", "question": "What is one plus one?",
        "choices": ["two", "three", "four", "five"], "choice_labels": list("ABCD"), "fewshot_sample_ids": []}
    return dict(original, rows=[row], test_identities=[identity])


def selected_cells(config=None):
    originals = {c["cell_id"]: c for c in build_panel("gpqa_main", config)["cells"]}
    suffixes = ("native", "w15-18-k2-online-current-t-lambda0.5",
                "w15-18-k3-online-fixed-t0-lambda0.9")
    result = {}
    for key, suffix in zip(CELLS, suffixes):
        c = copy.deepcopy(originals["gpqa_main-q4-" + suffix])
        c.update(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
                 cell_id="thinking2507-gpqa-" + key, cell_key=key, prompt_mode="thinking",
                 damping_scope="none" if key == "native" else "full")
        result[key] = c
    return result


def prepare_pool(original, tokenizer):
    from .phase11_data import render_generation_prompt
    require(original.get("target_gold_loaded") is False and original["task"] == "gpqa_main",
            "Thinking2507 must derive from the existing gold-free GPQA pool")
    result = copy.deepcopy(original)
    for row in result["rows"]:
        content = render_generation_prompt({"question": row["question"],
            "options": row["choices"], "category": row["subject"]})
        messages = [{"role": "user", "content": content}]
        options = dict(add_generation_prompt=True, enable_thinking=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True, **options))
        require(tokenizer.encode(prompt, add_special_tokens=False) == ids,
                "Thinking2507 native template text/token projections differ")
        require(len(ids) + MAX_NEW_TOKENS <= MAX_LENGTH, "Thinking2507 prompt plus 8192 exceeds native capacity")
        require(tokenizer.convert_tokens_to_ids("</think>") not in ids,
                "Thinking2507 thinking template unexpectedly contains a closed thinking span")
        row.update(prompt=prompt, input_ids=ids, prompt_token_length=len(ids))
    result.update(model=MODEL, model_revision=REVISION, tokenizer_revision=REVISION,
                  prompt_mode="thinking", enable_thinking=True, max_length=MAX_LENGTH,
                  max_new_tokens=MAX_NEW_TOKENS, sampling=dict(SAMPLING),
                  runtime_seed=SEED, seed_rule="20261018+canonical_index",
                  think_open_id=tokenizer.convert_tokens_to_ids("<think>"),
                  think_close_id=tokenizer.convert_tokens_to_ids("</think>"))
    validate_pool(result)
    return result


def validate_pool(pool):
    require(pool.get("model") == MODEL and pool.get("model_revision") == REVISION and
            pool.get("tokenizer_revision") == REVISION and pool.get("target_gold_loaded") is False,
            "Thinking2507 model/tokenizer/gold-free binding differs")
    require(pool.get("enable_thinking") is True and pool.get("sampling") == SAMPLING and
            pool.get("max_length") == MAX_LENGTH and pool.get("max_new_tokens") == MAX_NEW_TOKENS,
            "Thinking2507 prompt/sampling/capacity binding differs")
    rows = pool["rows"]
    require(len(rows) == (1 if pool["engineering_synthetic"] else COUNT), "Thinking2507 population differs")
    require(pool["test_identities"] == [r["identity"] for r in rows] and
            len(set(pool["test_identities"])) == len(rows), "Thinking2507 canonical identity order differs")
    for r in rows:
        require(r["task"] == "gpqa_main" and r["choice_labels"] == list("ABCD") and
                len(r["choices"]) == 4 and not r["fewshot_sample_ids"] and
                r["prompt_token_length"] == len(r["input_ids"]) and
                len(r["input_ids"]) + MAX_NEW_TOKENS <= MAX_LENGTH,
                "Thinking2507 complete input/option/zero-shot contract differs")
    return pool


def actual_seed(index):
    require(type(index) is int and index >= 0, "Thinking2507 canonical seed index is invalid")
    return SEED + index


def answer_span(ids, prompt_ids, tokenizer, eos_ids):
    """Score the entire decoded output; retain first-close boundaries as diagnostics."""
    opened = tokenizer.convert_tokens_to_ids("<think>")
    closed = tokenizer.convert_tokens_to_ids("</think>")
    require(opened != closed and tokenizer.encode("</think>", add_special_tokens=False) == [closed],
            "Thinking2507 requires an actual single closing special token")
    closes = [i for i, t in enumerate(ids) if t == closed]
    opens = [i for i, t in enumerate(ids) if t == opened]
    end = len(ids) - int(bool(ids) and ids[-1] in eos_ids)
    boundary = closes[0] if closes else None
    start = 1 if opens and opens[0] == 0 else 0
    final_start = end if boundary is None else boundary + 1
    final_ids = ids[final_start:end]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True) if boundary is not None else ""
    whole_text = tokenizer.decode(ids, skip_special_tokens=True)
    parsed = extract_s2(whole_text)
    return {"parser": PARSER, "scoring_region": SCORING_REGION, "think_open_id": opened, "think_close_id": closed,
        "first_think_close_index": boundary, "generated_think_open_count": len(opens),
        "generated_think_close_count": len(closes), "prompt_think_open_count": prompt_ids.count(opened),
        "unclosed_thinking": boundary is None, "final_empty": not final_text.strip(),
        "anomalous_thinking_boundaries": len(closes) > 1 or len(opens) > 1 or
            bool(boundary is not None and any(i > boundary for i in opens)),
        "thinking_token_range": [start, end if boundary is None else boundary],
        "final_token_range": [final_start, end],
        "thinking_token_count": (end if boundary is None else boundary) - start,
        "final_token_count": len(final_ids), "final_text": final_text, **parsed}


def validate_job(job, config=None):
    require(job.get("seed") == SEED and type(job.get("synthetic")) is bool and
            type(job.get("engineering")) is bool and (not job["synthetic"] or job["engineering"]),
            "Thinking2507 runtime seed or engineering scope differs")
    cells = selected_cells(config)
    require(job.get("cell_key") in cells, "Thinking2507 cell outside the frozen three groups")
    c = cells[job["cell_key"]]
    require(all(job.get(k) == c[k] for k in ("cell_id", "prompt_mode", "damping_scope")),
            "Thinking2507 cell binding differs")
    require(all(isinstance(job.get(k), str) and job[k] for k in ("pool", "run_root", "source_commit")),
            "Thinking2507 job lacks source/input/output binding")
    require(job.get("indices") == sorted(set(job.get("indices", []))) and job["indices"],
            "Thinking2507 requires explicit canonical indices")
    return c
