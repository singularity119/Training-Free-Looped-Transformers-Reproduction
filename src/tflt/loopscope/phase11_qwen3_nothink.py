"""Qwen3 nonthinking frozen greedy recipe and whole-output S2; no gold access."""
import copy

from .phase11_panel import build_panel, require
from .phase11_s2 import extract_s2

MODEL = "Qwen/Qwen3-4B"
REVISION = "UNVERIFIED_OSS_COPY"
SEED = 20261018
COUNT = 448
MAX_NEW_TOKENS = 32768
MAX_LENGTH = 40960
CELLS = ("native", "current_t_k2_l05", "fixed_t0_k3_l09")
SAMPLING = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}
PARSER = "D12-nonthinking-whole-final-GPQA-S2-v1"
SCORING_REGION = "whole_generated_output_as_final"
EMPTY_THINKING_SPAN = "<think>\n\n</think>\n\n"


def engineering_pool(original):
    require(original.get("engineering_synthetic") is True and original.get("scope") == "PREFLIGHT_ONLY",
            "Qwen3Nothink engineering fixture cannot replace a formal population")
    identity = "phase11-qwen3_nothink-synthetic:gpqa_main:0"
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
                 cell_id="qwen3_nothink-gpqa-" + key, cell_key=key, prompt_mode="cot",
                 damping_scope="none" if key == "native" else "full")
        result[key] = c
    return result


def prepare_pool(original, tokenizer):
    from .phase11_data import render_generation_prompt
    require(original.get("target_gold_loaded") is False and original["task"] == "gpqa_main",
            "Qwen3Nothink must derive from the existing gold-free GPQA pool")
    result = copy.deepcopy(original)
    opened = tokenizer.convert_tokens_to_ids("<think>")
    closed = tokenizer.convert_tokens_to_ids("</think>")
    require(opened != closed and tokenizer.encode("<think>", add_special_tokens=False) == [opened] and
            tokenizer.encode("</think>", add_special_tokens=False) == [closed],
            "Qwen3Nothink requires actual single thinking special tokens")
    empty_ids = list(tokenizer.encode(EMPTY_THINKING_SPAN, add_special_tokens=False))
    require(empty_ids and empty_ids[0] == opened and empty_ids.count(opened) == 1 and
            empty_ids.count(closed) == 1, "Qwen3Nothink empty span token projection differs")
    for row in result["rows"]:
        content = render_generation_prompt({"question": row["question"],
            "options": row["choices"], "category": row["subject"]})
        messages = [{"role": "user", "content": content}]
        options = dict(add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True, **options))
        require(tokenizer.encode(prompt, add_special_tokens=False) == ids,
                "Qwen3Nothink native template text/token projections differ")
        require(len(ids) + MAX_NEW_TOKENS <= MAX_LENGTH, "Qwen3Nothink prompt plus 32768 exceeds native capacity")
        require(prompt.endswith(EMPTY_THINKING_SPAN) and ids[-len(empty_ids):] == empty_ids,
                "Qwen3Nothink native disabled-thinking template lacks exact empty closed span")
        row.update(prompt=prompt, input_ids=ids, prompt_token_length=len(ids))
    result.update(model=MODEL, model_revision=REVISION, tokenizer_revision=REVISION,
                  prompt_mode="cot", enable_thinking=False, max_length=MAX_LENGTH,
                  max_new_tokens=MAX_NEW_TOKENS, sampling=dict(SAMPLING),
                  empty_thinking_span=EMPTY_THINKING_SPAN, empty_thinking_span_ids=empty_ids,
                  runtime_seed=SEED, seed_rule="20261018+canonical_index",
                  think_open_id=tokenizer.convert_tokens_to_ids("<think>"),
                  think_close_id=tokenizer.convert_tokens_to_ids("</think>"))
    validate_pool(result)
    return result


def validate_pool(pool):
    require(pool.get("model") == MODEL and pool.get("model_revision") == REVISION and
            pool.get("tokenizer_revision") == REVISION and pool.get("target_gold_loaded") is False,
            "Qwen3Nothink model/tokenizer/gold-free binding differs")
    require(pool.get("enable_thinking") is False and pool.get("prompt_mode") == "cot" and pool.get("sampling") == SAMPLING and
            pool.get("max_length") == MAX_LENGTH and pool.get("max_new_tokens") == MAX_NEW_TOKENS,
            "Qwen3Nothink prompt/sampling/capacity binding differs")
    empty_ids = pool.get("empty_thinking_span_ids", [])
    require(pool.get("empty_thinking_span") == EMPTY_THINKING_SPAN and empty_ids and
            empty_ids[0] == pool["think_open_id"] and
            empty_ids.count(pool["think_open_id"]) == 1 and
            empty_ids.count(pool["think_close_id"]) == 1,
            "Qwen3Nothink empty closed prefix binding differs")
    rows = pool["rows"]
    require(len(rows) == (1 if pool["engineering_synthetic"] else COUNT), "Qwen3Nothink population differs")
    require(pool["test_identities"] == [r["identity"] for r in rows] and
            len(set(pool["test_identities"])) == len(rows), "Qwen3Nothink canonical identity order differs")
    for r in rows:
        require(r["task"] == "gpqa_main" and r["choice_labels"] == list("ABCD") and
                len(r["choices"]) == 4 and not r["fewshot_sample_ids"] and
                r["prompt_token_length"] == len(r["input_ids"]) and
                len(r["input_ids"]) + MAX_NEW_TOKENS <= MAX_LENGTH and
                r["prompt"].endswith(EMPTY_THINKING_SPAN) and
                r["input_ids"][-len(empty_ids):] == empty_ids,
                "Qwen3Nothink complete input/option/zero-shot contract differs")
    return pool


def actual_seed(index):
    require(type(index) is int and index >= 0, "Qwen3Nothink canonical seed index is invalid")
    return SEED + index


def answer_span(ids, prompt_ids, tokenizer, eos_ids):
    """Score the entire decoded output; retain first-close boundaries as diagnostics."""
    opened = tokenizer.convert_tokens_to_ids("<think>")
    closed = tokenizer.convert_tokens_to_ids("</think>")
    require(opened != closed and tokenizer.encode("</think>", add_special_tokens=False) == [closed],
            "Qwen3Nothink requires an actual single closing special token")
    closes = [i for i, t in enumerate(ids) if t == closed]
    opens = [i for i, t in enumerate(ids) if t == opened]
    end = len(ids) - int(bool(ids) and ids[-1] in eos_ids)
    boundary = closes[0] if closes else None
    start = 1 if opens and opens[0] == 0 else 0
    has_thinking = bool(opens)
    final_start = 0  # D12: the entire generated output is the final scoring region.
    final_ids = ids[final_start:end]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    unmatched_opens = 0
    for token in ids:
        if token == opened:
            unmatched_opens += 1
        elif token == closed and unmatched_opens:
            unmatched_opens -= 1
    thinking_end = (end if has_thinking else start) if boundary is None else boundary
    whole_text = tokenizer.decode(ids, skip_special_tokens=True)
    parsed = extract_s2(whole_text)
    return {"parser": PARSER, "scoring_region": SCORING_REGION, "think_open_id": opened, "think_close_id": closed,
        "first_think_close_index": boundary, "generated_think_open_count": len(opens),
        "generated_think_close_count": len(closes), "prompt_think_open_count": prompt_ids.count(opened),
        "unclosed_thinking": unmatched_opens > 0, "final_empty": not final_text.strip(),
        "anomalous_thinking_boundaries": len(closes) > 1 or len(opens) > 1 or
            bool(boundary is not None and any(i > boundary for i in opens)),
        "thinking_token_range": [start, thinking_end],
        "final_token_range": [final_start, end],
        "thinking_token_count": thinking_end - start,
        "final_token_count": len(final_ids), "final_text": final_text, **parsed}


def validate_job(job, config=None):
    require(job.get("seed") == SEED and type(job.get("synthetic")) is bool and
            type(job.get("engineering")) is bool and (not job["synthetic"] or job["engineering"]),
            "Qwen3Nothink runtime seed or engineering scope differs")
    cells = selected_cells(config)
    require(job.get("cell_key") in cells, "Qwen3Nothink cell outside the frozen three groups")
    c = cells[job["cell_key"]]
    require(all(job.get(k) == c[k] for k in ("cell_id", "prompt_mode", "damping_scope")),
            "Qwen3Nothink cell binding differs")
    require(all(isinstance(job.get(k), str) and job[k] for k in ("pool", "run_root", "source_commit")),
            "Qwen3Nothink job lacks source/input/output binding")
    require(job.get("indices") == sorted(set(job.get("indices", []))) and job["indices"],
            "Qwen3Nothink requires explicit canonical indices")
    return c
