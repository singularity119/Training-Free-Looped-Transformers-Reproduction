#!/usr/bin/env python3
"""Synthetic Phase 11 Gate B CUDA checks, exclusively PREFLIGHT_ONLY.

Run inside a <30 minute Slurm debug GPU allocation. No dataset or target gold
is loaded. The three-token generation limit is an engineering control, not
the formal Phase 11 2048-token recipe. Hidden/logit/direction tensors are never
written to disk. Only scalar checks and synthetic generated ids are saved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ATOL = RTOL = 1e-4
GENERATION_TEXT = ("Synthetic engineering input only. Describe a triangle briefly.\n"
                   "Options:\n(A) triangle\n(B) circle\n"
                   "Reason step by step, then end your response with exactly one line in the form "
                   "Final answer: (X), where X is the letter of the correct option.")
ARC_PROMPT = "Synthetic engineering input only.\nQuestion: Name a coloured shape.\nAnswer:"
ARC_CHOICES = ("a green triangle", "two red squares")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True, help="fresh evidence directory")
    result.add_argument("--commit", required=True, help="audited source revision supplied by submitter")
    result.add_argument("--config", type=Path, help="optional frozen Phase 11 config for validation")
    result.add_argument("--dry-run", action="store_true")
    return result


def check_plan():
    return {"schema": "loopscope.phase11.gpu_check_plan.v1", "scope": "PREFLIGHT_ONLY",
            "model": MODEL, "model_revision": REVISION, "tokenizer_revision": REVISION,
            "dtype": "bfloat16", "window": [15, 18], "cache_strategy": "first", "decode_mode": "full",
            "generation": {"do_sample": False, "num_beams": 1, "use_cache": True,
                           "min_new_tokens": 3, "max_new_tokens": 3,
                           "eos_token_ids_source": "cached native model generation_config"},
            "atol": ATOL, "rtol": RTOL, "target_gold_loaded": False,
            "cases": ["Native-original", "Native-monitored", "Loop-K2", "K2-current-lambda0",
                      "K2-fixed_t0-lambda0.5", "K2-lag1-lambda0.5", "K3-fixed_t0-lambda0.5",
                      "K3-lag1-lambda0.5", "K3-current_t-lambda0.5"],
            "arc_cases": ["Native-original-HFLM", "Native-Phase11-shim", "Loop-K2",
                          "K2-current-lambda0", "K2-fixed_t0-lambda0.5", "K2-lag1-lambda0.5"],
            "resource_scope": "small synthetic functional path; not formal memory or packing sizing"}


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class WrapperAudit:
    """Group the unchanged block wrapper's scalar events by actual forward."""
    def __init__(self, k):
        self.k = k
        self.groups = []

    def record(self, event, payload):
        if event == "wrapper_forward":
            self.groups.append({"wrapper_forward": dict(payload), "body_call_count": 0,
                                "stash_pass_count": 0, "stash_cache_strategy": None})
        elif event in ("body_call", "stash_pass", "bypass"):
            require(bool(self.groups), "wrapper event lacks forward group")
            group = self.groups[-1]
            if event == "bypass":
                raise ValueError("full-decode engineering check unexpectedly bypassed the loop")
            group[event + "_count"] += 1
            if event == "stash_pass":
                group["stash_cache_strategy"] = payload.get("cache_strategy")

    def validate(self, forward_count, generation_report=None):
        require(len(self.groups) == forward_count, "wrapper/model forward counts differ")
        for i, group in enumerate(self.groups):
            metadata = group["wrapper_forward"]
            require(metadata["wrapper_type"] == "block" and not metadata["bypass"], "unexpected wrapper mode")
            require(group["body_call_count"] == self.k and group["stash_pass_count"] == 1,
                    "actual forward does not have exactly K body calls and one stash pass")
            require(group["stash_cache_strategy"] == "first", "stash cache policy differs")
            if generation_report is not None:
                forward = generation_report["forwards"][i]
                require(metadata["hidden_seq_length"] == forward["sequence_length"], "wrapper input sequence differs")
                require(bool(metadata["is_incremental_decode_step"]) == (forward["kind"] == "decode"),
                        "wrapper does not recognize actual incremental decode")
        return list(self.groups)


def tensor_comparison(torch, left, right):
    require(tuple(left.shape) == tuple(right.shape), "comparison tensor shapes differ")
    require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), "comparison tensor is nonfinite")
    return {"max_abs_error": float((left.float() - right.float()).abs().max()),
            "allclose": bool(torch.allclose(left.float(), right.float(), atol=ATOL, rtol=RTOL)),
            "atol": ATOL, "rtol": RTOL}


def _cache_layers(cache, layer_count):
    if cache is None:
        return None
    if hasattr(cache, "get_seq_length"):
        return [int(cache.get_seq_length(i)) for i in range(layer_count)]
    return [int(cache[i][0].shape[-2]) for i in range(layer_count)]


@contextmanager
def capture_logits(torch, model):
    captured = {"logits": [], "forwards": []}
    def hook(module, inputs, output):
        logits = output.logits
        require(bool(torch.isfinite(logits).all()), "model emitted nonfinite logits")
        captured["logits"].append(logits.detach().float().cpu())
        cache = getattr(output, "past_key_values", None)
        captured["forwards"].append({"logits_shape": list(logits.shape),
                                     "cache_lengths_by_layer": _cache_layers(cache, model.config.num_hidden_layers)})
    handle = model.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def checking_runtime(torch, policy, k, strength):
    from tflt.loopscope.phase11_runtime import Phase11Runtime
    class CheckingRuntime(Phase11Runtime):
        def __init__(self):
            super().__init__(policy, k, strength)
            self.fit_events, self.freeze_checks = [], []
            self.bank_reference = {}

        @contextmanager
        def question(self, *args, **kwargs):
            try:
                with super().question(*args, **kwargs):
                    yield self
            finally:
                self.bank_reference.clear()

        def _fit_native_direction(self, delta, t):
            require(self._forward["kind"] == "prefill", "decode attempted a direction fit")
            require(delta.dtype == torch.bfloat16 and delta.device.type == "cuda",
                    "direction fit did not receive original BF16 CUDA residual")
            super()._fit_native_direction(delta, t)
            metadata = self.fit_metadata[t]
            require(metadata["algorithm"] == "cuda_gesvd_fp32_reduced_svd", "SVD path differs from frozen CUDA gesvd")
            self.fit_events.append({"kind": self._forward["kind"], "t": t, **metadata})

        def __call__(self, delta, t):
            result = super().__call__(delta, t)
            if self._forward["kind"] == "prefill" and t == self.k - 1:
                self.bank_reference = {index: direction.detach().clone() for index, direction in self.directions.items()}
            elif self._forward["kind"] == "decode":
                frozen = (set(self.directions) == set(self.bank_reference) and all(
                    torch.equal(direction, self.bank_reference[index]) for index, direction in self.directions.items()))
                require(frozen, "decode changed the prefill direction bank")
                self.freeze_checks.append({"forward_index": self._forward["index"], "t": t, "bank_unchanged": frozen})
            return result
    return CheckingRuntime()


def loop_config(k, runtime, collector):
    from tflt.config import LoopConfig
    return LoopConfig(model_alias=MODEL, window=(15, 18), k=k, iteration_mode="block",
                      strategy="damped_euler", alpha=1.0, beta=0.0, cache_strategy="first",
                      decode_mode="full", residual_transform=runtime, audit_collector=collector)


def compare_generation(torch, name, left, right):
    require(len(left["logits"]) == len(right["logits"]), "generation forward counts differ")
    comparisons = [tensor_comparison(torch, a, b) for a, b in zip(left["logits"], right["logits"])]
    same_ids = left["generated_ids"] == right["generated_ids"]
    return {"check": name, "generated_ids_equal": same_ids, "logit_comparisons": comparisons,
            "allclose": same_ids and all(c["allclose"] for c in comparisons)}


def runtime_evidence(runtime):
    from tflt.loopscope.phase11_runtime import validate_calls
    count = validate_calls(runtime)
    require(runtime._active is None and runtime._forward is None and not runtime.directions and not runtime.bank_reference,
            "question end retained an active context or direction tensor")
    return {"validated_forward_count": count, "directions_released": True,
            "runtime": runtime.summary(), "fit_events": runtime.fit_events,
            "decode_fit_count": sum(event["kind"] == "decode" for event in runtime.fit_events),
            "direction_freeze_checks": runtime.freeze_checks}


def run(args):
    plan = check_plan()
    if args.config is not None:
        from tflt.loopscope.phase11_panel import load_config
        load_config(args.config)
        plan["validated_config"] = str(args.config)
    if args.dry_run:
        print(json.dumps({"status": "PHASE11_GPU_CHECK_DRY_RUN", "source_commit": args.commit,
                          "run_root": str(args.run_root), **plan}, sort_keys=True))
        return 0
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "Phase 11 GPU check requires a real Slurm debug allocation")
    from run_phase11_accuracy import source_provenance
    source = source_provenance(args.commit)
    args.run_root.mkdir(parents=True, exist_ok=False)
    command = {**plan, **source, "source_commit": args.commit, "argv": sys.argv,
               "job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"]}
    write_once(args.run_root / "command_args.json", command)
    started = time.monotonic()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lm_eval.models.huggingface import HFLM
        from lm_eval.api.instance import Instance
        from tflt.wrapper import looped_model
        from tflt.loopscope.phase11_adapter import build_hflm_class, generation_forward_context
        require(torch.cuda.is_available(), "CUDA unavailable in debug allocation")
        require(importlib.metadata.version("lm_eval") == "0.4.11", "Phase 11 requires lm_eval 0.4.11")
        torch.set_grad_enabled(False)
        torch.manual_seed(20261008)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                    trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        require(getattr(model.config, "_commit_hash", REVISION) in (None, REVISION), "cached model revision differs")
        require(model.dtype == torch.bfloat16, "loaded model dtype differs")
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        require(eos_ids == [151645, 151643], "cached native generation EOS/EOT ids differ from frozen revision")
        environment = {"python": sys.executable, "platform": platform.platform(),
                       "model_revision": REVISION, "tokenizer_revision": REVISION,
                       "dtype": str(model.dtype), "native_generation_eos_ids": eos_ids,
                       "gpu": torch.cuda.get_device_name(),
                       "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
                       "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "lm_eval")}}
        write_once(args.run_root / "env.json", environment)
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": GENERATION_TEXT}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        tokens, mask = inputs["input_ids"][0].tolist(), inputs["attention_mask"][0].tolist()
        require(len(tokens) + 3 <= model.config.max_position_embeddings, "synthetic prompt exceeds native context")
        generation_records, retained = {}, {}
        comparisons = []

        def generate_case(name, k=None, policy=None, strength=None, monitor=True):
            runtime = checking_runtime(torch, policy, k, strength) if policy is not None else None
            collector = WrapperAudit(k) if k is not None else None
            manager = looped_model(model, loop_config(k, runtime, collector)) if k is not None else nullcontext()
            lifecycle = None
            with manager, capture_logits(torch, model) as captured:
                context = generation_forward_context(model, runtime, name, tokens, attention_mask=mask,
                           special_ids=tokenizer.all_special_ids, pad_token_id=tokenizer.pad_token_id) if monitor else nullcontext()
                with context as lifecycle:
                    output = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                                            min_new_tokens=3, max_new_tokens=3)
            generated = output[0, len(tokens):].tolist()
            require(len(generated) == 3 and len(captured["forwards"]) == 3,
                    "synthetic generation did not provide prefill plus two decode steps")
            for i, forward in enumerate(captured["forwards"]):
                require(forward["cache_lengths_by_layer"] == [len(tokens) + i] * model.config.num_hidden_layers,
                        "per-layer output caches do not close one append per actual forward")
            if monitor:
                require(lifecycle["prefill_count"] == 1 and lifecycle["decode_count"] == 2,
                        "generation lifecycle counts differ")
            record = {"generated_synthetic_ids": generated, "forward_outputs": captured["forwards"],
                      "generation_lifecycle": lifecycle}
            if collector is not None:
                record["wrapper_audit"] = collector.validate(3, lifecycle)
            if runtime is not None:
                record["online_evidence"] = runtime_evidence(runtime)
                require(len(runtime.freeze_checks) == 2 * k, "decode freeze evidence lacks K checks per step")
            generation_records[name] = record
            return {"logits": captured["logits"], "generated_ids": generated}

        retained["native"] = generate_case("Native-original", monitor=False)
        monitored = generate_case("Native-monitored")
        comparisons.append(compare_generation(torch, "Native-standard-vs-monitored", retained["native"], monitored))
        monitored["logits"].clear()
        retained["loop"] = generate_case("Loop-K2", k=2)
        zero = generate_case("K2-current-lambda0", 2, "current_t", 0.0)
        comparisons.append(compare_generation(torch, "LoopK2-vs-current-lambda0", retained["loop"], zero))
        zero["logits"].clear()
        fixed = generate_case("K2-fixed_t0-lambda0.5", 2, "fixed_t0", 0.5)
        lag = generate_case("K2-lag1-lambda0.5", 2, "lag1", 0.5)
        comparisons.append(compare_generation(torch, "K2-fixed_t0-vs-lag1", fixed, lag))
        fixed["logits"].clear()
        lag["logits"].clear()
        for policy in ("fixed_t0", "lag1", "current_t"):
            result = generate_case("K3-" + policy + "-lambda0.5", 3, policy, 0.5)
            result["logits"].clear()
        for result in retained.values():
            result["logits"].clear()

        # Native lm-eval owns the raw full-text scores, including all multi-token
        # continuation positions; the shim adds only common-prompt metadata.
        hflm_kwargs = {"pretrained": model, "tokenizer": tokenizer, "batch_size": "1", "trust_remote_code": True}
        original_hflm = HFLM(**hflm_kwargs)
        adapter = build_hflm_class(HFLM)(**hflm_kwargs)
        require(all(len(tokenizer.encode(" " + choice, add_special_tokens=False)) > 1 for choice in ARC_CHOICES),
                "synthetic ARC choices must exercise multi-token scoring")
        arc_records = {}
        def arc_case(name, k=None, policy=None, strength=None, base=False):
            runtime = checking_runtime(torch, policy, k, strength) if policy is not None else None
            collector = WrapperAudit(k) if k is not None else None
            adapter.phase11_runtime = runtime
            manager = looped_model(model, loop_config(k, runtime, collector)) if k is not None else nullcontext()
            with manager, capture_logits(torch, model) as captured:
                if base:
                    requests = [Instance(request_type="loglikelihood", doc={}, arguments=(ARC_PROMPT, " " + choice), idx=i)
                                for i, choice in enumerate(ARC_CHOICES)]
                    scores = original_hflm.loglikelihood(requests, disable_tqdm=True)
                else:
                    scores = adapter.score_choices(ARC_PROMPT, name, ARC_CHOICES)
            captured["logits"].clear()
            record = {"rawscores": [[float(score), bool(greedy)] for score, greedy in scores],
                      "forward_count": len(captured["forwards"]), "positions": [] if base else adapter.phase11_positions}
            if collector is not None:
                record["wrapper_audit"] = collector.validate(record["forward_count"])
            if runtime is not None:
                record["online_evidence"] = runtime_evidence(runtime)
                require(record["forward_count"] == len(runtime.context_summaries), "ARC callback path was not exercised")
                require(all(c["kind"] == "prefill" for c in runtime.context_summaries), "ARC candidate misclassified as decode")
            arc_records[name] = record
            return scores

        def compare_scores(name, a, b):
            import math
            require(len(a) == len(b) and len(a) > 0, "ARC score counts differ or empty")
            left, right = [float(v[0]) for v in a], [float(v[0]) for v in b]
            require(all(math.isfinite(v) for v in left + right), "ARC scores are nonfinite")
            differences = [abs(x-y) for x, y in zip(left, right)]
            same_flags = [bool(v[1]) for v in a] == [bool(v[1]) for v in b]
            comparison = {"check": name, "greedy_flags_equal": same_flags,
                          "max_abs_error": max(differences), "atol": ATOL, "rtol": RTOL,
                          "allclose": same_flags and all(d <= ATOL + RTOL*abs(y) for d, y in zip(differences, right))}
            comparisons.append(comparison)
        native_arc = arc_case("Native-original-HFLM", base=True)
        compare_scores("ARC-Native-HFLM-vs-Phase11-shim", native_arc, arc_case("Native-Phase11-shim"))
        loop_arc = arc_case("Loop-K2", k=2)
        compare_scores("ARC-LoopK2-vs-current-lambda0", loop_arc, arc_case("K2-current-lambda0", 2, "current_t", 0.0))
        fixed_arc = arc_case("K2-fixed_t0-lambda0.5", 2, "fixed_t0", 0.5)
        compare_scores("ARC-K2-fixed_t0-vs-lag1", fixed_arc, arc_case("K2-lag1-lambda0.5", 2, "lag1", 0.5))
        torch.cuda.synchronize()
        passed = all(comparison["allclose"] for comparison in comparisons)
        summary = {"schema": "loopscope.phase11.gpu_check.v1", "scope": "PREFLIGHT_ONLY",
                   "status": "GPU_CHECK_COMPLETE" if passed else "GPU_CHECK_FAILED", "target_gold_loaded": False,
                   "metadata": command, "generation_checks": generation_records, "arc_checks": arc_records,
                   "comparisons": comparisons, "elapsed_seconds": time.monotonic() - started,
                   "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                   "formal_resource_shape_validated": False}
        write_once(args.run_root / "summary.json", summary)
        print(json.dumps({"status": summary["status"], "allclose": passed, "target_gold_loaded": False}), flush=True)
        return 0 if passed else 1
    except Exception as exc:
        write_once(args.run_root / "failure.json", {"status": "GPU_CHECK_EXCEPTION", "exception": type(exc).__name__,
                   "message": str(exc), "elapsed_seconds": time.monotonic() - started, "scope": "PREFLIGHT_ONLY",
                   "target_gold_loaded": False})
        raise


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
