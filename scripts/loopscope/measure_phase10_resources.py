#!/usr/bin/env python3
"""Gate B synthetic resource measurements inside an already allocated debug GPU.

No scheduler, gold, accuracy, saved scores, or hidden tensors. Each packed child
loads its own model and runs the native HFLM token scorer through the existing
Phase 10/MMLU adapter. Synthetic token expansion preserves the validation
prompt's tokens and answer suffix; it is resource evidence only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from tflt.loopscope.phase10_accuracy import load_pool, require, row_identity, write_json_once
from tflt.loopscope.phase10_panel import load_config, validate_score_manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("mmlu", "arc_challenge"), required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--cell-id", required=True)
    result.add_argument("--pool-row", type=int, default=0)
    result.add_argument("--context-tokens", type=int, required=True)
    result.add_argument("--continuation-tokens", type=int, required=True)
    result.add_argument("--max-length", type=int)
    result.add_argument("--packing", type=int, choices=(1, 2), default=1)
    result.add_argument("--packing1-summary", type=Path,
                        help="Required packing2 admission: successful same-shape packing1 aggregate")
    result.add_argument("--iterations", type=int, default=1)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--max-seconds", type=int, default=1650)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--child-index", type=int, help=argparse.SUPPRESS)
    return result


def inputs(args):
    require(args.context_tokens > 0 and args.continuation_tokens > 0, "positive token shapes required")
    require(1 <= args.iterations <= 10, "resource iterations must be between one and ten")
    require(1 <= args.max_seconds <= 1650, "resource process deadline must fit a 29-minute job")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = validate_score_manifest(manifest, config=load_config(args.config),
                                    pool_path=args.pool, scope="PREFLIGHT_ONLY")
    require(manifest["dataset"] == args.dataset and args.cell_id in cells, "resource manifest/cell differs")
    cell = cells[args.cell_id]
    require(cell["arm"] == "Online" and cell["direction_policy"] == "current_t",
            "maximum SVD resource probe requires an existing current_t Online cell")
    rows = load_pool(args.pool, args.dataset, "PREFLIGHT_ONLY")
    require(0 <= args.pool_row < len(rows), "resource seed row is out of range")
    row = rows[args.pool_row]
    require(row["split"] == "validation", "synthetic resources must seed from validation only")
    metadata = {"schema": "loopscope.phase10.synthetic_resources.v1", "scope": "PREFLIGHT_ONLY",
        "synthetic_resource_only": True, "target_gold_loaded": False, "accuracy_computed": False,
        "source_commit": args.commit, "cell": cell, "pool": str(args.pool),
        "manifest": str(args.manifest), "seed_identity": row_identity(row, args.dataset),
        "context_tokens_requested": args.context_tokens,
        "continuation_tokens_requested": args.continuation_tokens,
        "candidate_count": 2, "iterations": args.iterations, "packing": args.packing,
        "max_length_requested": args.max_length, "argv": sys.argv,
        "resource_transfer_boundary": "measured GPU only; no A800 packing extrapolation"}
    return cell, row, metadata


def packing_admission(args, metadata):
    if args.packing == 1:
        require(args.packing1_summary is None, "packing1 does not consume prior packing evidence")
        return None
    require(args.packing1_summary is not None, "packing2 requires packing1 summary evidence")
    previous = json.loads(args.packing1_summary.read_text(encoding="utf-8"))
    require(previous.get("status") == "RESOURCE_COMPLETE" and previous.get("packing") == 1,
            "packing2 requires successful packing1")
    before = previous["metadata"]
    for key in ("cell", "context_tokens_requested", "continuation_tokens_requested", "candidate_count",
                "max_length_requested", "source_commit"):
        require(before[key] == metadata[key], "packing1 evidence differs: " + key)
    measurement = previous["gpu_sampling"]
    peak, total = measurement["peak_whole_gpu_used_mib"], measurement["total_memory_mib"]
    require(peak and total and 2 * peak <= 0.90 * total,
            "packing1 evidence has insufficient conservative headroom for packing2")
    return {"path": str(args.packing1_summary), "gpu_name": measurement["gpu_name"],
            "total_memory_mib": total, "packing1_peak_whole_gpu_used_mib": peak,
            "admission_rule": "2 * measured whole-GPU peak <= 90% total"}


def synthetic_requests(adapter, tokenizer, row, args):
    """Exact token shapes, native seed template intact, two synthetic candidates."""
    continuation = " A" if args.dataset == "mmlu" else " " + row["choices"][0]
    context, answer = adapter._encode_pair(row["prompt"], continuation)
    context, answer = list(context), list(answer)
    special = set(tokenizer.all_special_ids or ())
    if tokenizer.pad_token_id is not None:
        special.add(tokenizer.pad_token_id)
    filler = [token for token in tokenizer.encode(" resource", add_special_tokens=False) if token not in special]
    require(filler and context and answer, "synthetic resource tokens are empty")
    require(args.context_tokens >= len(context), "requested context is shorter than native validation seed")
    # Preserve the full seed and its final template tokens. No target test text
    # enters a forward. The inserted sequence contains only ordinary text tokens.
    suffix_length = min(8, len(context))
    extra = args.context_tokens - len(context)
    repeated = (filler * ((extra + len(filler) - 1) // len(filler)))[:extra]
    context = context[:-suffix_length] + repeated + context[-suffix_length:]
    require(context[-1] not in special, "native pre-answer token is special/padding")
    safe_answer = [token for token in answer if token not in special]
    require(safe_answer, "native candidate has no ordinary continuation token")
    require(args.context_tokens + args.continuation_tokens - 1 <= adapter.max_length,
            "requested resource shape would truncate the prompt; increase --max-length explicitly")
    candidates = []
    for index in range(2):
        # Equal lengths and prefixes avoid candidate-dependent truncation. Native
        # HFLM scores every continuation token; candidate values are discarded.
        sequence = safe_answer if index == 0 else filler
        candidate = (sequence * ((args.continuation_tokens + len(sequence) - 1) // len(sequence)))[:args.continuation_tokens]
        candidates.append((("synthetic-resource", index), list(context), candidate))
    return candidates, {"native_seed_context_tokens": len(context) - extra,
        "context_tokens": len(context), "continuation_tokens": args.continuation_tokens,
        "model_input_tokens": len(context) + args.continuation_tokens - 1,
        "inserted_context_tokens": extra, "preserved_suffix_tokens": suffix_length,
        "construction": "native validation token context plus repeated ordinary text tokens"}


def child(args):
    cell, row, metadata = inputs(args)
    output = args.run_root / ("child-" + str(args.child_index))
    output.mkdir(exist_ok=False)
    write_json_once(output / "command_args.json", metadata)
    started = time.monotonic()
    try:
        import torch
        from run_phase10_accuracy import load_runtime, make_loop_config
        from tflt.loopscope.phase10_accuracy import check_position_metadata
        from tflt.loopscope.phase10_runtime import validate_calls
        from tflt.wrapper import looped_model
        require(torch.cuda.device_count() == 1, "resource packing requires exactly one visible allocated GPU")
        torch.cuda.reset_peak_memory_stats()
        loading_started = time.monotonic()
        torch, model, tokenizer, adapter, runtime, env = load_runtime(cell, args.max_length)
        torch.cuda.synchronize()
        loading = {"elapsed_seconds": time.monotonic() - loading_started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        env.update(source_commit=args.commit, target_gold_loaded=False, synthetic_resource_only=True,
                   pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        write_json_once(output / "env.json", env)
        requests, shape = synthetic_requests(adapter, tokenizer, row, args)
        write_json_once(output / "ready.json", {"pid": os.getpid(), "shape": shape})
        while not (args.run_root / "score_release.json").exists():
            require(time.monotonic() - started < args.max_seconds, "resource scoring barrier timed out")
            time.sleep(0.1)
        torch.set_grad_enabled(False)
        torch.manual_seed(20261002)
        svd = {"fit_count": 0, "cuda_event_seconds": 0.0, "wall_seconds": 0.0,
               "max_rows": 0, "hidden_size": 0, "fit_count_by_t": {str(t): 0 for t in range(cell["k"])}}
        def measured_transform(delta, t):
            result = runtime(delta, t)
            fit = runtime._fit_metadata[t]
            require(fit["svd_timing"] == "cuda_event", "resource SVD is not CUDA-timed")
            svd["fit_count"] += 1
            svd["fit_count_by_t"][str(t)] += 1
            svd["cuda_event_seconds"] += fit["svd_seconds"]
            svd["wall_seconds"] += fit["svd_wall_seconds"]
            svd["max_rows"] = max(svd["max_rows"], fit["row_count"])
            svd["hidden_size"] = fit["hidden_size"]
            return result
        identity_attr = "phase9_identity" if args.dataset == "mmlu" else "phase10_identity"
        position_attr = "phase9_positions" if args.dataset == "mmlu" else "phase10_positions"
        durations, context_counts, context_count, position_shape = [], [], 0, None
        score_started = time.monotonic()
        with looped_model(model, make_loop_config(cell, measured_transform)):
            for iteration in range(args.iterations):
                setattr(adapter, identity_attr, "synthetic-resource-" + str(iteration))
                torch.cuda.synchronize()
                iteration_started = time.monotonic()
                values = adapter._loglikelihood_tokens(requests, disable_tqdm=True)
                torch.cuda.synchronize()
                durations.append(time.monotonic() - iteration_started)
                require(len(values) == 2 and all(math.isfinite(float(v[0])) for v in values),
                        "synthetic full-continuation scoring is nonfinite/incomplete")
                del values
                positions = getattr(adapter, position_attr)
                check_position_metadata(positions, tokenizer)
                position_shape = [{key: item[key] for key in
                    ("position", "input_length", "continuation_length", "left_truncated_tokens")}
                    | {"valid_prefix_count": len(item["effective_valid_positions"])} for item in positions]
                require(all(item["left_truncated_tokens"] == 0 and
                            item["input_length"] == shape["model_input_tokens"] and
                            item["continuation_length"] == args.continuation_tokens for item in positions),
                        "actual adapter resource shape differs from request")
                actual_contexts = validate_calls(runtime, cell["k"])
                require(1 <= actual_contexts <= 2, "native scoring did not execute one or two candidate contexts")
                context_counts.append(actual_contexts)
                context_count += actual_contexts
                runtime.calls.clear()
                runtime.context_summaries.clear()
                setattr(adapter, identity_attr, None)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - score_started
        require(svd["fit_count"] == context_count * cell["k"],
                "resource measurement did not execute all current_t SVD rounds")
        summary = {"schema": "loopscope.phase10.resource_child.v1", "status": "RESOURCE_COMPLETE",
            "synthetic_resource_only": True, "target_gold_loaded": False, "pid": os.getpid(),
            "shape": shape, "positions": position_shape, "model_loading": loading,
            "peak_including_loading_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_including_loading_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "scoring_seconds": elapsed, "iteration_seconds": durations,
            "scoring_started_monotonic": score_started,
            "scoring_completed_monotonic": score_started + elapsed,
            "candidate_scores": args.iterations * 2,
            "actual_model_contexts": context_count, "model_contexts_by_iteration": context_counts,
            "actual_model_contexts_per_second": context_count / elapsed,
            "candidate_scores_per_second": args.iterations * 2 / elapsed,
            "scored_continuation_tokens_per_second": args.iterations * 2 * args.continuation_tokens / elapsed,
            "native_logits_cache_enabled": bool(adapter.logits_cache),
            "svd": svd, "gpu_name": env["gpu"], "total_memory_bytes": env["total_memory"]}
        write_json_once(output / "summary.json", summary)
        return 0
    except Exception as error:
        write_json_once(output / "failure.json", {"status": "RESOURCE_FAILED", "pid": os.getpid(),
            "error_class": type(error).__name__, "error": str(error),
            "oom": "out of memory" in str(error).lower(), "elapsed_seconds": time.monotonic() - started,
            "synthetic_resource_only": True, "target_gold_loaded": False})
        raise


def sample_gpu(pids, state):
    """Whole-device occupancy and this attempt's independent process footprint."""
    gpu_result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10, check=True)
    process_result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10, check=True)
    devices = {}
    for line in gpu_result.stdout.splitlines():
        uuid, name, used, total = [part.strip() for part in line.split(",")]
        devices[uuid] = (name, int(used), int(total))
    own = []
    for line in process_result.stdout.splitlines():
        uuid, pid, used = [part.strip() for part in line.split(",")]
        if int(pid) in pids:
            own.append((uuid, int(pid), int(used)))
    if not own:
        return
    uuids = {item[0] for item in own}
    require(len(uuids) == 1, "packed resource children are not on the same GPU")
    uuid = next(iter(uuids))
    require(state.get("gpu_uuid", uuid) == uuid, "resource GPU changed during attempt")
    name, used, total = devices[uuid]
    state.update(gpu_uuid=uuid, gpu_name=name, total_memory_mib=total)
    state["samples_with_children"] += 1
    state["peak_whole_gpu_used_mib"] = max(state["peak_whole_gpu_used_mib"], used)
    state["peak_child_process_sum_mib"] = max(state["peak_child_process_sum_mib"], sum(item[2] for item in own))
    state["max_simultaneous_visible_child_processes"] = max(state["max_simultaneous_visible_child_processes"], len(own))


def parent(args):
    cell, row, metadata = inputs(args)
    admission = packing_admission(args, metadata)
    if args.dry_run:
        print(json.dumps({"status": "RESOURCE_DRY_RUN", "metadata": metadata, "packing_admission": admission}))
        return 0
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "Gate B resources require an already allocated debug Slurm job")
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(args.run_root / "command_args.json", metadata)
    children, logs = [], []
    sampling = {"sample_interval_seconds": 0.5, "samples_with_children": 0,
        "peak_whole_gpu_used_mib": 0, "peak_child_process_sum_mib": 0,
        "max_simultaneous_visible_child_processes": 0,
        "peak_is_sampled_lower_bound": True, "includes_model_loading": True}
    started, released, orchestration_error = time.monotonic(), None, None
    try:
        for index in range(args.packing):
            stdout = (args.run_root / ("child-" + str(index) + ".stdout.log")).open("x")
            stderr = (args.run_root / ("child-" + str(index) + ".stderr.log")).open("x")
            logs.extend((stdout, stderr))
            children.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                *sys.argv[1:], "--child-index", str(index)], stdout=stdout, stderr=stderr))
        while any(process.poll() is None for process in children):
            require(time.monotonic() - started < args.max_seconds, "resource parent deadline exceeded")
            sample_gpu({process.pid for process in children}, sampling)
            if admission and sampling.get("gpu_name"):
                require(sampling["gpu_name"] == admission["gpu_name"] and
                        sampling["total_memory_mib"] == admission["total_memory_mib"],
                        "packing2 GPU hardware differs from packing1 admission")
            if released is None and all((args.run_root / ("child-" + str(i)) / "ready.json").exists()
                                        or process.poll() is not None for i, process in enumerate(children)):
                require(all(process.poll() is None for process in children), "child failed before concurrent scoring barrier")
                released = time.monotonic()
                write_json_once(args.run_root / "score_release.json", {"ready_child_count": len(children)})
            time.sleep(0.5)
    except Exception as error:
        orchestration_error = {"error_class": type(error).__name__, "error": str(error)}
        for process in children:
            if process.poll() is None:
                process.terminate()
    finally:
        for process in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for handle in logs:
            handle.close()
    summaries = []
    for index, process in enumerate(children):
        root = args.run_root / ("child-" + str(index))
        path = root / "summary.json"
        summaries.append({"index": index, "pid": process.pid, "exit_code": process.returncode,
            "summary": json.loads(path.read_text()) if path.exists() else None,
            "failure_path": str(root / "failure.json") if (root / "failure.json").exists() else None})
        root.mkdir(exist_ok=True)
        write_json_once(root / "process_terminal.json", {"pid": process.pid,
            "exit_code": process.returncode, "summary_present": path.exists(),
            "parent_orchestration_error": orchestration_error})
    success = (not orchestration_error and len(children) == args.packing and
               all(item["exit_code"] == 0 and item["summary"] for item in summaries) and
               sampling["max_simultaneous_visible_child_processes"] == args.packing)
    completed = [item["summary"] for item in summaries if item["summary"]]
    score_elapsed = (max(item["scoring_completed_monotonic"] for item in completed) -
                     min(item["scoring_started_monotonic"] for item in completed)) if completed else None
    aggregate = {"schema": "loopscope.phase10.resource_aggregate.v1",
        "status": "RESOURCE_COMPLETE" if success else "RESOURCE_FAILED", "metadata": metadata,
        "synthetic_resource_only": True, "target_gold_loaded": False, "packing": args.packing,
        "packing_admission": admission, "children": summaries, "gpu_sampling": sampling,
        "attempt_seconds": time.monotonic() - started, "concurrent_scoring_seconds": score_elapsed,
        "aggregate_actual_model_contexts_per_second":
            sum(item["actual_model_contexts"] for item in completed) / score_elapsed if score_elapsed else None,
        "aggregate_candidate_scores_per_second":
            sum(item["candidate_scores"] for item in completed) / score_elapsed if score_elapsed else None,
        "aggregate_scored_continuation_tokens_per_second":
            sum(item["candidate_scores"] for item in completed) * args.continuation_tokens / score_elapsed
            if score_elapsed else None, "orchestration_error": orchestration_error}
    write_json_once(args.run_root / "summary.json", aggregate)
    print(json.dumps({"status": aggregate["status"], "run_root": str(args.run_root), "packing": args.packing}), flush=True)
    return 0 if success else 1


def main():
    args = parser().parse_args()
    if args.child_index is not None:
        require(0 <= args.child_index < args.packing, "invalid child index")
        return child(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
