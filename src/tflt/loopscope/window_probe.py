"""Actual K=2 damped-Euler window activity/contraction probe."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tflt.config import LoopConfig
from tflt.loopscope.grid import (
    candidate_windows,
    format_window,
    parse_window,
    validate_window_grid,
)
from tflt.loopscope.metrics import MetricError, contraction_ratio, relative_activity, summarize
from tflt.loopscope.probe import (
    _write_failure,
    environment_snapshot,
    git_provenance,
    load_probe_records,
    model_metadata,
    probe_pool_metadata,
    resolve_answer_position,
    runtime_metadata,
    serializable_args,
    select_device,
    tokenizer_revision,
    torch_dtype,
)
from tflt.loopscope.schema import (
    PROBE_SCHEMA_VERSION,
    ensure_new_directory,
    probe_summary,
    validate_probe_report,
    write_new_json,
)
from tflt.loopscope.revisions import strict_revision_closure
from tflt.models import resolve_model
from tflt.wrapper import apply_loop_wrapper


def add_probe_window_args(parser: Any) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--window-grid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", action="append", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")


def cmd_probe_window(args: Any) -> int:
    output_dir = ensure_new_directory(Path(args.output_dir))
    try:
        write_new_json(output_dir / "command_args.json", serializable_args(args))
        write_new_json(output_dir / "env.json", environment_snapshot())
        report = run_window_probe(args)
        validate_probe_report(report)
        write_new_json(output_dir / "probe_report.json", report)
        write_new_json(output_dir / "probe_summary.json", probe_summary(report))
    except Exception as exc:
        _write_failure(output_dir, exc)
        raise
    print(str(output_dir / "probe_report.json"))
    print(str(output_dir / "probe_summary.json"))
    return 0 if all(item["valid"] for item in report["window_metrics"]) else 2


def run_window_probe(args: Any) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote-only dependency path.
        raise RuntimeError("probe-window requires torch and transformers on HPC2") from exc

    windows = []
    for text in args.window:
        window = parse_window(text)
        if window[1] - window[0] + 1 != 4:
            raise ValueError("phase-one probe windows must have inclusive width 4")
        if window not in windows:
            windows.append(window)
    records = load_probe_records(Path(args.input_jsonl), args.max_examples)
    pool_metadata, pool_warnings = probe_pool_metadata(records, args.input_manifest)
    grid_payload = json.loads(Path(args.window_grid).read_text(encoding="utf-8"))
    validate_window_grid(grid_payload)
    model_info = resolve_model(args.model)
    repo_id = model_info["repo_id"]
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(repo_id, **load_kwargs)
    device = select_device(torch, args.device)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype(torch, args.dtype),
        **load_kwargs,
    )
    revision_closure = (
        strict_revision_closure(model, tokenizer, args.revision)
        if args.revision
        else None
    )
    model.eval()
    model.to(device)
    layer_count = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if int(grid_payload["layer_count"]) != layer_count:
        raise ValueError("window-grid layer_count does not match the loaded model")
    allowed_windows = set(candidate_windows(grid_payload))
    unknown_windows = [window for window in windows if window not in allowed_windows]
    if unknown_windows:
        raise ValueError("requested windows are absent from the frozen grid: %s" % unknown_windows)

    window_metrics = []
    for window in windows:
        if window[1] >= layer_count:
            raise ValueError("window %s exceeds model layer count %d" % (window, layer_count))
        collector = WindowProbeCollector(window_width=4)
        config = LoopConfig(
            model_alias=args.model,
            window=window,
            k=2,
            iteration_mode="block",
            strategy="damped_euler",
            alpha=1.0,
            beta=0.0,
            cache_strategy="last",
            decode_mode="bypass",
            audit_collector=collector,
        )
        handle = apply_loop_wrapper(model, config)
        try:
            for record in records:
                encoded = tokenizer(record["text"], return_tensors="pt")
                inputs = {key: value.to(device) for key, value in encoded.items()}
                answer_position = resolve_answer_position(inputs["attention_mask"], record)
                valid_positions = (
                    inputs["attention_mask"][0]
                    .detach()
                    .cpu()
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                collector.begin_forward(record["id"], valid_positions, answer_position)
                with torch.inference_mode():
                    outputs = model(**inputs, use_cache=False, return_dict=True)
                collector.end_forward()
                del outputs, inputs, encoded
                if device == "cuda":
                    torch.cuda.empty_cache()
        finally:
            handle.restore()
        window_metrics.append(collector.window_summary(window))

    actual_model_revision = (
        revision_closure["model_commit"]
        if revision_closure is not None
        else getattr(model.config, "_commit_hash", None) or args.revision
    )
    actual_tokenizer_revision = (
        revision_closure["tokenizer_commit"]
        if revision_closure is not None
        else tokenizer_revision(tokenizer, actual_model_revision)
    )
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "git": git_provenance(),
        "model": model_metadata(model, model_info, layer_count, actual_model_revision),
        "tokenizer": {
            "revision": actual_tokenizer_revision,
            "choice_token_ids": {},
        },
        "revision_closure": revision_closure,
        "runtime": runtime_metadata(torch, device, args.dtype),
        "probe_pool": pool_metadata,
        "position_rule": "explicit answer_position or last non-padding token",
        "loop_config": {
            "k": 2,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "use_cache": False,
        },
        "window_grid": {
            "path": str(Path(args.window_grid)),
            "manifest_sha256": grid_payload["manifest_sha256"],
            "layer_count": grid_payload["layer_count"],
            "candidate_windows": [item["window"] for item in grid_payload["windows"]],
        },
        "boundary_metrics": [],
        "window_metrics": window_metrics,
        "warnings": list(pool_warnings),
    }


class WindowProbeCollector:
    """Consumes unchanged wrapper events with per-forward mask/position context."""

    def __init__(self, window_width: int = 4) -> None:
        self.window_width = int(window_width)
        self.samples: List[Dict[str, Any]] = []
        self._active: Optional[Dict[str, Any]] = None
        self._answer_r: List[float] = []
        self._answer_q: List[float] = []
        self._all_r: List[float] = []
        self._all_q: List[float] = []

    def begin_forward(
        self, sample_id: str, valid_positions: Sequence[int], answer_position: int
    ) -> None:
        if self._active is not None:
            raise RuntimeError("begin_forward called before the previous forward ended")
        self._active = {
            "sample_id": str(sample_id),
            "valid_positions": [int(value) for value in valid_positions],
            "answer_position": int(answer_position),
            "counts": Counter(),
            "timeline": [],
            "g_records": [],
            "looped_hidden_max_abs": None,
            "errors": [],
        }

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        active = self._require_active()
        active["counts"][event] += 1
        active["timeline"].append(event)
        if event == "body_call":
            active["counts"]["operator_body_calls"] += 1
        elif event == "wrapper_forward":
            if payload.get("bypass"):
                active["counts"]["bypass_true"] += 1
            else:
                active["counts"]["bypass_false"] += 1

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        active = self._require_active()
        active["counts"][name] += 1
        active["timeline"].append(name)
        if name not in ("g_minus_x", "looped_hidden_vs_input"):
            return
        try:
            import torch

            lhs = before.detach().float()
            rhs = after.detach().float()
            if lhs.shape != rhs.shape or lhs.ndim != 3 or int(lhs.shape[0]) != 1:
                raise ValueError("expected matching [1, seq, hidden] tensors")
            delta = rhs - lhs
            if not bool(torch.isfinite(delta).all().item()):
                raise ValueError("tensor diff contains NaN/Inf")
            if name == "looped_hidden_vs_input":
                max_abs = float(delta.abs().max().item())
                active["looped_hidden_max_abs"] = max_abs
                if max_abs <= 1e-12:
                    active["errors"].append("looped_hidden_vs_input norm is zero")
                return
            residual_norms = torch.linalg.vector_norm(delta, dim=-1)[0].cpu().tolist()
            state_norms = torch.linalg.vector_norm(lhs, dim=-1)[0].cpu().tolist()
            active["g_records"].append(
                {
                    "ordinal": len(active["g_records"]) + 1,
                    "residual_norms": [float(value) for value in residual_norms],
                    "state_norms": [float(value) for value in state_norms],
                }
            )
        except Exception as exc:
            active["errors"].append("g_minus_x tensor error: %s" % exc)

    def end_forward(self) -> Dict[str, Any]:
        active = self._require_active()
        counts = active["counts"]
        if counts.get("wrapper_forward", 0) != 1:
            active["errors"].append("expected one wrapper_forward event")
        if counts.get("bypass_false", 0) != 1 or counts.get("bypass_true", 0) != 0:
            active["errors"].append("prefill wrapper unexpectedly bypassed")
        if counts.get("operator_body_calls", 0) != 2:
            active["errors"].append("expected exactly two operator body calls")
        if counts.get("g_minus_x", 0) != 2:
            active["errors"].append("expected exactly two g_minus_x tensor events")
        if len(active["g_records"]) != 2:
            active["errors"].append("expected exactly two ordered g_minus_x records")
        if counts.get("looped_hidden_vs_input", 0) != 1:
            active["errors"].append("expected one looped_hidden_vs_input tensor event")
        if counts.get("stash_pass", 0) != 1:
            active["errors"].append("expected one stash_pass event")
        if counts.get("identity_forward", 0) != self.window_width - 1:
            active["errors"].append("identity_forward count does not match window width")
        expected_timeline = [
            "wrapper_forward",
            "body_call",
            "g_minus_x",
            "body_call",
            "g_minus_x",
            "looped_hidden_vs_input",
            "stash_pass",
        ] + ["identity_forward"] * (self.window_width - 1)
        if active["timeline"] != expected_timeline:
            active["errors"].append(
                "unexpected event timeline: %s" % ",".join(active["timeline"])
            )

        result: Dict[str, Any] = {
            "sample_id": active["sample_id"],
            "answer_position": active["answer_position"],
            "event_counts": dict(counts),
            "event_timeline": list(active["timeline"]),
            "looped_hidden_max_abs": active["looped_hidden_max_abs"],
            "valid": False,
            "errors": list(active["errors"]),
        }
        if not result["errors"]:
            first, second = active["g_records"]
            metrics = build_window_sample_metrics(
                first["residual_norms"],
                second["residual_norms"],
                first["state_norms"],
                active["valid_positions"],
                active["answer_position"],
            )
            result.update({key: value for key, value in metrics.items() if not key.startswith("_")})
            if metrics["valid"]:
                self._answer_r.append(metrics["answer_position_metrics"]["r"])
                self._answer_q.append(metrics["answer_position_metrics"]["q"])
                self._all_r.extend(metrics["_all_r"])
                self._all_q.extend(metrics["_all_q"])
        self.samples.append(result)
        self._active = None
        return result

    def window_summary(self, window: Sequence[int]) -> Dict[str, Any]:
        errors = [
            {"sample_id": sample["sample_id"], "errors": sample["errors"]}
            for sample in self.samples
            if not sample.get("valid")
        ]
        valid = not errors and bool(self.samples)
        return {
            "window": format_window((int(window[0]), int(window[1]))),
            "start": int(window[0]),
            "end": int(window[1]),
            "valid": valid,
            "sample_count": len(self.samples),
            "valid_sample_count": sum(1 for sample in self.samples if sample.get("valid")),
            "answer_position": {
                "r": summarize(self._answer_r) if valid else None,
                "q": summarize(self._answer_q) if valid else None,
            },
            "all_non_padding_tokens": {
                "r": summarize(self._all_r) if valid else None,
                "q": summarize(self._all_q) if valid else None,
            },
            "examples": self.samples,
            "errors": errors,
        }

    def _require_active(self) -> Dict[str, Any]:
        if self._active is None:
            raise RuntimeError("wrapper event received outside begin_forward/end_forward")
        return self._active


def build_window_sample_metrics(
    first_residual_norms: Sequence[float],
    second_residual_norms: Sequence[float],
    first_state_norms: Sequence[float],
    valid_positions: Sequence[int],
    answer_position: int,
) -> Dict[str, Any]:
    lengths = {len(first_residual_norms), len(second_residual_norms), len(first_state_norms)}
    errors = []
    if len(lengths) != 1:
        errors.append("residual/state norm arrays have inconsistent lengths")
    if not valid_positions:
        errors.append("attention mask has no non-padding token")
    if answer_position not in valid_positions:
        errors.append("answer position is not a valid non-padding token")
    r_by_position: Dict[int, float] = {}
    q_by_position: Dict[int, float] = {}
    if not errors:
        for position in valid_positions:
            if position < 0 or position >= len(first_residual_norms):
                errors.append("valid token position %d is out of tensor range" % position)
                continue
            try:
                r_by_position[position] = relative_activity(
                    first_residual_norms[position], first_state_norms[position]
                )
                q_by_position[position] = contraction_ratio(
                    second_residual_norms[position], first_residual_norms[position]
                )
            except MetricError as exc:
                errors.append("token %d: %s" % (position, exc))
    all_r = [r_by_position[position] for position in valid_positions if position in r_by_position]
    all_q = [q_by_position[position] for position in valid_positions if position in q_by_position]
    valid = not errors
    return {
        "valid": valid,
        "errors": errors,
        "answer_position_metrics": (
            {"r": r_by_position[answer_position], "q": q_by_position[answer_position]}
            if valid
            else None
        ),
        "all_non_padding_tokens": {
            "r": summarize(all_r) if valid else None,
            "q": summarize(all_q) if valid else None,
        },
        "_all_r": all_r,
        "_all_q": all_q,
    }
