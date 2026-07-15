"""Source-aware Gate D atlas assembly and one-shot verification."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase2_fixed_horizon import (
    CARD_ID,
    EXECUTOR_THREAD_ID,
    FixedHorizonError,
    WINDOW,
    _verify_manifest,
    utc_now,
    validate_fixed_horizon_card,
)
from tflt.loopscope.phase2_schema import (
    DIRECT_PROBE_SCORE_SOURCE,
    atomic_write_new_json,
    file_sha256,
    make_hashed_manifest,
    protocol_cell_id,
    validate_no_persisted_vectors,
)


LATENT_METRICS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("raw_residual_norm", ("raw_residual_norm",)),
    ("raw_relative_activity", ("raw_relative_activity",)),
    ("residual_ratio_to_previous", ("residual_ratio_to_previous",)),
    ("adjacent_residual_cosine", ("adjacent_residual_cosine",)),
    ("repeated_step_nca", ("repeated_step_nca", "value")),
    ("h_scaled_applied_update_norm", ("h_scaled_applied_update_norm",)),
    ("h_scaled_applied_relative_magnitude", ("h_scaled_applied_relative_magnitude",)),
    ("actual_applied_state_delta_norm", ("actual_applied_state_delta_norm",)),
    ("applied_state_rounding_gap_norm", ("applied_state_rounding_gap_norm",)),
    ("cumulative_h_scaled_path_length", ("cumulative_h_scaled_path_length",)),
    ("cumulative_applied_displacement_norm", ("cumulative_applied_displacement_norm",)),
    ("intermediate_lens_entry_entropy_nats", ("intermediate_lens_entry", "entropy_nats")),
    ("intermediate_lens_raw_exit_entropy_nats", ("intermediate_lens_raw_exit", "entropy_nats")),
    ("intermediate_lens_applied_state_entropy_nats", ("intermediate_lens_applied_state", "entropy_nats")),
    ("intermediate_lens_entry_top_margin_raw", ("intermediate_lens_entry", "top_margin_raw")),
    ("intermediate_lens_raw_exit_top_margin_raw", ("intermediate_lens_raw_exit", "top_margin_raw")),
    ("intermediate_lens_applied_state_top_margin_raw", ("intermediate_lens_applied_state", "top_margin_raw")),
    ("raw_entropy_drop_nats", ("raw_entropy_drop_nats",)),
    ("applied_entropy_drop_nats", ("applied_entropy_drop_nats",)),
    ("raw_call_entropy_flatness_negative_population_std", ("raw_call_entropy_flatness_negative_population_std",)),
    ("entry_kl_to_native_no_loop_reference_nats", ("entry_kl_to_native_no_loop_reference_nats",)),
    ("raw_kl_to_reference_drop_nats", ("raw_kl_to_reference_drop_nats",)),
    ("applied_kl_to_reference_drop_nats", ("applied_kl_to_reference_drop_nats",)),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FixedHorizonError("JSON source must contain an object: %s" % path)
    return value


def _load_hashed(path: Path) -> Mapping[str, Any]:
    value = _load_json(path)
    _verify_manifest(value)
    return value


def _source_ref(path: Path, *, cell_id: Optional[str] = None) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    payload = _load_json(resolved)
    result: Dict[str, Any] = {"path": str(resolved), "file_sha256": file_sha256(resolved)}
    if "manifest_sha256" in payload:
        _verify_manifest(payload)
        result["manifest_sha256"] = payload["manifest_sha256"]
    if cell_id is not None:
        result["cell_id"] = cell_id
    return result


def _resolve_gate_c_ref(ref: Mapping[str, Any], base: Path) -> Path:
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise FixedHorizonError("Gate C source ref lacks path")
    path = Path(ref["path"])
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def build_source_manifest(
    *,
    card_path: Path,
    profile_manifest_path: Path,
    gate_c_analysis_input_path: Path,
    gate_c_report_path: Path,
    phase1_selection_report_path: Path,
) -> Mapping[str, Any]:
    """Bind D2, Gate B/C, Phase 1, full K1--K4, and label evidence."""

    card = _load_hashed(card_path)
    validate_fixed_horizon_card(card)
    profile = _load_hashed(profile_manifest_path)
    if profile.get("mode") != "d2-profile" or profile.get("sample_count") != 512:
        raise FixedHorizonError("source closure requires the complete D2 512 profile")
    request_path = Path(gate_c_analysis_input_path).resolve()
    request = _load_hashed(request_path)
    sources = request.get("sources")
    if not isinstance(sources, Mapping):
        raise FixedHorizonError("Gate C analysis input lacks sources")
    full_refs = sources.get("full_final_output_cells")
    if not isinstance(full_refs, list):
        raise FixedHorizonError("Gate C analysis input lacks full cells")
    expected = {
        1: protocol_cell_id("baseline_no_loop", "none", 1, 1.0),
        2: protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0),
        3: protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
        4: protocol_cell_id("fixed_horizon", "12:15", 4, 1.0),
    }
    by_id = {ref.get("cell_id"): ref for ref in full_refs if isinstance(ref, Mapping)}
    full: Dict[str, Any] = {}
    for k, cell_id in expected.items():
        if cell_id not in by_id:
            raise FixedHorizonError("Gate C analysis input lacks full K%d cell" % k)
        path = _resolve_gate_c_ref(by_id[cell_id], request_path.parent)
        ref = _source_ref(path, cell_id=cell_id)
        if by_id[cell_id].get("sha256") != ref.get("manifest_sha256"):
            raise FixedHorizonError("Gate C full source ref hash differs for K%d" % k)
        full["k%d" % k] = ref
    routed_names = (
        "card",
        "calibration_identity_manifest",
        "calibration_aggregate",
        "calibration_labels",
        "unseal_authorization",
        "unseal_receipt",
    )
    routed: Dict[str, Any] = {}
    for name in routed_names:
        if name not in sources:
            raise FixedHorizonError("Gate C source request lacks %s" % name)
        path = _resolve_gate_c_ref(sources[name], request_path.parent)
        ref = _source_ref(path)
        if sources[name].get("sha256") != ref.get("manifest_sha256"):
            raise FixedHorizonError("Gate C routed source hash differs: %s" % name)
        routed[name] = ref
    gate_c_report = _source_ref(gate_c_report_path)
    phase1_report = _source_ref(phase1_selection_report_path)
    manifest = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-source-manifest.v1",
            "card_id": CARD_ID,
            "card_manifest_sha256": card["manifest_sha256"],
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "created_at_utc": utc_now(),
            "profile_manifest": _source_ref(profile_manifest_path),
            "gate_c_analysis_input": _source_ref(request_path),
            "gate_c_report": gate_c_report,
            "phase1_selection_report": phase1_report,
            "gate_b_calibration_aggregate": routed["calibration_aggregate"],
            "calibration_identity_manifest": routed["calibration_identity_manifest"],
            "calibration_labels": routed["calibration_labels"],
            "unseal_authorization": routed["unseal_authorization"],
            "unseal_receipt": routed["unseal_receipt"],
            "h1_card": routed["card"],
            "full_cells": full,
            "source_namespace_separation": {
                "calibration": "phase1_frozen_validation_512_direct_probe",
                "full": "phase1_gatec_full_14042_lm_eval",
                "cross_namespace_sample_join": False,
            },
            "new_full_run_count": 0,
        }
    )
    validate_source_manifest(manifest, card)
    return manifest


def validate_source_manifest(manifest: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    _verify_manifest(manifest)
    if manifest.get("schema_version") != "loopscope.h2-fixed-horizon-source-manifest.v1":
        raise FixedHorizonError("unsupported source manifest schema")
    if manifest.get("card_manifest_sha256") != card["manifest_sha256"] or manifest.get("new_full_run_count") != 0:
        raise FixedHorizonError("source manifest card/new-full binding differs")
    if manifest.get("source_namespace_separation") != card["source_namespaces"]:
        raise FixedHorizonError("source namespaces differ from card")
    refs: List[Mapping[str, Any]] = []
    for key in (
        "profile_manifest",
        "gate_c_analysis_input",
        "gate_c_report",
        "phase1_selection_report",
        "gate_b_calibration_aggregate",
        "calibration_identity_manifest",
        "calibration_labels",
        "unseal_authorization",
        "unseal_receipt",
        "h1_card",
    ):
        refs.append(manifest[key])
    refs.extend(manifest["full_cells"].values())
    for ref in refs:
        path = Path(ref["path"])
        if file_sha256(path) != ref["file_sha256"]:
            raise FixedHorizonError("source file hash drift: %s" % path)
        payload = _load_json(path)
        if "manifest_sha256" in ref:
            _verify_manifest(payload)
            if payload["manifest_sha256"] != ref["manifest_sha256"]:
                raise FixedHorizonError("source manifest hash drift: %s" % path)


class BootstrapEngine:
    """One deterministic seed stream for all new scalar percentile intervals."""

    def __init__(self, *, replicates: int, seed: int) -> None:
        self.replicates = int(replicates)
        self.seed = int(seed)
        try:
            import numpy as np
        except Exception:
            self.np = None
            self.rng = random.Random(seed)
        else:
            self.np = np
            self.rng = np.random.default_rng(seed)

    def summarize(self, values: Sequence[float]) -> Optional[Dict[str, Any]]:
        data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not data:
            return None
        point = math.fsum(data) / len(data)
        if self.np is not None:
            array = self.np.asarray(data, dtype=self.np.float64)
            samples = []
            remaining = self.replicates
            while remaining:
                count = min(200, remaining)
                indices = self.rng.integers(0, len(array), size=(count, len(array)))
                samples.extend(array[indices].mean(axis=1).tolist())
                remaining -= count
        else:
            samples = [math.fsum(self.rng.choice(data) for _ in data) / len(data) for _ in range(self.replicates)]
        samples.sort()
        lower = _percentile(samples, 0.025)
        upper = _percentile(samples, 0.975)
        return {
            "count": len(data),
            "mean": point,
            "median": statistics.median(data),
            "bootstrap_percentile_95": [lower, upper],
            "replicates": self.replicates,
            "seed_stream_origin": self.seed,
        }


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise FixedHorizonError("percentile requires values")
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _at_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _load_profile_cells(profile_manifest: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Dict[int, Mapping[str, Any]]]:
    root = Path(profile_manifest["artifact_root"])
    baseline = _load_hashed(root / profile_manifest["baseline_reference"]["path"])
    cells = {}
    for ref in profile_manifest["cells"]:
        cell = _load_hashed(root / ref["path"])
        if cell["manifest_sha256"] != ref["sha256"]:
            raise FixedHorizonError("profile cell ref hash differs")
        cells[int(ref["k"])] = cell
    if set(cells) != {1, 2, 3, 4}:
        raise FixedHorizonError("profile K matrix is incomplete")
    return baseline, cells


def summarize_latent_trajectory(
    card: Mapping[str, Any],
    profile_manifest: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> Mapping[str, Any]:
    baseline, cells = _load_profile_cells(profile_manifest)
    stats = card["statistics"]["new_scalar_bootstrap"]
    engine = BootstrapEngine(replicates=stats["replicates"], seed=stats["seed"])
    nca_stats = card["statistics"]["nca_bootstrap"]
    nca_engine = BootstrapEngine(replicates=nca_stats["replicates"], seed=nca_stats["seed"])
    kwise = []
    for k in (1, 2, 3, 4):
        cell = cells[k]
        if len(cell["samples"]) != 512:
            raise FixedHorizonError("D2 cell does not contain 512 samples")
        steps = []
        for t in range(k):
            rows = [sample["steps"][t] for sample in cell["samples"]]
            metrics = {}
            for name, path in LATENT_METRICS:
                values = [_at_path(row, path) for row in rows]
                summary_engine = nca_engine if name == "repeated_step_nca" else engine
                metrics[name] = summary_engine.summarize([value for value in values if value is not None])
            top1 = {}
            for stage in ("entry", "raw_exit", "applied_state"):
                key = "intermediate_lens_%s" % stage
                counts = [0, 0, 0, 0]
                for row in rows:
                    counts[int(row[key]["top1_index"])] += 1
                top1[stage] = {"counts_A_B_C_D": counts, "fractions_A_B_C_D": [value / 512.0 for value in counts]}
            steps.append(
                {
                    "body_call_t": t,
                    "tau_entry": t / float(k),
                    "tau_applied": (t + 1) / float(k),
                    "h": 1.0 / k,
                    "metrics": metrics,
                    "intermediate_lens_top1": top1,
                    "cohort_effective_rank": cell["cohort_effective_rank"][t],
                }
            )
        kwise.append({"k": k, "sample_count": 512, "body_call_count": 512 * k, "steps": steps})
    subgroups = calibration_development_subgroups(cells, labels)
    result = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-kwise-latent-trajectory.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "source_namespace": "phase1_frozen_validation_512_direct_probe",
            "label_role": "development_diagnostic_post_hoc_only",
            "baseline_reference_sample_count": baseline["sample_count"],
            "total_body_calls": sum(row["body_call_count"] for row in kwise),
            "expected_total_body_calls": 5120,
            "kwise": kwise,
            "development_subgroups": subgroups,
            "bootstrap": stats,
            "nca_bootstrap": nca_stats,
            "vectors_persisted": False,
        }
    )
    validate_no_persisted_vectors(result)
    return result


def _choice_with_gold(row: Mapping[str, Any], gold: int) -> Mapping[str, Any]:
    from tflt.loopscope.phase2_analysis import choice_output

    return choice_output(
        row["raw_choice_scores"],
        identity=row["sample_identity"],
        gold_index=int(gold),
        score_source=DIRECT_PROBE_SCORE_SOURCE,
    )


def _transition_name(left: bool, right: bool) -> str:
    if left and right:
        return "right_to_right"
    if not left and right:
        return "wrong_to_right"
    if left and not right:
        return "right_to_wrong"
    return "wrong_to_wrong"


def calibration_development_subgroups(cells: Mapping[int, Mapping[str, Any]], labels: Mapping[str, Any]) -> Mapping[str, Any]:
    label_rows = labels.get("samples")
    if not isinstance(label_rows, list) or len(label_rows) != 512:
        raise FixedHorizonError("authorized calibration label sidecar must contain 512 rows")
    label_by_key = {_identity_key(row["sample_identity"]): int(row["gold_index"]) for row in label_rows}
    result = {}
    for left_k, right_k in ((1, 2), (2, 3), (3, 4)):
        left_samples = cells[left_k]["samples"]
        right_samples = cells[right_k]["samples"]
        grouped: Dict[str, List[int]] = {name: [] for name in ("right_to_right", "wrong_to_right", "right_to_wrong", "wrong_to_wrong")}
        for index, (left, right) in enumerate(zip(left_samples, right_samples)):
            if left["sample_identity"] != right["sample_identity"]:
                raise FixedHorizonError("calibration K cells are not identity-aligned")
            gold = label_by_key.get(_identity_key(left["sample_identity"]))
            if gold is None:
                raise FixedHorizonError("calibration label sidecar lacks an identity")
            left_choice = _choice_with_gold(left["final_direct_probe_output"], gold)
            right_choice = _choice_with_gold(right["final_direct_probe_output"], gold)
            grouped[_transition_name(bool(left_choice["correctness"]), bool(right_choice["correctness"]))].append(index)
        group_reports = {}
        for group, indices in grouped.items():
            per_step = []
            for t in range(right_k):
                rows = [right_samples[index]["steps"][t] for index in indices]
                metrics = {}
                for name, path in LATENT_METRICS:
                    values = [_at_path(row, path) for row in rows]
                    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
                    metrics[name] = (
                        {"count": len(finite), "mean": math.fsum(finite) / len(finite), "median": statistics.median(finite)}
                        if finite else None
                    )
                per_step.append({"body_call_t": t, "tau_entry": t / float(right_k), "tau_applied": (t + 1) / float(right_k), "metrics": metrics})
            group_reports[group] = {"count": len(indices), "right_k_stepwise_point_summaries": per_step}
        result["k%d_to_k%d" % (left_k, right_k)] = {
            "score_namespace": "direct_probe_next_token_log_probability_over_frozen_choice_token_ids",
            "role": "development_diagnostic_post_hoc_only",
            "groups": group_reports,
        }
    return result


def _identity_key(identity: Mapping[str, Any]) -> str:
    return "%s\0%s\0%s" % (identity["task"], identity["doc_id"], identity["doc_hash"])


def _full_rows(cell: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    rows = cell.get("samples")
    if not isinstance(rows, list) or len(rows) != 14042:
        raise FixedHorizonError("full cell must contain 14,042 rows")
    result = []
    for row in rows:
        value = row.get("final_output") if isinstance(row, Mapping) and "final_output" in row else row
        if not isinstance(value, Mapping) or value.get("choice_score_source") != "lm_eval_acc_none_raw_per_choice_loglikelihood":
            raise FixedHorizonError("full row uses the wrong score namespace")
        result.append(value)
    return result


def summarize_full(
    card: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Dict[int, List[Mapping[str, Any]]]]:
    from tflt.loopscope.phase2_analysis import (
        jensen_shannon,
        paired_accuracy_contrast,
        summarize_choice_pair,
    )

    cells = {k: _load_hashed(Path(source_manifest["full_cells"]["k%d" % k]["path"])) for k in (1, 2, 3, 4)}
    for k, cell in cells.items():
        revision = cell.get("revision") or cell.get("model_revision") or cell.get("cell", {}).get("revision")
        if revision is not None and revision != card["science"]["revision"]:
            raise FixedHorizonError("full K%d revision differs from the H2 card" % k)
    rows = {k: _full_rows(cells[k]) for k in cells}
    baseline = rows[1]
    absolute = []
    for k in (1, 2, 3, 4):
        current = rows[k]
        correct = [bool(row["correctness"]) for row in current]
        js = [jensen_shannon(left["choice_probabilities"], right["choice_probabilities"]) for left, right in zip(baseline, current)]
        retained = [int(left["top1_index"]) == int(right["top1_index"]) for left, right in zip(baseline, current)]
        absolute.append(
            {
                "k": k,
                "sample_count": len(current),
                "accuracy_fraction": sum(correct) / len(correct),
                "accuracy_percent": 100.0 * sum(correct) / len(correct),
                "mean_entropy_nats": math.fsum(float(row["entropy_nats"]) for row in current) / len(current),
                "mean_top_margin_raw": math.fsum(float(row["top_margin_raw"]) for row in current) / len(current),
                "mean_correct_margin_raw": math.fsum(float(row["correct_margin_raw"]) for row in current) / len(current),
                "mean_js_to_k1_nats": math.fsum(js) / len(js),
                "top1_retained_vs_k1_fraction": sum(retained) / len(retained),
            }
        )
    absolute_payload = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-kwise-absolute-full-summary.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "source_namespace": "phase1_gatec_full_14042_lm_eval",
            "sample_count": 14042,
            "new_full_run_count": 0,
            "kwise": absolute,
        }
    )
    stats = card["statistics"]["full_paired_bootstrap"]
    contrasts = []
    for left_k, right_k in ((1, 2), (2, 3), (3, 4)):
        accuracy = paired_accuracy_contrast(
            rows[left_k], rows[right_k],
            bootstrap_replicates=stats["replicates"],
            bootstrap_seed=stats["seed"],
        )
        mechanism = summarize_choice_pair(rows[left_k], rows[right_k])
        contrasts.append(
            {
                "contrast": "k%d_to_k%d" % (left_k, right_k),
                "left_k": left_k,
                "right_k": right_k,
                "accuracy": accuracy,
                "mechanism": mechanism,
            }
        )
    contrasts_payload = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-adjacent-full-contrasts.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "source_namespace": "phase1_gatec_full_14042_lm_eval",
            "sample_count": 14042,
            "bootstrap": stats,
            "contrasts": contrasts,
        }
    )
    return absolute_payload, contrasts_payload, rows


def assemble_outputs(card: Mapping[str, Any], source_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    validate_source_manifest(source_manifest, card)
    profile = _load_hashed(Path(source_manifest["profile_manifest"]["path"]))
    labels = _load_hashed(Path(source_manifest["calibration_labels"]["path"]))
    latent = summarize_latent_trajectory(card, profile, labels)
    absolute, contrasts, _ = summarize_full(card, source_manifest)
    accuracy_by_k = {row["k"]: row["accuracy_percent"] for row in absolute["kwise"]}
    best_k = max(accuracy_by_k, key=lambda k: accuracy_by_k[k])
    atlas = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-trajectory-atlas.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "role": "post_hoc_exploratory_mechanism_profile_not_confirmatory_selector_evidence",
            "frozen_scope": {"window": WINDOW, "k": [1, 2, 3, 4], "alpha": 1.0, "calibration": 512, "full": 14042},
            "headline": {
                "best_observed_full_accuracy_k": best_k,
                "full_accuracy_percent_by_k": accuracy_by_k,
                "causal_claim": False,
                "selector_claim": False,
                "four_k_point_global_correlation_reported": False,
            },
            "latent_summary_sha256": latent["manifest_sha256"],
            "absolute_full_summary_sha256": absolute["manifest_sha256"],
            "adjacent_full_contrasts_sha256": contrasts["manifest_sha256"],
            "namespace_guard": card["source_namespaces"],
            "deliberately_not_claimed": [
                "causal explanation from four K points",
                "confirmatory selector evidence",
                "cross-model or cross-task generality",
                "sample-level join between 512 direct-probe and 14042 lm-eval namespaces",
            ],
        }
    )
    markdown = render_markdown(latent, absolute, contrasts, atlas)
    return {"latent": latent, "absolute": absolute, "contrasts": contrasts, "atlas": atlas, "markdown": markdown}


def render_markdown(latent: Mapping[str, Any], absolute: Mapping[str, Any], contrasts: Mapping[str, Any], atlas: Mapping[str, Any]) -> str:
    lines = [
        "# LoopScope Gate D：12:15 fixed-horizon 逐步机制图谱",
        "",
        "本图谱是冻结 12:15、K1–K4、总 horizon=1 的事后探索性机制分析，不是 selector 的确认性证据，也不从四个 K 点宣称总体相关或因果。",
        "",
        "## 既有 full-14042 结果",
        "",
        "| K | Accuracy (%) | Mean entropy | Mean top margin | Mean correct margin | JS to K1 | Top1 retained |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in absolute["kwise"]:
        lines.append(
            "| {k} | {accuracy_percent:.6f} | {mean_entropy_nats:.6f} | {mean_top_margin_raw:.6f} | {mean_correct_margin_raw:.6f} | {mean_js_to_k1_nats:.6f} | {top1_retained_vs_k1_fraction:.6f} |".format(**row)
        )
    lines.extend(["", "## 相邻 full contrasts", "", "| Contrast | Δ accuracy (pp) | WR | RW | RR | WW | Mean Δentropy | Mean Δtop-margin | Wrong overconfidence |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in contrasts["contrasts"]:
        transitions = row["mechanism"]["transitions"]
        over = row["mechanism"]["wrong_overconfidence"]
        lines.append(
            "| {name} | {delta:.6f} | {wr} | {rw} | {rr} | {ww} | {entropy:.6f} | {margin:.6f} | {over:.6f} |".format(
                name=row["contrast"],
                delta=row["accuracy"]["delta_acc_pp"],
                wr=transitions["wrong_to_right"], rw=transitions["right_to_wrong"],
                rr=transitions["right_to_right"], ww=transitions["wrong_to_wrong"],
                entropy=row["mechanism"]["mean_entropy_delta_nats"],
                margin=row["mechanism"]["mean_top_margin_delta_raw"],
                over=(over["fraction"] if over["fraction"] is not None else 0.0),
            )
        )
    lines.extend(["", "## validation-512 逐步轨迹（均值）", "", "| K | t | τ entry→applied | raw r | q | cosine | NCA | applied r | raw ΔH | applied ΔH | flatness | raw KL drop | applied KL drop | eRank entry/raw/applied |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for cell in latent["kwise"]:
        for step in cell["steps"]:
            m = step["metrics"]
            mean = lambda name: (m[name]["mean"] if m[name] is not None else None)
            fmt = lambda value: "NA" if value is None else "%.6f" % value
            erank = step["cohort_effective_rank"]
            lines.append(
                "| {k} | {t} | {te:.3f}→{ta:.3f} | {r} | {q} | {c} | {nca} | {ar} | {rh} | {ah} | {flat} | {rk} | {ak} | {ee:.3f}/{er:.3f}/{ea:.3f} |".format(
                    k=cell["k"], t=step["body_call_t"], te=step["tau_entry"], ta=step["tau_applied"],
                    r=fmt(mean("raw_relative_activity")), q=fmt(mean("residual_ratio_to_previous")),
                    c=fmt(mean("adjacent_residual_cosine")), nca=fmt(mean("repeated_step_nca")),
                    ar=fmt(mean("h_scaled_applied_relative_magnitude")), rh=fmt(mean("raw_entropy_drop_nats")),
                    ah=fmt(mean("applied_entropy_drop_nats")), flat=fmt(mean("raw_call_entropy_flatness_negative_population_std")),
                    rk=fmt(mean("raw_kl_to_reference_drop_nats")), ak=fmt(mean("applied_kl_to_reference_drop_nats")),
                    ee=erank["entry_effective_rank"], er=erank["raw_exit_effective_rank"], ea=erank["applied_state_effective_rank"],
                )
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- validation-512 的 correctness/flip 与分组只作 development diagnostic；真实 accuracy/flip/confidence 来自既有 full-14042 lm-eval sidecars。",
            "- intermediate lens 只描述冻结 final norm 与 A/B/C/D output weights 下的辅助 choice-lens，不等同于模型最终输出。",
            "- 本实验没有新增 full MMLU，没有测试其他 window、K>4、alpha、模型或任务，也没有拟合阈值、复合分数或 selector。",
            "- observed best full K=%s；该事实不自动构成因果机制或部署规则。" % atlas["headline"]["best_observed_full_accuracy_k"],
            "",
        ]
    )
    return "\n".join(lines)


def materialize_atlas(output_dir: Path, card: Mapping[str, Any], source_manifest: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(output_dir / "fixed_horizon_stepwise_card.json", card)
    atomic_write_new_json(output_dir / "source_manifest.json", source_manifest)
    atomic_write_new_json(output_dir / "kwise_latent_trajectory.json", outputs["latent"])
    atomic_write_new_json(output_dir / "kwise_absolute_full_summary.json", outputs["absolute"])
    atomic_write_new_json(output_dir / "adjacent_full_contrasts.json", outputs["contrasts"])
    atomic_write_new_json(output_dir / "fixed_horizon_trajectory_atlas.json", outputs["atlas"])
    markdown_path = output_dir / "fixed_horizon_trajectory_atlas.md"
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(outputs["markdown"])


def verify_atlas(output_dir: Path) -> Mapping[str, Any]:
    card = _load_hashed(output_dir / "fixed_horizon_stepwise_card.json")
    source_manifest = _load_hashed(output_dir / "source_manifest.json")
    expected = assemble_outputs(card, source_manifest)
    observed = {
        "latent": _load_hashed(output_dir / "kwise_latent_trajectory.json"),
        "absolute": _load_hashed(output_dir / "kwise_absolute_full_summary.json"),
        "contrasts": _load_hashed(output_dir / "adjacent_full_contrasts.json"),
        "atlas": _load_hashed(output_dir / "fixed_horizon_trajectory_atlas.json"),
        "markdown": (output_dir / "fixed_horizon_trajectory_atlas.md").read_text(encoding="utf-8"),
    }
    if observed != expected:
        raise FixedHorizonError("atlas verifier recomputation differs from write-once outputs")
    files = {}
    for name in (
        "fixed_horizon_stepwise_card.json",
        "source_manifest.json",
        "kwise_latent_trajectory.json",
        "kwise_absolute_full_summary.json",
        "adjacent_full_contrasts.json",
        "fixed_horizon_trajectory_atlas.json",
        "fixed_horizon_trajectory_atlas.md",
    ):
        files[name] = file_sha256(output_dir / name)
    receipt = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-atlas-verifier-receipt.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "verified_at_utc": utc_now(),
            "recomputed_from_sources": True,
            "all_outputs_equal": True,
            "file_sha256": files,
            "verifier_execution_count": 1,
        }
    )
    atomic_write_new_json(output_dir / "verifier_receipt.json", receipt)
    return receipt


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source-manifest", help="bind D2 and existing Gate B/C/Phase 1 evidence")
    source.add_argument("--card", required=True)
    source.add_argument("--profile-manifest", required=True)
    source.add_argument("--gate-c-analysis-input", required=True)
    source.add_argument("--gate-c-report", required=True)
    source.add_argument("--phase1-selection-report", required=True)
    source.add_argument("--output", required=True)
    atlas = sub.add_parser("atlas", help="execute the one write-once atlas analysis")
    atlas.add_argument("--card", required=True)
    atlas.add_argument("--source-manifest", required=True)
    atlas.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify", help="execute the one source-aware atlas verifier")
    verify.add_argument("--output-dir", required=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.command == "source-manifest":
        manifest = build_source_manifest(
            card_path=Path(args.card),
            profile_manifest_path=Path(args.profile_manifest),
            gate_c_analysis_input_path=Path(args.gate_c_analysis_input),
            gate_c_report_path=Path(args.gate_c_report),
            phase1_selection_report_path=Path(args.phase1_selection_report),
        )
        atomic_write_new_json(Path(args.output), manifest)
        print(manifest["manifest_sha256"])
        return 0
    if args.command == "verify":
        receipt = verify_atlas(Path(args.output_dir).resolve())
        print(receipt["manifest_sha256"])
        return 0
    card = _load_hashed(Path(args.card))
    validate_fixed_horizon_card(card)
    source_manifest = _load_hashed(Path(args.source_manifest))
    outputs = assemble_outputs(card, source_manifest)
    materialize_atlas(Path(args.output_dir).resolve(), card, source_manifest, outputs)
    print(outputs["atlas"]["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
