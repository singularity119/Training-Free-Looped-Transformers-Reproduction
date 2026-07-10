"""Artifact schemas and write-once helpers for LoopScope phase one."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


PROBE_SCHEMA_VERSION = "loopscope.probe.v2"
WINDOW_GRID_SCHEMA_VERSION = "loopscope.window-grid.v1"
SELECTION_SCHEMA_VERSION = "loopscope.selection.v1"
ANALYSIS_SCHEMA_VERSION = "loopscope.analysis.v1"


class SchemaError(ValueError):
    """Raised when a LoopScope artifact violates its public contract."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes and reject NaN/Inf."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any], hash_field: str = "manifest_sha256") -> str:
    """Hash a manifest while excluding its top-level self-hash field."""

    body = {key: value for key, value in payload.items() if key != hash_field}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def attach_manifest_sha256(
    payload: MutableMapping[str, Any], hash_field: str = "manifest_sha256"
) -> MutableMapping[str, Any]:
    payload[hash_field] = manifest_sha256(payload, hash_field=hash_field)
    return payload


def verify_manifest_sha256(
    payload: Mapping[str, Any], hash_field: str = "manifest_sha256"
) -> None:
    expected = payload.get(hash_field)
    actual = manifest_sha256(payload, hash_field=hash_field)
    if not isinstance(expected, str) or expected != actual:
        raise SchemaError(
            "%s mismatch: expected %r, computed %s" % (hash_field, expected, actual)
        )


def ensure_new_directory(path: Path) -> Path:
    """Create an output directory only when no object already exists there."""

    if path.exists():
        raise FileExistsError("refusing to reuse existing output directory: %s" % path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_new_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact without replacing an existing file."""

    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def validate_probe_report(payload: Mapping[str, Any]) -> None:
    """Validate the common envelope plus layer/window report invariants."""

    required = {
        "schema_version",
        "git",
        "model",
        "tokenizer",
        "runtime",
        "probe_pool",
        "position_rule",
        "boundary_metrics",
        "window_metrics",
        "warnings",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise SchemaError("probe report missing fields: %s" % ", ".join(missing))
    if payload["schema_version"] != PROBE_SCHEMA_VERSION:
        raise SchemaError("unsupported probe schema_version: %r" % payload["schema_version"])
    for key in ("git", "model", "tokenizer", "runtime", "probe_pool"):
        if not isinstance(payload[key], Mapping):
            raise SchemaError("probe report %s must be an object" % key)
    for key in ("boundary_metrics", "window_metrics", "warnings"):
        if not isinstance(payload[key], list):
            raise SchemaError("probe report %s must be a list" % key)
    _reject_non_finite(payload)
    _validate_probe_provenance(payload)
    boundary_metrics = payload["boundary_metrics"]
    window_metrics = payload["window_metrics"]
    if bool(boundary_metrics) == bool(window_metrics):
        raise SchemaError(
            "probe report must contain exactly one of boundary_metrics/window_metrics"
        )
    if boundary_metrics:
        _validate_boundary_probe(payload)
    else:
        _validate_window_probe(payload)


def probe_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a scalar-only summary compatible with the existing report CLI."""

    validate_probe_report(payload)
    probe_pool = payload["probe_pool"]
    model = payload["model"]
    windows = payload["window_metrics"]
    valid_windows = sum(
        1 for item in windows if isinstance(item, Mapping) and bool(item.get("valid"))
    )
    metrics: Dict[str, Any] = {
        "schema_version": payload["schema_version"],
        "model_alias": str(model.get("alias", "")),
        "model_revision": str(model.get("revision") or ""),
        "layer_count": int(model.get("layer_count", 0) or 0),
        "probe_count": int(probe_pool.get("count", 0) or 0),
        "boundary_metric_count": len(payload["boundary_metrics"]),
        "window_metric_count": len(windows),
        "valid_window_count": valid_windows,
        "warning_count": len(payload["warnings"]),
    }
    if any(not isinstance(value, (str, int, float, bool)) for value in metrics.values()):
        raise SchemaError("probe summary contains a non-scalar metric")
    return {"results": {"loopscope_probe": metrics}}


def require_keys(payload: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys).difference(payload))
    if missing:
        raise SchemaError("%s missing fields: %s" % (context, ", ".join(missing)))


def _validate_probe_provenance(payload: Mapping[str, Any]) -> None:
    git = payload["git"]
    model = payload["model"]
    tokenizer = payload["tokenizer"]
    runtime = payload["runtime"]
    pool = payload["probe_pool"]
    require_keys(git, ("root", "branch", "commit", "dirty"), "git")
    require_keys(
        model,
        ("alias", "repo_id", "revision", "layer_count"),
        "model",
    )
    require_keys(tokenizer, ("revision", "choice_token_ids"), "tokenizer")
    require_keys(runtime, ("device", "dtype", "versions"), "runtime")
    require_keys(
        pool,
        (
            "source",
            "split",
            "count",
            "seed",
            "manifest_sha256",
            "source_manifest_sha256",
            "selected_subset_sha256",
            "source_manifest_count",
            "sample_ids",
            "records",
            "task_group",
            "num_fewshot",
            "uses_target_gold_labels",
            "fewshot_answers_present",
            "renderer",
            "render_contract_sha256",
            "rendering_records",
            "source_render_contract_subset_sha256",
            "selected_render_contract_subset_sha256",
        ),
        "probe_pool",
    )
    for context, mapping, keys in (
        ("git", git, ("root", "branch", "commit")),
        ("model", model, ("alias", "repo_id", "revision")),
        ("tokenizer", tokenizer, ("revision",)),
        ("runtime", runtime, ("device", "dtype")),
        ("probe_pool", pool, ("source", "split")),
    ):
        for key in keys:
            if not str(mapping.get(key) or "").strip():
                raise SchemaError("%s.%s must be non-empty" % (context, key))
    if not isinstance(git["dirty"], bool):
        raise SchemaError("git.dirty must be boolean")
    layer_count = int(model["layer_count"])
    if layer_count < 1:
        raise SchemaError("model.layer_count must be positive")
    versions = runtime["versions"]
    if not isinstance(versions, Mapping):
        raise SchemaError("runtime.versions must be an object")
    require_keys(
        versions,
        ("python", "torch", "transformers", "lm_eval", "tflt"),
        "runtime.versions",
    )
    for key in ("python", "torch", "transformers", "lm_eval", "tflt"):
        if not str(versions.get(key) or "").strip():
            raise SchemaError("runtime.versions.%s must be non-empty" % key)
    count = int(pool["count"])
    source_count = int(pool["source_manifest_count"])
    if count < 1 or source_count < count:
        raise SchemaError("probe_pool count/source_manifest_count are inconsistent")
    if not isinstance(pool["seed"], int):
        raise SchemaError("probe_pool.seed must be an integer")
    if "test" in str(pool["split"]).lower():
        raise SchemaError("probe_pool.split must not contain test")
    if pool["task_group"] != "mmlu" or pool["num_fewshot"] != 5:
        raise SchemaError("probe_pool must use MMLU five-shot rendering")
    if pool["uses_target_gold_labels"] is not False:
        raise SchemaError("probe_pool must exclude target gold labels")
    if pool["fewshot_answers_present"] is not True:
        raise SchemaError("probe_pool must retain five-shot answers")
    renderer = pool["renderer"]
    if not isinstance(renderer, Mapping) or renderer.get("lm_eval_version") != "0.4.11":
        raise SchemaError("probe_pool.renderer must identify lm-eval 0.4.11")
    renderer_hash_fields = (
        "renderer_source_sha256",
        "source_files_sha256",
        "template_sha256",
        "task_configs_sha256",
        "render_contract_sha256",
        "dataset_fingerprint_sha256",
        "source_projection_sha256",
        "render_sha256",
        "renderer_manifest_sha256",
    )
    for key in renderer_hash_fields:
        _validate_sha256(renderer.get(key), "probe_pool.renderer.%s" % key)
    revision = str(renderer.get("dataset_revision") or "")
    if len(revision) < 40 or any(char not in "0123456789abcdef" for char in revision):
        raise SchemaError("probe_pool.renderer.dataset_revision must be an exact commit")
    if renderer["renderer_source_sha256"] != renderer["source_files_sha256"]:
        raise SchemaError("probe_pool renderer source-file hashes disagree")
    if renderer["template_sha256"] != renderer["task_configs_sha256"]:
        raise SchemaError("probe_pool renderer task-template hashes disagree")
    if pool["render_contract_sha256"] != renderer["render_contract_sha256"]:
        raise SchemaError("probe_pool render contract disagrees with renderer evidence")
    for key in (
        "manifest_sha256",
        "source_manifest_sha256",
        "selected_subset_sha256",
        "source_render_contract_subset_sha256",
        "selected_render_contract_subset_sha256",
        "render_contract_sha256",
    ):
        _validate_sha256(pool[key], "probe_pool.%s" % key)
    if pool["manifest_sha256"] != pool["selected_subset_sha256"]:
        raise SchemaError("probe_pool manifest_sha256 must identify the selected subset")
    sample_ids = pool["sample_ids"]
    if not isinstance(sample_ids, list) or len(sample_ids) != count:
        raise SchemaError("probe_pool.sample_ids length must equal count")
    if len(set(str(item) for item in sample_ids)) != len(sample_ids):
        raise SchemaError("probe_pool.sample_ids must be unique")
    records = pool["records"]
    if not isinstance(records, list) or len(records) != count:
        raise SchemaError("probe_pool.records length must equal count")
    record_ids = []
    for item in records:
        if not isinstance(item, Mapping):
            raise SchemaError("probe_pool.records entries must be objects")
        require_keys(item, ("id", "prompt_sha256"), "probe_pool.records")
        record_ids.append(str(item["id"]))
        _validate_sha256(item["prompt_sha256"], "probe_pool.records.prompt_sha256")
    if record_ids != [str(item) for item in sample_ids]:
        raise SchemaError("probe_pool records must match sample_ids order")
    rendering_records = pool["rendering_records"]
    if not isinstance(rendering_records, list) or len(rendering_records) != count:
        raise SchemaError("probe_pool.rendering_records length must equal count")
    render_hash = hashlib.sha256(canonical_json_bytes(rendering_records)).hexdigest()
    if render_hash != pool["selected_render_contract_subset_sha256"]:
        raise SchemaError("probe_pool rendering-contract subset hash mismatch")
    selected_manifest = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": pool["source"],
        "split": pool["split"],
        "count": count,
        "seed": pool["seed"],
        "sample_ids": sample_ids,
        "records": records,
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": pool["renderer"],
        "render_contract_sha256": pool["render_contract_sha256"],
        "rendering_records": rendering_records,
        "render_contract_subset_sha256": pool[
            "selected_render_contract_subset_sha256"
        ],
        "source_manifest_sha256": pool["source_manifest_sha256"],
    }
    if manifest_sha256(selected_manifest) != pool["selected_subset_sha256"]:
        raise SchemaError("probe_pool selected-subset hash does not match its records")


def _validate_boundary_probe(payload: Mapping[str, Any]) -> None:
    layer_count = int(payload["model"]["layer_count"])
    pool_count = int(payload["probe_pool"]["count"])
    contract = payload.get("boundary_contract")
    if not isinstance(contract, Mapping):
        raise SchemaError("boundary probe requires boundary_contract")
    require_keys(
        contract,
        ("version", "definition", "boundary_count", "window_entry", "window_exit"),
        "boundary_contract",
    )
    if contract["version"] != "loopscope.boundary.v1":
        raise SchemaError("unsupported boundary_contract.version")
    if int(contract["boundary_count"]) != layer_count + 1:
        raise SchemaError("boundary_contract.boundary_count must equal layer_count + 1")
    if contract["window_entry"] != "B_a" or contract["window_exit"] != "B_(b+1)":
        raise SchemaError("boundary contract must score inclusive [a,b] as B_a to B_(b+1)")
    metrics = payload["boundary_metrics"]
    indexes = [
        int(item.get("boundary_index", -1)) for item in metrics if isinstance(item, Mapping)
    ]
    if len(metrics) != layer_count + 1 or indexes != list(range(layer_count + 1)):
        raise SchemaError("boundary_metrics must contain B_0..B_N exactly once in order")
    choice_ids = payload["tokenizer"]["choice_token_ids"]
    if not isinstance(choice_ids, Mapping) or len(choice_ids) < 2:
        raise SchemaError("boundary probe requires at least two choice-token entries")
    for boundary_index, item in enumerate(metrics):
        if not isinstance(item, Mapping):
            raise SchemaError("boundary_metrics entries must be objects")
        require_keys(
            item,
            (
                "boundary_index",
                "after_layer",
                "before_layer",
                "choice_entropy",
                "kl_to_final",
                "top1_to_final_agreement",
                "effective_rank",
                "effective_rank_sampling",
            ),
            "boundary_metrics[%d]" % boundary_index,
        )
        expected_after = boundary_index - 1 if boundary_index > 0 else None
        expected_before = boundary_index if boundary_index < layer_count else None
        if item["after_layer"] != expected_after or item["before_layer"] != expected_before:
            raise SchemaError(
                "boundary_metrics[%d] after_layer/before_layer mapping is inconsistent"
                % boundary_index
            )
        for key in ("choice_entropy", "kl_to_final", "top1_to_final_agreement"):
            _validate_summary(
                item[key], pool_count, "boundary_metrics[%d].%s" % (boundary_index, key)
            )
        if float(item["effective_rank"]) <= 0.0:
            raise SchemaError("effective_rank must be positive")
        sampling = item["effective_rank_sampling"]
        if not isinstance(sampling, Mapping):
            raise SchemaError("effective_rank_sampling must be an object")
        require_keys(
            sampling,
            (
                "representation_space",
                "estimator",
                "estimator_version",
                "spectrum",
                "unit_normalized",
                "centered_across_vectors",
                "count",
                "sample_ids",
            ),
            "effective_rank_sampling",
        )
        if (
            sampling["estimator"] != "gram_spectrum_shannon_effective_rank"
            or sampling["estimator_version"] != "1"
            or sampling["spectrum"] != "squared_singular_values"
            or sampling["unit_normalized"] is not True
            or sampling["centered_across_vectors"] is not True
        ):
            raise SchemaError("effective_rank_sampling estimator contract mismatch")
        sampled = int(sampling["count"])
        if sampled < 2 or sampled > pool_count or len(sampling["sample_ids"]) != sampled:
            raise SchemaError("effective_rank_sampling count/sample_ids are inconsistent")
    examples = payload.get("examples")
    _validate_example_ids(
        examples, payload["probe_pool"]["sample_ids"], "boundary examples"
    )
    for example in examples:
        sample_boundaries = example.get("boundary_metrics")
        if not isinstance(sample_boundaries, list):
            raise SchemaError("boundary example is missing boundary_metrics")
        sample_indexes = [
            int(item.get("boundary_index", -1)) for item in sample_boundaries
        ]
        if sample_indexes != list(range(layer_count + 1)):
            raise SchemaError("boundary example metrics do not cover B_0..B_N in order")


def _validate_window_probe(payload: Mapping[str, Any]) -> None:
    pool_count = int(payload["probe_pool"]["count"])
    grid = payload.get("window_grid")
    if not isinstance(grid, Mapping):
        raise SchemaError("window probe requires window_grid provenance")
    require_keys(
        grid,
        ("manifest_sha256", "layer_count", "candidate_windows"),
        "window_grid",
    )
    _validate_sha256(grid["manifest_sha256"], "window_grid.manifest_sha256")
    if int(grid["layer_count"]) != int(payload["model"]["layer_count"]):
        raise SchemaError("window_grid.layer_count must match model.layer_count")
    candidates = grid["candidate_windows"]
    if not isinstance(candidates, list) or len(set(candidates)) != len(candidates):
        raise SchemaError("window_grid.candidate_windows must be a unique list")
    seen = set()
    for index, item in enumerate(payload["window_metrics"]):
        if not isinstance(item, Mapping):
            raise SchemaError("window_metrics entries must be objects")
        require_keys(
            item,
            (
                "window",
                "valid",
                "sample_count",
                "valid_sample_count",
                "answer_position",
                "all_non_padding_tokens",
                "examples",
                "errors",
            ),
            "window_metrics[%d]" % index,
        )
        window = str(item["window"])
        if window in seen or window not in candidates:
            raise SchemaError("window metric is duplicate or absent from frozen grid: %s" % window)
        seen.add(window)
        if int(item["sample_count"]) != pool_count:
            raise SchemaError("window sample_count must equal probe_pool.count")
        _validate_example_ids(
            item["examples"],
            payload["probe_pool"]["sample_ids"],
            "window examples",
            id_key="sample_id",
        )
        valid_examples = sum(1 for example in item["examples"] if example.get("valid"))
        if int(item["valid_sample_count"]) != valid_examples:
            raise SchemaError("valid_sample_count disagrees with examples")
        if bool(item["valid"]):
            if item["errors"] or valid_examples != pool_count:
                raise SchemaError("valid window must have no errors and all samples valid")
            for group in ("answer_position", "all_non_padding_tokens"):
                for metric in ("r", "q"):
                    summary = item[group].get(metric)
                    expected = pool_count if group == "answer_position" else None
                    _validate_summary(summary, expected, "%s.%s" % (group, metric))
                    if group == "all_non_padding_tokens" and int(summary["count"]) < pool_count:
                        raise SchemaError("all-token summary count must be at least pool count")
        elif not item["errors"]:
            raise SchemaError("invalid window must retain explicit errors")


def _validate_example_ids(
    examples: Any,
    expected_ids: Sequence[Any],
    context: str,
    id_key: str = "id",
) -> None:
    if not isinstance(examples, list):
        raise SchemaError("%s must be a list" % context)
    actual = [str(item.get(id_key, "")) for item in examples if isinstance(item, Mapping)]
    if actual != [str(item) for item in expected_ids]:
        raise SchemaError("%s IDs/order must match probe_pool.sample_ids" % context)


def _validate_summary(value: Any, expected_count: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise SchemaError("%s must be a summary object" % context)
    require_keys(value, ("count", "mean", "median", "p90"), context)
    count = int(value["count"])
    if count < 1 or (expected_count is not None and count != int(expected_count)):
        raise SchemaError("%s count is inconsistent" % context)


def _validate_sha256(value: Any, context: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SchemaError("%s must be a lowercase SHA-256 hex digest" % context)


def _reject_non_finite(value: Any, path: Sequence[str] = ()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError("non-finite number at %s" % ".".join(path))
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, tuple(path) + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, tuple(path) + (str(index),))
