"""Cell-aware, source-aware Phase 1 reuse and full-output provenance closure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase2_schema import (
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    PHASE1_RUN_ROOT,
    PHASE2_CARD_ID,
    PHASE2_REUSE_SCHEMA_VERSION,
    REUSE_STATUSES,
    SchemaError,
    b2_logical_cells,
    expected_full_producer_kind,
    file_sha256,
    make_full_final_output_envelope,
    make_hashed_manifest,
    phase1_reuse_full_cell_ids,
    phase2_new_full_cell_ids,
    stable_sample_identity,
    validate_contained_path,
    validate_full_final_output_envelope,
    validate_identity_manifest,
    validate_phase2_card,
    validate_phase2_workspace_output_path,
    validate_protocol_cell,
    verify_phase2_implementation_hashes,
)
from tflt.loopscope.schema import verify_manifest_sha256


PHASE1_REUSE_JOB_BY_CELL = {
    phase1_reuse_full_cell_ids()[0]: "baseline-full",
    phase1_reuse_full_cell_ids()[1]: "window-11-14-full",
    phase1_reuse_full_cell_ids()[2]: "window-12-15-full",
    phase1_reuse_full_cell_ids()[3]: "window-13-16-full",
}


class Phase2ReuseError(ValueError):
    """Raised when a full output cannot prove its actual immutable source."""


def full_cell_producer_partition(card: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Return and revalidate the closed four-reuse/eight-new partition."""

    validate_phase2_card(card)
    reuse = phase1_reuse_full_cell_ids()
    new = phase2_new_full_cell_ids()
    if set(reuse) & set(new) or len(set(reuse) | set(new)) != 12:
        raise Phase2ReuseError("full producer partitions overlap or are incomplete")
    return {"phase1_immutable_reuse_adapter": reuse, "lm_eval_logged_samples_adapter": new}


def join_lm_eval_logged_samples(
    result: Any, expected_identities: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Exact-join arbitrary-order lm-eval samples and restore canonical order."""

    samples_by_task = result.get("samples") if isinstance(result, Mapping) else None
    if not isinstance(samples_by_task, Mapping):
        raise Phase2ReuseError("lm-eval result does not expose logged samples")
    expected = [
        stable_sample_identity(item.get("task"), item.get("doc_id"), item.get("doc_hash"))
        for item in expected_identities
    ]
    expected_keys = {(item["task"], item["doc_id"], item["doc_hash"]) for item in expected}
    if len(expected_keys) != len(expected):
        raise Phase2ReuseError("canonical full identities contain duplicates")
    observed: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    from tflt.loopscope.phase2_analysis import choice_output

    for task, task_samples in samples_by_task.items():
        if not isinstance(task, str) or not isinstance(task_samples, list):
            raise Phase2ReuseError("lm-eval samples mapping is malformed")
        for sample in task_samples:
            if not isinstance(sample, Mapping):
                raise Phase2ReuseError("lm-eval logged sample must be an object")
            if "task" in sample and str(sample["task"]) != task:
                raise Phase2ReuseError("logged sample task differs from its task bucket")
            try:
                sample_identity = stable_sample_identity(
                    task, sample["doc_id"], sample["doc_hash"]
                )
            except (KeyError, TypeError, ValueError, SchemaError) as exc:
                raise Phase2ReuseError(
                    "lm-eval sample lacks stable task/doc_id/doc_hash identity"
                ) from exc
            key = (sample_identity["task"], sample_identity["doc_id"], sample_identity["doc_hash"])
            if key in observed:
                raise Phase2ReuseError("lm-eval logged samples contain duplicate stable identity")
            scores = extract_four_raw_choice_scores(sample)
            gold_index = sample.get("target")
            if isinstance(gold_index, str) and gold_index in ("A", "B", "C", "D"):
                gold_index = ("A", "B", "C", "D").index(gold_index)
            if (
                isinstance(gold_index, bool)
                or not isinstance(gold_index, int)
                or not 0 <= gold_index < 4
            ):
                raise Phase2ReuseError("lm-eval sample lacks a frozen A-D gold index")
            try:
                observed[key] = choice_output(
                    scores,
                    identity=sample_identity,
                    score_source=FULL_LM_EVAL_SCORE_SOURCE,
                    gold_index=gold_index,
                    evaluator_acc=extract_sample_acc_none(sample),
                )
            except ValueError as exc:
                raise Phase2ReuseError(
                    "lm-eval raw choice/evaluator correctness closure failed"
                ) from exc
    if set(observed) != expected_keys:
        missing = len(expected_keys - set(observed))
        extra = len(set(observed) - expected_keys)
        raise Phase2ReuseError(
            "lm-eval exact identity join is incomplete; missing=%d extra=%d" % (missing, extra)
        )
    return [observed[(item["task"], item["doc_id"], item["doc_hash"])] for item in expected]


def extract_four_raw_choice_scores(sample: Mapping[str, Any]) -> List[float]:
    values = sample.get("filtered_resps")
    if isinstance(values, list) and len(values) == 4 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in values
    ):
        return [float(item) for item in values]
    values = sample.get("resps")
    scores: List[float] = []
    if isinstance(values, list) and len(values) == 4:
        for response in values:
            current = response
            while isinstance(current, (list, tuple)) and len(current) == 1:
                current = current[0]
            if isinstance(current, (list, tuple)) and current:
                current = current[0]
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                break
            scores.append(float(current))
    if len(scores) != 4:
        raise Phase2ReuseError(
            "lm-eval sample lacks exactly four raw choice log-likelihood scores"
        )
    return scores


def extract_sample_acc_none(sample: Mapping[str, Any]) -> bool:
    metrics = sample.get("metrics")
    if isinstance(metrics, Mapping):
        value = metrics.get("acc,none")
    elif metrics == ["acc"] and "acc,none" not in sample:
        # lm-eval 0.4.11 serializes the per-sample default-filter metric as
        # metrics=["acc"] plus a top-level binary ``acc`` value, while the
        # corresponding task aggregate remains ``acc,none``.  Accept only
        # that exact unambiguous logged-sample shape.
        value = sample.get("acc")
    else:
        value = sample.get("acc,none")
    if value not in (0, 1, False, True):
        raise Phase2ReuseError("lm-eval sample lacks binary acc,none")
    return bool(value)


def load_phase1_immutable_reuse_source(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    run_manifest_path: Path,
    *,
    expected_source_evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Reload one real Phase 1 baseline/K2 source and derive canonical choice rows."""

    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    if identity_manifest["identity_namespace"] != FULL_IDENTITY_NAMESPACE:
        raise Phase2ReuseError("Phase 1 reuse requires the full identity namespace")
    validate_protocol_cell(cell, allow_baseline=True)
    if expected_full_producer_kind(cell) != "phase1_immutable_reuse_adapter":
        raise Phase2ReuseError("only the four frozen baseline/K2 cells may reuse Phase 1")
    run_root = Path(PHASE1_RUN_ROOT)
    expected_manifest = run_root / "control" / "phase1_run_manifest.json"
    manifest_path = validate_contained_path(
        run_manifest_path,
        run_root,
        context="Phase 1 immutable run manifest",
        exact_relative="control/phase1_run_manifest.json",
    )
    if manifest_path != expected_manifest.resolve(strict=False):
        raise Phase2ReuseError("Phase 1 run manifest differs from the canonical root")
    run_manifest = _load_json_object(manifest_path, "Phase 1 run manifest")
    verify_manifest_sha256(run_manifest)
    source_provenance = identity_manifest["source_provenance"]
    if (
        source_provenance["source_manifest_path"] != str(expected_manifest)
        or source_provenance["source_manifest_sha256"] != run_manifest["manifest_sha256"]
    ):
        raise Phase2ReuseError("full identity manifest is not bound to the loaded Phase 1 run")
    jobs = run_manifest.get("stages", {}).get("gate-e-full", {}).get("jobs")
    if not isinstance(jobs, list):
        raise Phase2ReuseError("Phase 1 run manifest lacks gate-e-full jobs")
    expected_job_id = PHASE1_REUSE_JOB_BY_CELL[cell["cell_id"]]
    matches = [job for job in jobs if isinstance(job, Mapping) and job.get("job_id") == expected_job_id]
    if len(matches) != 1:
        raise Phase2ReuseError("Phase 1 run manifest has no unique mapped reuse job")
    job = matches[0]
    output_root = run_root / "gate-e-full" / expected_job_id
    if (
        job.get("stage") != "gate-e-full"
        or job.get("output_dir") != str(output_root)
        or job.get("automatic_retry") is not False
    ):
        raise Phase2ReuseError("Phase 1 cell-to-output mapping differs from the immutable manifest")
    _validate_phase1_job_argv(job.get("argv"), card, cell, output_root)
    paths = {
        "source_command_args": validate_contained_path(
            output_root / "command_args.json", output_root,
            context="Phase 1 command_args", exact_relative="command_args.json"
        ),
        "source_environment": validate_contained_path(
            output_root / "env.json", output_root,
            context="Phase 1 environment", exact_relative="env.json"
        ),
        "source_revision_report": validate_contained_path(
            output_root / "model_revision.json", output_root,
            context="Phase 1 revision", exact_relative="model_revision.json"
        ),
        "source_results": validate_contained_path(
            output_root / "results.json", output_root,
            context="Phase 1 results", exact_relative="results.json"
        ),
    }
    loaded = {key: _load_json_object(path, key) for key, path in paths.items()}
    _validate_phase1_command_args(loaded["source_command_args"], card, cell, output_root)
    _validate_revision(loaded["source_revision_report"], card)
    fixed_evidence = {
        "source_run_manifest": _manifest_ref(manifest_path, run_manifest),
        **{key: _raw_ref(path) for key, path in paths.items()},
    }
    if expected_source_evidence is not None:
        for key, ref in fixed_evidence.items():
            if expected_source_evidence.get(key) != ref:
                raise Phase2ReuseError(
                    "Phase 1 source paths/hashes differ from the frozen reuse request"
                )
    sample_result, sample_source = _load_phase1_logged_samples(
        loaded["source_results"], paths["source_results"], output_root
    )
    rows = join_lm_eval_logged_samples(
        sample_result, identity_manifest["ordered_sample_identity"]
    )
    evidence = {**fixed_evidence, "sample_source": sample_source}
    if expected_source_evidence is not None and evidence != expected_source_evidence:
        raise Phase2ReuseError(
            "Phase 1 source paths/hashes differ from the frozen reuse request"
        )
    return {
        "job_id": expected_job_id,
        "source_output_root": str(output_root.resolve(strict=False)),
        "samples": rows,
        "evidence": evidence,
    }


def build_phase1_reuse_full_envelope(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    run_manifest_path: Path,
    artifact_root: Path,
    reuse_request: Mapping[str, str],
    attempt_manifest: Mapping[str, str],
    receipt_manifest: Mapping[str, str],
    command_args: Mapping[str, str],
    environment: Mapping[str, str],
) -> Dict[str, Any]:
    """Build, but do not overwrite, a sidecar from one validated immutable source."""

    resolved_artifact_root = validate_phase2_workspace_output_path(
        artifact_root, card, context="Phase 1 reuse artifact root"
    )
    request_path = validate_contained_path(
        reuse_request["path"],
        card["write_once_contract"]["workspace_root"],
        context="Phase 1 reuse request",
    )
    request_payload = _load_json_object(request_path, "Phase 1 reuse request")
    verify_manifest_sha256(request_payload)
    if request_payload.get("manifest_sha256") != reuse_request["sha256"]:
        raise Phase2ReuseError("Phase 1 reuse request hash differs")
    if not isinstance(request_payload.get("source_evidence"), Mapping):
        raise Phase2ReuseError("Phase 1 reuse request lacks frozen source evidence")
    source = load_phase1_immutable_reuse_source(
        card,
        identity_manifest,
        cell,
        run_manifest_path,
        expected_source_evidence=request_payload.get("source_evidence"),
    )
    _validate_reuse_request_payload(
        request_payload,
        card=card,
        identity_manifest=identity_manifest,
        cell=cell,
        source_evidence=source["evidence"],
        attempt_manifest=attempt_manifest,
        receipt_manifest=receipt_manifest,
        output_root=resolved_artifact_root,
    )
    evidence = {
        "reuse_request": dict(reuse_request),
        "attempt_manifest": dict(attempt_manifest),
        "receipt_manifest": dict(receipt_manifest),
        "command_args": dict(command_args),
        "environment": dict(environment),
        **source["evidence"],
    }
    producer = {
        "producer_kind": "phase1_immutable_reuse_adapter",
        "attempt_manifest_sha256": attempt_manifest["sha256"],
        "receipt_manifest_sha256": receipt_manifest["sha256"],
        "command_sha256": command_args["sha256"],
        "environment_sha256": environment["sha256"],
        "revision_report_sha256": evidence["source_revision_report"]["sha256"],
        "results_sha256": evidence["source_results"]["sha256"],
        "source_manifest_sha256": evidence["source_run_manifest"]["sha256"],
    }
    return make_full_final_output_envelope(
        card,
        identity_manifest,
        cell=cell,
        samples=source["samples"],
        producer=producer,
        artifact_root=str(resolved_artifact_root),
        producer_evidence=evidence,
    )


def verify_full_final_output_artifact(
    sidecar_path: Path,
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reload a full sidecar and every actual producer/source file it claims."""

    governed_sidecar = validate_contained_path(
        sidecar_path,
        card["write_once_contract"]["workspace_root"],
        context="full final-output sidecar",
    )
    sidecar = _load_json_object(governed_sidecar, "full final-output sidecar")
    validate_full_final_output_envelope(sidecar, card, identity_manifest)
    root = validate_phase2_workspace_output_path(
        sidecar["artifact_root"], card, context="full final-output artifact root"
    )
    actual_sidecar = validate_contained_path(
        governed_sidecar,
        root,
        context="full final-output sidecar",
        exact_relative="phase2_final_outputs.json",
    )
    if actual_sidecar != Path(sidecar_path).resolve(strict=False):
        raise Phase2ReuseError("full sidecar path differs from its resolved file")
    kind = sidecar["producer"]["producer_kind"]
    if kind == "lm_eval_logged_samples_adapter":
        rows = _verify_new_full_source(sidecar, card, identity_manifest, root)
    elif kind == "phase1_immutable_reuse_adapter":
        rows = _verify_reuse_full_source(sidecar, card, identity_manifest, root)
    else:
        raise Phase2ReuseError("unsupported full producer kind")
    if rows != sidecar["samples"]:
        raise Phase2ReuseError("full sidecar rows differ from reloaded producer samples")
    return sidecar


def _verify_new_full_source(
    sidecar: Mapping[str, Any],
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    evidence = sidecar["producer_evidence"]
    workspace_root = Path(card["write_once_contract"]["workspace_root"])
    for key in ("adapter_request", "attempt_manifest", "receipt_manifest"):
        validate_contained_path(
            evidence[key]["path"], workspace_root, context="new full %s" % key
        )
    for key, filename in (
        ("command_args", "command_args.json"),
        ("environment", "env.json"),
        ("revision_report", "model_revision.json"),
        ("results", "results.json"),
    ):
        _require_exact_output_ref(evidence[key], output_root, filename, key)
    loaded = _load_full_evidence_files(evidence)
    attempt = loaded["attempt_manifest"]
    receipt = loaded["receipt_manifest"]
    authorization_root = _validate_full_control_packets(
        attempt, receipt, card=card, cell=sidecar["cell"], output_root=output_root,
        producer_kind="lm_eval_logged_samples_adapter"
    )
    _require_governed_ref(evidence["adapter_request"], authorization_root, "adapter request")
    _require_governed_ref(evidence["attempt_manifest"], authorization_root, "attempt manifest")
    _require_governed_ref(evidence["receipt_manifest"], authorization_root, "receipt manifest")
    for key, filename in (
        ("command_args", "command_args.json"),
        ("environment", "env.json"),
        ("revision_report", "model_revision.json"),
        ("results", "results.json"),
    ):
        _require_exact_output_ref(evidence[key], output_root, filename, key)
    request = loaded["adapter_request"]
    verify_manifest_sha256(request)
    request_keys = {
        "schema_version", "artifact_kind", "card", "identity_manifest", "cell",
        "attempt_manifest", "receipt_manifest", "identity_join", "score_source",
        "output_filename", "manifest_sha256",
    }
    card_ref = request.get("card")
    identity_ref = request.get("identity_manifest")
    refs_valid = (
        isinstance(card_ref, Mapping)
        and set(card_ref) == {"path", "sha256"}
        and card_ref["sha256"] == card["manifest_sha256"]
        and isinstance(identity_ref, Mapping)
        and set(identity_ref) == {"path", "sha256"}
        and identity_ref["sha256"] == identity_manifest["manifest_sha256"]
    )
    if (
        set(request) != request_keys
        or request.get("schema_version") != "loopscope.phase2-final-output-adapter-manifest.v2"
        or request.get("artifact_kind") != "full_final_output_adapter_request"
        or not refs_valid
        or request.get("cell") != sidecar["cell"]
        or request.get("identity_join") != "unordered_exact_task_doc_id_doc_hash_to_canonical_manifest"
        or request.get("score_source") != FULL_LM_EVAL_SCORE_SOURCE
        or request.get("output_filename") != "phase2_final_outputs.json"
    ):
        raise Phase2ReuseError("adapter request differs from the full sidecar")
    request_base = Path(evidence["adapter_request"]["path"]).parent
    for ref_key, expected_payload in (("card", card), ("identity_manifest", identity_manifest)):
        internal = request[ref_key]
        internal_path = Path(internal["path"])
        resolved = (
            internal_path.resolve(strict=False)
            if internal_path.is_absolute()
            else (request_base / internal_path).resolve(strict=False)
        )
        validate_contained_path(
            resolved,
            authorization_root,
            context="adapter %s" % ref_key,
        )
        loaded_payload = _load_json_object(resolved, "adapter %s" % ref_key)
        verify_manifest_sha256(loaded_payload)
        if loaded_payload != expected_payload or loaded_payload["manifest_sha256"] != internal["sha256"]:
            raise Phase2ReuseError("adapter %s ref differs from canonical evidence" % ref_key)
    for key, expected_ref in (
        ("attempt_manifest", evidence["attempt_manifest"]),
        ("receipt_manifest", evidence["receipt_manifest"]),
    ):
        internal = request[key]
        if not isinstance(internal, Mapping) or set(internal) != {"path", "sha256"}:
            raise Phase2ReuseError("adapter request has malformed %s ref" % key)
        internal_path = Path(internal["path"])
        internal_resolved = (
            internal_path.resolve(strict=False)
            if internal_path.is_absolute()
            else (request_base / internal_path).resolve(strict=False)
        )
        if internal_resolved != Path(expected_ref["path"]).resolve(strict=False) or (
            internal["sha256"] != expected_ref["sha256"]
        ):
            raise Phase2ReuseError("adapter request %s ref differs from producer evidence" % key)
    _validate_full_command_args(
        loaded["command_args"], card, sidecar["cell"], output_root,
        manifest_path=evidence["adapter_request"]["path"]
    )
    _validate_revision(loaded["revision_report"], card)
    if evidence["sample_source"] != {
        "kind": "results_inline_samples", "artifacts": [evidence["results"]]
    }:
        raise Phase2ReuseError("new-run full source must be its exact results.json")
    return join_lm_eval_logged_samples(
        loaded["results"], identity_manifest["ordered_sample_identity"]
    )


def _verify_reuse_full_source(
    sidecar: Mapping[str, Any],
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    evidence = sidecar["producer_evidence"]
    workspace_root = Path(card["write_once_contract"]["workspace_root"])
    for key in ("reuse_request", "attempt_manifest", "receipt_manifest"):
        validate_contained_path(
            evidence[key]["path"], workspace_root, context="reuse full %s" % key
        )
    _require_exact_output_ref(evidence["command_args"], output_root, "command_args.json", "command_args")
    _require_exact_output_ref(evidence["environment"], output_root, "env.json", "environment")
    source_root = Path(PHASE1_RUN_ROOT)
    validate_contained_path(
        evidence["source_run_manifest"]["path"],
        source_root,
        context="Phase 1 source run manifest",
        exact_relative="control/phase1_run_manifest.json",
    )
    job_id = PHASE1_REUSE_JOB_BY_CELL[sidecar["cell"]["cell_id"]]
    source_output = source_root / "gate-e-full" / job_id
    for key, filename in (
        ("source_command_args", "command_args.json"),
        ("source_environment", "env.json"),
        ("source_revision_report", "model_revision.json"),
        ("source_results", "results.json"),
    ):
        validate_contained_path(
            evidence[key]["path"],
            source_output,
            context=key,
            exact_relative=filename,
        )
    for ref in evidence["sample_source"]["artifacts"]:
        validate_contained_path(
            ref["path"], source_output, context="Phase 1 sample-source artifact"
        )
    loaded = _load_full_evidence_files(evidence)
    authorization_root = _validate_full_control_packets(
        loaded["attempt_manifest"], loaded["receipt_manifest"],
        card=card, cell=sidecar["cell"], output_root=output_root,
        producer_kind="phase1_immutable_reuse_adapter"
    )
    for key in ("reuse_request", "attempt_manifest", "receipt_manifest"):
        _require_governed_ref(evidence[key], authorization_root, key)
    _require_exact_output_ref(evidence["command_args"], output_root, "command_args.json", "command_args")
    _require_exact_output_ref(evidence["environment"], output_root, "env.json", "environment")
    source = load_phase1_immutable_reuse_source(
        card, identity_manifest, sidecar["cell"],
        Path(evidence["source_run_manifest"]["path"]),
        expected_source_evidence={
            key: evidence[key]
            for key in (
                "source_run_manifest", "source_command_args", "source_environment",
                "source_revision_report", "source_results", "sample_source",
            )
        },
    )
    for key, ref in source["evidence"].items():
        if evidence[key] != ref:
            raise Phase2ReuseError("Phase 1 reuse evidence differs from the reloaded source: %s" % key)
    command = loaded["command_args"]
    expected = {
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "cell_id": sidecar["cell"]["cell_id"],
        "phase1_run_manifest": evidence["source_run_manifest"]["path"],
        "output_dir": str(output_root),
        "attempt_manifest": evidence["attempt_manifest"]["path"],
        "receipt_manifest": evidence["receipt_manifest"]["path"],
        "reuse_request": evidence["reuse_request"]["path"],
    }
    if set(command) != set(expected) or any(command[key] != value for key, value in expected.items()):
        raise Phase2ReuseError("Phase 1 reuse command_args differs from actual binding")
    request = loaded["reuse_request"]
    verify_manifest_sha256(request)
    _validate_reuse_request_payload(
        request,
        card=card,
        identity_manifest=identity_manifest,
        cell=sidecar["cell"],
        source_evidence={
            key: evidence[key]
            for key in (
                "source_run_manifest", "source_command_args", "source_environment",
                "source_revision_report", "source_results", "sample_source",
            )
        },
        attempt_manifest=evidence["attempt_manifest"],
        receipt_manifest=evidence["receipt_manifest"],
        output_root=output_root,
    )
    return source["samples"]


def _validate_reuse_request_payload(
    request: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    attempt_manifest: Mapping[str, Any],
    receipt_manifest: Mapping[str, Any],
    output_root: Path,
) -> None:
    request_keys = {
        "schema_version", "artifact_kind", "card_manifest_sha256",
        "identity_manifest_sha256", "cell", "source_evidence",
        "attempt_manifest", "receipt_manifest", "output_root", "manifest_sha256",
    }
    if (
        set(request) != request_keys
        or request.get("schema_version") != "loopscope.phase2-phase1-reuse-request.v1"
        or request.get("artifact_kind") != "phase1_immutable_reuse_request"
        or request.get("card_manifest_sha256") != card["manifest_sha256"]
        or request.get("identity_manifest_sha256") != identity_manifest["manifest_sha256"]
        or request.get("cell") != cell
        or request.get("output_root") != str(output_root)
        or request.get("source_evidence") != source_evidence
        or request.get("attempt_manifest") != attempt_manifest
        or request.get("receipt_manifest") != receipt_manifest
    ):
        raise Phase2ReuseError("Phase 1 reuse request differs from its actual binding")


def _load_full_evidence_files(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    for key, ref in evidence.items():
        if key == "sample_source":
            continue
        path = Path(ref["path"])
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise Phase2ReuseError("full producer file is missing or unreadable: %s" % key) from exc
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            payload = None
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                pass
            if not isinstance(payload, Mapping) or payload.get("manifest_sha256") != ref["sha256"]:
                raise Phase2ReuseError("full producer file hash differs: %s" % key)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise Phase2ReuseError("full producer evidence JSON is malformed: %s" % key) from exc
        if not isinstance(payload, Mapping):
            raise Phase2ReuseError("full producer evidence must contain JSON objects")
        loaded[key] = payload
    return loaded


def _validate_full_control_packets(
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
    cell: Mapping[str, Any],
    output_root: Path,
    producer_kind: str,
) -> Path:
    attempt_keys = {
        "schema_version", "artifact_kind", "producer_kind", "card_manifest_sha256",
        "cell_id", "revision", "executor_thread_id", "created_at_utc",
        "authorization_root", "output_root", "manifest_sha256",
    }
    receipt_keys = attempt_keys | {"attempt_manifest_sha256"}
    if set(attempt) != attempt_keys or set(receipt) != receipt_keys:
        raise Phase2ReuseError("full attempt/receipt exact-key contract differs")
    verify_manifest_sha256(attempt)
    verify_manifest_sha256(receipt)
    if (
        attempt["schema_version"] != "loopscope.phase2-full-attempt.v2"
        or receipt["schema_version"] != "loopscope.phase2-full-receipt.v2"
        or attempt["artifact_kind"] != "full_final_output_attempt"
        or receipt["artifact_kind"] != "full_final_output_execution_receipt"
        or attempt["producer_kind"] != producer_kind
        or receipt["producer_kind"] != producer_kind
    ):
        raise Phase2ReuseError("unsupported full attempt/receipt schema or producer")
    expected = (card["manifest_sha256"], cell["cell_id"], card["science"]["revision"])
    if (
        (attempt["card_manifest_sha256"], attempt["cell_id"], attempt["revision"]) != expected
        or (receipt["card_manifest_sha256"], receipt["cell_id"], receipt["revision"]) != expected
        or receipt["attempt_manifest_sha256"] != attempt["manifest_sha256"]
        or attempt["executor_thread_id"] != receipt["executor_thread_id"]
    ):
        raise Phase2ReuseError("full attempt/receipt differs from card/cell/revision/attempt")
    if attempt["authorization_root"] != receipt["authorization_root"]:
        raise Phase2ReuseError("full attempt/receipt authorization roots differ")
    authorization_root = validate_phase2_workspace_output_path(
        attempt["authorization_root"], card, context="full authorization root"
    )
    governed_output = validate_contained_path(
        output_root, authorization_root, context="full output root", allow_root=True
    )
    if governed_output != output_root.resolve(strict=False):
        raise Phase2ReuseError("full output root differs from authorization")
    if attempt["output_root"] != str(output_root) or receipt["output_root"] != str(output_root):
        raise Phase2ReuseError("full attempt/receipt output root differs")
    for packet in (attempt, receipt):
        if not isinstance(packet["executor_thread_id"], str) or not packet["executor_thread_id"].strip():
            raise Phase2ReuseError("full attempt/receipt executor must be non-empty")
    if _parse_utc(receipt["created_at_utc"]) < _parse_utc(attempt["created_at_utc"]):
        raise Phase2ReuseError("full receipt predates its attempt")
    return authorization_root


def _require_governed_ref(ref: Mapping[str, Any], root: Path, context: str) -> Path:
    path = validate_contained_path(ref["path"], root, context=context)
    if path != Path(ref["path"]).resolve(strict=False):
        raise Phase2ReuseError("%s resolved path differs" % context)
    return path


def _require_exact_output_ref(
    ref: Mapping[str, Any], output_root: Path, filename: str, context: str
) -> Path:
    return validate_contained_path(
        ref["path"], output_root, context=context, exact_relative=filename
    )


def _load_phase1_logged_samples(
    results: Mapping[str, Any], results_path: Path, output_root: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "samples" in results:
        if not isinstance(results["samples"], Mapping):
            raise Phase2ReuseError("Phase 1 results.samples is malformed")
        return dict(results), {
            "kind": "results_inline_samples",
            "artifacts": [_raw_ref(results_path)],
        }
    sidecars = sorted(output_root.glob("samples_*.jsonl"), key=lambda path: path.name)
    if not sidecars:
        raise Phase2ReuseError(
            "Phase 1 results lacks samples and no exact logged-sample sidecars exist"
        )
    discovered = sorted(
        path for path in output_root.iterdir()
        if path.is_file() and path.name.startswith("samples_") and path.suffix == ".jsonl"
    )
    if sidecars != discovered:
        raise Phase2ReuseError("Phase 1 logged-sample sidecar set is ambiguous")
    task_results = results.get("results")
    task_names = [str(name) for name in task_results] if isinstance(task_results, Mapping) else []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    refs: List[Dict[str, str]] = []
    for path in sidecars:
        resolved = validate_contained_path(
            path, output_root, context="Phase 1 logged-sample sidecar"
        )
        refs.append(_raw_ref(resolved))
        raw_namespace = path.stem[len("samples_") :]
        matching_names = [
            name for name in task_names
            if raw_namespace == name or raw_namespace.startswith(name + "_")
        ]
        task = max(matching_names, key=len) if matching_names else raw_namespace
        if not task or task in grouped:
            raise Phase2ReuseError(
                "Phase 1 sample sidecars map to a duplicate/empty task namespace"
            )
        rows: List[Dict[str, Any]] = []
        for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise Phase2ReuseError("Phase 1 sample sidecar contains a blank line")
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise Phase2ReuseError("Phase 1 sample sidecar JSONL is malformed") from exc
            if not isinstance(row, dict):
                raise Phase2ReuseError("Phase 1 sample sidecar row must be an object")
            if "task" in row and row["task"] != task:
                raise Phase2ReuseError(
                    "Phase 1 sample sidecar row %d disagrees with its task namespace"
                    % line_number
                )
            rows.append(row)
        grouped[task] = rows
    return {"samples": grouped}, {
        "kind": "exact_sorted_logged_sample_sidecars",
        "artifacts": refs,
    }


def _validate_phase1_job_argv(
    argv: Any, card: Mapping[str, Any], cell: Mapping[str, Any], output_root: Path
) -> None:
    expected = [
        "python", "-m", "tflt.eval_runner", "--model", "qwen3-1.7b-base",
        "--revision", card["science"]["revision"], "--tasks", "mmlu",
        "--output-dir", str(output_root), "--num-fewshot", "5",
        "--batch-size", "auto", "--dtype", "float16",
    ]
    if cell["protocol"] != "baseline_no_loop":
        expected.extend(
            [
                "--loop", "--window", cell["window"], "--k", "2",
                "--iteration-mode", "block", "--strategy", "damped_euler",
                "--alpha", "1.0", "--beta", "0.0", "--cache-strategy", "last",
                "--decode-mode", "bypass",
            ]
        )
    if argv != expected:
        raise Phase2ReuseError("Phase 1 job argv differs from the frozen cell mapping")


def _validate_phase1_command_args(
    command: Mapping[str, Any], card: Mapping[str, Any], cell: Mapping[str, Any], output_root: Path
) -> None:
    required = {
        "model": "qwen3-1.7b-base", "revision": card["science"]["revision"],
        "tasks": "mmlu", "output_dir": str(output_root), "limit": None,
        "num_fewshot": 5, "batch_size": "auto", "dtype": "float16",
        "loop": cell["protocol"] != "baseline_no_loop",
    }
    for key, expected in required.items():
        if command.get(key) != expected:
            raise Phase2ReuseError("Phase 1 command_args.%s differs" % key)
    if "phase2_final_output_manifest" in command:
        raise Phase2ReuseError("immutable Phase 1 command cannot contain a Phase 2 adapter")
    if cell["protocol"] != "baseline_no_loop":
        loop = {
            "window": cell["window"], "k": 2, "iteration_mode": "block",
            "strategy": "damped_euler", "alpha": 1.0, "beta": 0.0,
            "cache_strategy": "last", "decode_mode": "bypass", "first_n": None,
        }
        for key, expected in loop.items():
            if command.get(key) != expected:
                raise Phase2ReuseError("Phase 1 loop command_args.%s differs" % key)


def _validate_full_command_args(
    command: Mapping[str, Any], card: Mapping[str, Any], cell: Mapping[str, Any],
    output_root: Path, *, manifest_path: str
) -> None:
    required = {
        "model": "qwen3-1.7b-base", "revision": card["science"]["revision"],
        "tasks": "mmlu", "output_dir": str(output_root), "limit": None,
        "num_fewshot": 5, "batch_size": "auto", "dtype": "float16",
        "loop": True, "window": cell["window"], "k": cell["k"],
        "iteration_mode": "block", "strategy": "damped_euler",
        "alpha": cell["alpha"], "beta": 0.0, "cache_strategy": "last",
        "decode_mode": "bypass", "first_n": None,
        "phase2_final_output_manifest": manifest_path,
    }
    for key, expected in required.items():
        if command.get(key) != expected:
            raise Phase2ReuseError("new full command_args.%s differs" % key)


def _validate_revision(revision: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    expected = card["science"]["revision"]
    if (
        revision.get("repo_id") != card["science"]["model"]
        or revision.get("model_commit") != expected
        or revision.get("tokenizer_commit") != expected
        or revision.get("manifest_commit") != expected
        or revision.get("match") is not True
    ):
        raise Phase2ReuseError("model/tokenizer revision closure differs from the card")


def _manifest_ref(path: Path, payload: Mapping[str, Any]) -> Dict[str, str]:
    return {"path": str(path.resolve(strict=False)), "sha256": str(payload["manifest_sha256"])}


def _raw_ref(path: Path) -> Dict[str, str]:
    resolved = Path(path).resolve(strict=False)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _load_json_object(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise Phase2ReuseError("%s is missing or unreadable" % context) from exc
    except ValueError as exc:
        raise Phase2ReuseError("%s is malformed JSON" % context) from exc
    if not isinstance(payload, dict):
        raise Phase2ReuseError("%s must contain a JSON object" % context)
    return payload


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise Phase2ReuseError("full provenance timestamp must be text")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise Phase2ReuseError("full provenance timestamp must be real UTC") from exc


def build_phase1_reuse_matrix() -> Dict[str, Any]:
    """Return schema-confirmed facts without pretending to read live HPC artifacts."""

    entries: List[Dict[str, Any]] = [
        _entry(
            "calibration_pool_identity",
            "requires_gate_b_live_check",
            "Local schema defines the Phase 1 pool identity projection; Gate B must live-read the immutable 512-record manifest and derive the ordered identities.",
        ),
        _entry(
            "inclusive_boundary_and_k2_activity",
            "local_schema_confirmed",
            "Phase 1 probe/window schemas bind inclusive width-4 boundaries, K=2 body-call timeline, r/q, and validity.",
        ),
        _entry(
            "phase1_full_completion_and_revision_closure",
            "historical_only",
            "Control/history records baseline plus K2 completion and revision closure; Gate A performs no live HPC read.",
        ),
        _entry(
            "raw_four_choice_loglikelihood",
            "requires_gate_b_live_check",
            "Layer probe stores distributions, not the four raw lm-eval acc,none log-likelihood scores.",
        ),
        _entry(
            "full_ordered_doc_hash_identity",
            "requires_gate_b_live_check",
            "Local Phase 1 analysis accepts doc_id correctness but does not prove ordered task/doc_id/doc_hash closure.",
        ),
        _entry(
            "phase1_k2_per_step_nca_absence",
            "local_schema_confirmed",
            "The Phase 1 window-probe schema does not persist the Phase 2 repeated-step NCA trajectory.",
        ),
        _entry(
            "phase1_k2_raw_final_choice_availability",
            "requires_gate_b_live_check",
            "Gate B must inspect immutable Phase 1 logged-sample artifacts for four raw choice scores and evaluator identity before any K2 reuse decision.",
        ),
        _entry(
            "phase1_full_cell_producer_partition",
            "local_schema_confirmed",
            "The versioned card binds exactly baseline plus three shared-K2 cells to immutable reuse and exactly eight K3/K4 cells to new lm-eval runs.",
        ),
        _entry(
            "phase1_immutable_reuse_source_digests",
            "requires_gate_b_live_check",
            "The fail-closed loader is implemented and fixture-tested; Gate B must live-read and freeze the actual manifest/results-or-sidecar digests before materialization.",
        ),
        _entry(
            "logged_sample_exact_identity_and_raw_choice_join",
            "requires_gate_b_live_check",
            "The final-output adapter is order-independent and batch-size agnostic, but Gate B must verify that live lm-eval 0.4.11 logged samples expose exact task/doc_id/doc_hash and four raw choice scores.",
        ),
    ]
    return {
        "schema_version": PHASE2_REUSE_SCHEMA_VERSION,
        "verification_scope": "local_read_only_no_ssh",
        "allowed_statuses": sorted(REUSE_STATUSES),
        "entries": entries,
    }


def validate_reuse_matrix(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"schema_version", "verification_scope", "allowed_statuses", "entries"}:
        raise ValueError("reuse matrix has an exact-key mismatch")
    if payload.get("schema_version") != PHASE2_REUSE_SCHEMA_VERSION:
        raise ValueError("unsupported reuse matrix schema")
    if payload.get("verification_scope") != "local_read_only_no_ssh":
        raise ValueError("Gate A reuse matrix cannot claim live remote verification")
    if payload.get("allowed_statuses") != sorted(REUSE_STATUSES):
        raise ValueError("reuse matrix allowed statuses differ")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reuse matrix requires entries")
    names = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"field", "status", "evidence"}:
            raise ValueError("reuse entry has an exact-key mismatch")
        if entry.get("status") not in REUSE_STATUSES:
            raise ValueError("reuse entry has invalid status")
        if not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
            raise ValueError("reuse entry evidence must be non-empty")
        names.append(str(entry.get("field") or ""))
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("reuse entry names must be non-empty and unique")


def build_phase2_freeze_candidate(card: Mapping[str, Any], repo_root: Any) -> Dict[str, Any]:
    """Single canonical prepare producer shared by the CLI and helper script."""

    validate_phase2_card(card)
    verify_phase2_implementation_hashes(card, repo_root)
    reuse = build_phase1_reuse_matrix()
    validate_reuse_matrix(reuse)
    return make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-preparation.v2",
            "status": "freeze_candidate_not_preregistered",
            "card_id": PHASE2_CARD_ID,
            "card_manifest_sha256": card["manifest_sha256"],
            "evidence_scale_contract": dict(card["evidence_scales"]),
            "workspace_root": card["write_once_contract"]["workspace_root"],
            "legacy_phase1_policy": "read_only_in_place_no_move_copy_replace_or_symlink",
            "b1_modes": {
                "smoke": "four_samples_with_k1_and_cross_k_prefix_proof",
                "calibration": "requires_successful_b1_proof_and_512_sealed_identity",
            },
            "no_loop_boundary": {
                "logical_cell_count": 1,
                "native_continuation_lifetime": "same_process_memory_only",
            },
            "loop_cells": list(b2_logical_cells(card)),
            "full_final_output_adapter": {
                "identity_join": "unordered_exact_task_doc_id_doc_hash_to_canonical_manifest",
                "batch_size_auto_allowed": True,
                "residual_collector_connected": False,
                "score_source": card["measurement"]["full_choice_score_source"],
            },
            "full_final_output_producer_partition": {
                kind: list(cell_ids)
                for kind, cell_ids in full_cell_producer_partition(card).items()
            },
            "phase1_immutable_reuse_loader": {
                "source_root": PHASE1_RUN_ROOT,
                "cell_to_job": dict(PHASE1_REUSE_JOB_BY_CELL),
                "live_digest_status": "requires_gate_b_live_check",
                "unknown_digest_policy": "fail_closed_no_synthesis_no_k2_rerun",
            },
            "phase1_reuse_matrix": reuse,
            "gate_a_execution": {
                "remote_reads": 0,
                "gpu_hours": 0,
                "new_run_count": 0,
            },
        }
    )


def _entry(field: str, status: str, evidence: str) -> Dict[str, str]:
    if status not in REUSE_STATUSES:
        raise ValueError("invalid reuse status")
    return {"field": field, "status": status, "evidence": evidence}


__all__ = [
    "PHASE1_REUSE_JOB_BY_CELL",
    "Phase2ReuseError",
    "build_phase1_reuse_matrix",
    "build_phase1_reuse_full_envelope",
    "build_phase2_freeze_candidate",
    "extract_four_raw_choice_scores",
    "extract_sample_acc_none",
    "full_cell_producer_partition",
    "join_lm_eval_logged_samples",
    "load_phase1_immutable_reuse_source",
    "validate_reuse_matrix",
    "verify_full_final_output_artifact",
]
