"""Write-once LoopScope Phase 5 Gate I adjacent angular-distance diagnostic.

The producer is deliberately outcome-blind.  It reuses the exact Phase 3
gold-free validation pool, performs one ordinary Qwen native forward per
identity, captures the raw input to FinalNorm, and persists scalars only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class GateIAngularTrajectoryError(ValueError):
    """Fail-closed Gate I contract violation."""


CARD_SCHEMA = "loopscope.phase5.gate-i-angular-trajectory-card.v1"
RECORD_SCHEMA = "loopscope.phase5.gate-i-angular-distance-record.v1"
MANIFEST_SCHEMA = "loopscope.phase5.gate-i-membership-manifest.v1"
ACQUISITION_SCHEMA = "loopscope.phase5.gate-i-acquisition-receipt.v1"
SEAL_SCHEMA = "loopscope.phase5.gate-i-seal-receipt.v1"
ANALYSIS_SCHEMA = "loopscope.phase5.gate-i-analysis.v1"
DATA_VERIFIER_SCHEMA = "loopscope.phase5.gate-i-data-verifier-receipt.v1"
FIGURE_SCHEMA = "loopscope.phase5.gate-i-figure-receipt.v1"
VERIFIER_SCHEMA = "loopscope.phase5.gate-i-verifier-receipt.v1"
FINAL_SCHEMA = "loopscope.phase5.gate-i-manifest-receipt.v1"

EXECUTOR_THREAD = "019fa464-6734-7ec0-a57a-e07a33223331"
PLANNING_THREAD = "019f8604-4717-7be2-8bf8-9d4a26a3d7f7"
AUTHORIZED_BASE = "97de998504a9bad9867688f307a8bfc6fffdfb81"
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase5-gate-i-angular-trajectory-20260727T161957Z"
)
SOURCE_CHECKOUT = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
POOL_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3b-20260715T223304Z"
)
IMPLEMENTATION_PATHS = (
    "configs/loopscope/phase5_angular_trajectory_gate_i_card.json",
    "src/tflt/loopscope/phase5_angular_trajectory.py",
    "scripts/loopscope/run_qwen_phase5_gate_i.py",
    "tests/test_loopscope_phase5_angular_trajectory.py",
)
RECORD_FIELDS = {
    "schema_version",
    "model_key",
    "model_repo",
    "model_revision",
    "identity",
    "subject",
    "split",
    "prompt_sha256",
    "renderer_provenance",
    "answer_position",
    "sequence_length",
    "loop_insertions",
    "boundary_count",
    "transition_count",
    "adjacent_angular_distance",
    "raw_residual_before_final_norm",
}
IDENTITY_FIELDS = {"task", "doc_id", "doc_hash"}
RENDERER_PROVENANCE_FIELDS = {
    "dataset_repo",
    "dataset_revision",
    "renderer_id",
    "renderer_source_sha256",
    "source_projection_sha256",
    "render_contract_sha256",
    "source_manifest_sha256",
    "pool_manifest_sha256",
}
FORBIDDEN_TOKENS = (
    "target",
    "gold",
    "label",
    "correct",
    "accuracy",
    "gain",
    "flip",
    "outcome",
    "test",
    "hidden",
    "logit",
    "probab",
    "token_id",
    "input_id",
    "loop_residual",
    "prompt_text",
    "rendered_prompt",
    "question",
    "ordered_choices",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    def reject(value: str) -> None:
        raise GateIAngularTrajectoryError("non-finite JSON constant: %s" % value)

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateIAngularTrajectoryError("cannot load strict JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise GateIAngularTrajectoryError("JSON root must be an object: %s" % path)
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    raise GateIAngularTrajectoryError(
                        "blank JSONL line at %s:%d" % (path, number)
                    )
                value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
                    GateIAngularTrajectoryError("non-finite JSON constant: %s" % x)
                ))
                if not isinstance(value, dict):
                    raise GateIAngularTrajectoryError("JSONL row is not an object")
                result.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateIAngularTrajectoryError("cannot load strict JSONL: %s" % path) from exc
    if not result:
        raise GateIAngularTrajectoryError("empty JSONL is forbidden: %s" % path)
    return result


def write_new_json(path: Path, value: Any) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def write_new_text(path: Path, text: str, *, executable: bool = False) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(0o755)
    return file_sha256(path)


def write_new_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        raise GateIAngularTrajectoryError("refusing to write empty JSONL")
    return write_new_text(
        path,
        "".join(canonical_json_bytes(record).decode("utf-8") + "\n" for record in records),
    )


def write_new_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def load_card(path: Path) -> Dict[str, Any]:
    card = load_json(path)
    if card.get("schema_version") != CARD_SCHEMA:
        raise GateIAngularTrajectoryError("Gate I card schema mismatch")
    exact = {
        "executor_thread_id": EXECUTOR_THREAD,
        "planning_thread_id": PLANNING_THREAD,
        "authorized_base_commit": AUTHORIZED_BASE,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "source_checkout": str(SOURCE_CHECKOUT),
    }
    for key, expected in exact.items():
        if card.get(key) != expected:
            raise GateIAngularTrajectoryError("Gate I card %s mismatch" % key)
    if set(card.get("models", {})) != {"qwen3_1p7b", "qwen3_4b"}:
        raise GateIAngularTrajectoryError("Gate I card model membership mismatch")
    return card


def assert_run_root(path: Path, *, must_exist: Optional[bool] = None) -> Path:
    resolved = Path(path).resolve()
    if resolved != AUTHORIZED_RUN_ROOT:
        raise GateIAngularTrajectoryError("run root differs from authorization")
    if must_exist is True and not resolved.is_dir():
        raise GateIAngularTrajectoryError("authorized run root is absent")
    if must_exist is False and resolved.exists():
        raise FileExistsError("authorized write-once run root already exists")
    return resolved


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    return {name: file_sha256(root / name) for name in IMPLEMENTATION_PATHS}


def _assert_no_loop_modules_loaded() -> None:
    forbidden = ("tflt.wrapper", "tflt.strategies", "tflt.cache")
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == item or name.startswith(item + ".") for item in forbidden)
    )
    if loaded:
        raise GateIAngularTrajectoryError("forbidden loop modules loaded: %s" % loaded)


def cosine_to_angle(cosine: float) -> float:
    value = float(cosine)
    if not math.isfinite(value):
        raise GateIAngularTrajectoryError("cosine is non-finite")
    clamped = min(1.0, max(-1.0, value))
    angle = math.acos(clamped) / math.pi
    if not math.isfinite(angle) or not 0.0 <= angle <= 1.0:
        raise GateIAngularTrajectoryError("angular distance is invalid")
    return angle


def angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise GateIAngularTrajectoryError("angular vectors have invalid shape")
    av = [float(value) for value in a]
    bv = [float(value) for value in b]
    if not all(math.isfinite(value) for value in av + bv):
        raise GateIAngularTrajectoryError("angular vectors contain non-finite values")
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    norm = na * nb
    if not all(math.isfinite(value) for value in (dot, na, nb, norm)) or norm == 0.0:
        raise GateIAngularTrajectoryError("angular reduction has zero/non-finite norm")
    return cosine_to_angle(dot / norm)


def torch_adjacent_angular_distances(boundaries: Sequence[Any]) -> List[float]:
    if len(boundaries) < 2:
        raise GateIAngularTrajectoryError("at least two boundaries are required")
    torch = sys.modules.get("torch")
    if torch is None:
        import torch  # type: ignore[no-redef]
    result: List[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        a = left.to(dtype=torch.float32)
        b = right.to(dtype=torch.float32)
        dot = torch.sum(a * b, dtype=torch.float32)
        na = torch.linalg.vector_norm(a, dtype=torch.float32)
        nb = torch.linalg.vector_norm(b, dtype=torch.float32)
        norm = na * nb
        if not bool(torch.isfinite(dot).item()) or not bool(torch.isfinite(norm).item()):
            raise GateIAngularTrajectoryError("non-finite torch dot/norm")
        norm_value = float(norm.to(dtype=torch.float64).cpu())
        if norm_value == 0.0:
            raise GateIAngularTrajectoryError("zero torch norm")
        cosine = float((dot / norm).to(dtype=torch.float64).cpu())
        result.append(cosine_to_angle(cosine))
    return result


def linear_quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= q <= 1.0 or not all(map(math.isfinite, ordered)):
        raise GateIAngularTrajectoryError("invalid quantile input")
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _identity_tuple(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    identity = record["identity"]
    return (str(identity["task"]), str(identity["doc_id"]), str(identity["doc_hash"]))


def validate_sanitized_record(record: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    if set(record) != RECORD_FIELDS:
        raise GateIAngularTrajectoryError(
            "record fields differ: missing=%s extra=%s"
            % (sorted(RECORD_FIELDS - set(record)), sorted(set(record) - RECORD_FIELDS))
        )
    lowered = " ".join(str(key).lower() for key in record)
    matches = [token for token in FORBIDDEN_TOKENS if token in lowered]
    if matches:
        raise GateIAngularTrajectoryError("forbidden persisted field token: %s" % matches)
    if record.get("schema_version") != RECORD_SCHEMA:
        raise GateIAngularTrajectoryError("record schema mismatch")
    if set(record.get("identity", {})) != IDENTITY_FIELDS:
        raise GateIAngularTrajectoryError("identity fields mismatch")
    if set(record.get("renderer_provenance", {})) != RENDERER_PROVENANCE_FIELDS:
        raise GateIAngularTrajectoryError("renderer provenance fields mismatch")
    if record.get("split") != "validation" or record.get("loop_insertions") != 0:
        raise GateIAngularTrajectoryError("split/loop contract mismatch")
    layers = int(model["decoder_layers"])
    if record.get("boundary_count") != layers + 1 or record.get("transition_count") != layers:
        raise GateIAngularTrajectoryError("boundary/transition count mismatch")
    angles = record.get("adjacent_angular_distance")
    if not isinstance(angles, list) or len(angles) != layers:
        raise GateIAngularTrajectoryError("angular-distance length mismatch")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value))
               and 0.0 <= float(value) <= 1.0 for value in angles):
        raise GateIAngularTrajectoryError("invalid angular-distance scalar")
    if record.get("raw_residual_before_final_norm") is not True:
        raise GateIAngularTrajectoryError("raw FinalNorm-input flag mismatch")
    return dict(record)


def _pool_and_manifests(card: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    from tflt.loopscope.phase3_acquisition import load_live_b1_artifacts

    phase3_card = SOURCE_CHECKOUT / "configs/loopscope/phase3_card.json"
    _, records, source_manifest, pool_manifest = load_live_b1_artifacts(POOL_ROOT, phase3_card)
    pool = card["pool"]
    actual_files = {
        "records_file_sha256": file_sha256(POOL_ROOT / pool["records_file"]),
        "manifest_file_sha256": file_sha256(POOL_ROOT / pool["manifest_file"]),
        "source_manifest_file_sha256": file_sha256(POOL_ROOT / pool["source_manifest_file"]),
    }
    for key, actual in actual_files.items():
        if actual != pool[key]:
            raise GateIAngularTrajectoryError("pool %s mismatch" % key)
    if pool_manifest.get("manifest_sha256") != pool["manifest_internal_sha256"]:
        raise GateIAngularTrajectoryError("pool internal manifest mismatch")
    if source_manifest.get("manifest_sha256") != pool["source_manifest_internal_sha256"]:
        raise GateIAngularTrajectoryError("source internal manifest mismatch")
    if len(records) != pool["record_count"] or len({r["subject"] for r in records}) != pool["subject_count"]:
        raise GateIAngularTrajectoryError("pool count/subject closure mismatch")
    if pool_manifest.get("ordered_identity_sha256") != pool["ordered_identity_sha256"]:
        raise GateIAngularTrajectoryError("ordered pool membership mismatch")
    if pool_manifest.get("ordered_prompt_sha256") != pool["ordered_prompt_sha256"]:
        raise GateIAngularTrajectoryError("ordered prompt hash mismatch")
    return records, source_manifest, pool_manifest


def _verify_snapshot(model: Mapping[str, Any]) -> Dict[str, str]:
    snapshot = Path(model["snapshot"])
    if not snapshot.is_dir():
        raise GateIAngularTrajectoryError("model snapshot is absent")
    expected = {
        "config.json": model["config_sha256"],
        "tokenizer_config.json": model["tokenizer_config_sha256"],
        "tokenizer.json": model["tokenizer_json_sha256"],
        **model["weight_files"],
    }
    actual = {name: file_sha256(snapshot / name) for name in expected}
    if actual != expected:
        raise GateIAngularTrajectoryError("model snapshot hash closure mismatch")
    return actual


def initialize_run_root(card_path: Path, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    if expected_commit != AUTHORIZED_BASE:
        raise GateIAngularTrajectoryError("expected implementation base mismatch")
    _assert_no_loop_modules_loaded()
    stage_root = repository_root().resolve()
    if stage_root.parent != root or stage_root.name != "stage":
        raise GateIAngularTrajectoryError("implementation is not staged at run_root/stage")
    if sorted(path.name for path in root.iterdir()) != ["stage"]:
        raise GateIAngularTrajectoryError("new run root contains content other than exact stage")
    records, source_manifest, pool_manifest = _pool_and_manifests(card)
    snapshots = {key: _verify_snapshot(value) for key, value in card["models"].items()}
    payload = {
        "schema_version": "loopscope.phase5.gate-i-admission-receipt.v1",
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD,
        "planning_thread_id": PLANNING_THREAD,
        "authorized_base_commit": AUTHORIZED_BASE,
        "run_root": str(root),
        "card_sha256": file_sha256(card_path),
        "implementation_sha256": implementation_hashes(),
        "pool": {
            "record_count": len(records),
            "subject_count": len({r["subject"] for r in records}),
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        },
        "snapshot_hashes": snapshots,
        "information_barrier": {
            "test_read": False,
            "gold_read": False,
            "outcome_read": False,
            "accuracy_computed": False,
            "loop_insertions": 0,
        },
        "status": "PASS",
    }
    write_new_json(root / "admission_receipt.json", payload)
    return payload


def _membership_record(record: Mapping[str, Any], ordinal: int) -> Dict[str, Any]:
    return {
        "ordinal": int(ordinal),
        "identity": dict(record["identity"]),
        "subject": str(record["subject"]),
        "split": str(record["split"]),
        "prompt_sha256": str(record["prompt_sha256"]),
    }


def _smoke_indices(records: Sequence[Mapping[str, Any]]) -> List[int]:
    first_subject = str(records[0]["subject"])
    first = [i for i, record in enumerate(records) if record["subject"] == first_subject][:2]
    second_subject = next(str(r["subject"]) for r in records if r["subject"] != first_subject)
    second = [i for i, record in enumerate(records) if record["subject"] == second_subject][:2]
    indices = first + second
    if len(indices) != 4:
        raise GateIAngularTrajectoryError("cannot freeze 4-identity 2-subject smoke")
    return indices


def freeze_membership(
    card_path: Path, run_root: Path, model_key: str, mode: str, shard_count: int = 4
) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    if model_key not in card["models"] or mode not in {"smoke", "formal"}:
        raise GateIAngularTrajectoryError("invalid model/mode")
    records, source_manifest, pool_manifest = _pool_and_manifests(card)
    model = card["models"][model_key]
    base = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "mode": mode,
        "model_key": model_key,
        "model_repo": model["repo"],
        "model_revision": model["revision"],
        "decoder_layers": model["decoder_layers"],
        "card_sha256": file_sha256(card_path),
        "implementation_sha256": implementation_hashes(),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
    }
    if mode == "smoke":
        members = [_membership_record(records[i], i) for i in _smoke_indices(records)]
        base.update(
            record_count=4,
            subject_count=len({r["subject"] for r in members}),
            records=members,
            membership_sha256=sha256_bytes(canonical_json_bytes(members)),
        )
        destination = root / model_key / "smoke_membership_manifest.json"
    else:
        smoke = load_json(root / model_key / "smoke_seal_receipt.json")
        if smoke.get("status") != "PASS":
            raise GateIAngularTrajectoryError("formal freeze requires sealed smoke PASS")
        if shard_count != int(card["acquisition"]["formal_shard_count"]):
            raise GateIAngularTrajectoryError("formal shard count differs from card")
        shards = []
        for shard_id in range(shard_count):
            members = [
                _membership_record(record, ordinal)
                for ordinal, record in enumerate(records)
                if ordinal % shard_count == shard_id
            ]
            shards.append(
                {
                    "shard_id": shard_id,
                    "record_count": len(members),
                    "records": members,
                    "membership_sha256": sha256_bytes(canonical_json_bytes(members)),
                }
            )
        base.update(
            record_count=len(records),
            subject_count=len({r["subject"] for r in records}),
            shard_count=shard_count,
            shards=shards,
            membership_sha256=sha256_bytes(
                canonical_json_bytes([_membership_record(r, i) for i, r in enumerate(records)])
            ),
            smoke_seal_sha256=file_sha256(root / model_key / "smoke_seal_receipt.json"),
        )
        destination = root / model_key / "formal_membership_manifest.json"
    base["manifest_sha256"] = sha256_bytes(canonical_json_bytes(base))
    write_new_json(destination, base)
    return base


def _select_members(
    manifest: Mapping[str, Any], mode: str, shard_id: Optional[int]
) -> Tuple[List[Dict[str, Any]], str]:
    if mode == "smoke":
        return [dict(v) for v in manifest["records"]], str(manifest["membership_sha256"])
    if shard_id is None or shard_id < 0 or shard_id >= int(manifest["shard_count"]):
        raise GateIAngularTrajectoryError("formal shard id out of range")
    shard = manifest["shards"][shard_id]
    return [dict(v) for v in shard["records"]], str(shard["membership_sha256"])


def _renderer_provenance(
    pool_record: Mapping[str, Any], source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    source = pool_record["renderer_provenance"]
    return {
        "dataset_repo": source["dataset_repo"],
        "dataset_revision": source["dataset_revision"],
        "renderer_id": source["renderer_id"],
        "renderer_source_sha256": source["renderer_source_sha256"],
        "source_projection_sha256": source["source_projection_sha256"],
        "render_contract_sha256": source["render_contract_sha256"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
    }


def _capture_raw_final_boundary(
    final_norm: Any,
    forward: Callable[[], Any],
    *,
    torch_module: Any,
) -> Tuple[Any, Any, int, float]:
    captured: List[Any] = []

    def prehook(_module: Any, args: Tuple[Any, ...]) -> None:
        if not args:
            raise GateIAngularTrajectoryError("FinalNorm prehook lacks positional input")
        captured.append(args[0])

    handle = final_norm.register_forward_pre_hook(prehook)
    try:
        outputs = forward()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise GateIAngularTrajectoryError(
            "FinalNorm prehook call count is %d, expected 1" % len(captured)
        )
    raw_final = captured[0]
    with torch_module.inference_mode():
        recomputed = final_norm(raw_final)
    post = tuple(outputs.hidden_states or ())[-1]
    if recomputed.shape != post.shape:
        raise GateIAngularTrajectoryError("FinalNorm closure shape mismatch")
    difference = (recomputed - post).detach().abs().max()
    max_abs = float(difference.to(dtype=torch_module.float64).cpu())
    if not bool(torch_module.equal(recomputed, post)):
        raise GateIAngularTrajectoryError(
            "FinalNorm(raw B_L) does not exactly close hidden_states[-1], max=%r" % max_abs
        )
    return outputs, raw_final, len(captured), max_abs


def _load_runtime(model: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    _assert_no_loop_modules_loaded()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise GateIAngularTrajectoryError("remote ML runtime imports failed") from exc
    if not torch.cuda.is_available():
        raise GateIAngularTrajectoryError("Gate I forward requires a visible CUDA GPU")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[model["dtype"]]
    snapshot = str(model["snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    causal = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if int(causal.config.num_hidden_layers) != int(model["decoder_layers"]):
        raise GateIAngularTrajectoryError("loaded decoder layer count mismatch")
    causal.eval().to("cuda")
    native = causal.model
    final_norm = native.norm
    for name, module in causal.named_modules():
        module_name = str(getattr(module.__class__, "__module__", ""))
        if module_name == "tflt.wrapper" or module_name.startswith("tflt.wrapper."):
            raise GateIAngularTrajectoryError("active loop wrapper: %s" % name)
    _assert_no_loop_modules_loaded()
    return torch, tokenizer, causal, final_norm


def _acquire_one(
    pool_record: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    model_key: str,
    model: Mapping[str, Any],
    runtime: Tuple[Any, Any, Any, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch, tokenizer, causal, final_norm = runtime
    prompt = str(pool_record["rendered_prompt"])
    if not prompt.rstrip().endswith("Answer:"):
        raise GateIAngularTrajectoryError("prompt does not end at Answer:")
    encoded = tokenizer(prompt, return_tensors="pt")
    if set(("input_ids", "attention_mask")) - set(encoded):
        raise GateIAngularTrajectoryError("tokenizer lacks ids/mask")
    valid = encoded["attention_mask"][0].nonzero(as_tuple=False).flatten().tolist()
    if not valid or int(valid[-1]) != int(encoded["attention_mask"].shape[1]) - 1:
        raise GateIAngularTrajectoryError("answer position is padded/ambiguous")
    answer_position = int(valid[-1])
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        outputs, raw_final, hook_calls, closure = _capture_raw_final_boundary(
            final_norm,
            lambda: causal.model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            ),
            torch_module=torch,
        )
        hidden_states = tuple(outputs.hidden_states or ())
        layers = int(model["decoder_layers"])
        if len(hidden_states) != layers + 1:
            raise GateIAngularTrajectoryError("native hidden-state count mismatch")
        raw_boundaries = list(hidden_states[:-1]) + [raw_final]
        answer_vectors = [
            hidden[0, answer_position, :].detach() for hidden in raw_boundaries
        ]
        angles = torch_adjacent_angular_distances(answer_vectors)
    record = {
        "schema_version": RECORD_SCHEMA,
        "model_key": model_key,
        "model_repo": model["repo"],
        "model_revision": model["revision"],
        "identity": dict(pool_record["identity"]),
        "subject": str(pool_record["subject"]),
        "split": "validation",
        "prompt_sha256": str(pool_record["prompt_sha256"]),
        "renderer_provenance": _renderer_provenance(
            pool_record, source_manifest, pool_manifest
        ),
        "answer_position": answer_position,
        "sequence_length": int(encoded["attention_mask"].shape[1]),
        "loop_insertions": 0,
        "boundary_count": layers + 1,
        "transition_count": layers,
        "adjacent_angular_distance": angles,
        "raw_residual_before_final_norm": True,
    }
    validate_sanitized_record(record, model)
    evidence = {
        "identity": dict(pool_record["identity"]),
        "answer_position": answer_position,
        "answer_position_is_last_non_padding": True,
        "sequence_length": int(encoded["attention_mask"].shape[1]),
        "final_norm_prehook_calls": hook_calls,
        "final_norm_closure_max_abs": closure,
        "boundary_count": layers + 1,
        "transition_count": layers,
    }
    del outputs, raw_final, hidden_states, raw_boundaries, answer_vectors, inputs, encoded
    return record, evidence


def acquire(
    card_path: Path,
    run_root: Path,
    model_key: str,
    mode: str,
    shard_id: Optional[int] = None,
    smoke_attempt: int = 1,
) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    model = card["models"].get(model_key)
    if model is None or mode not in {"smoke", "formal"}:
        raise GateIAngularTrajectoryError("invalid acquisition model/mode")
    if mode == "formal" and smoke_attempt != 1:
        raise GateIAngularTrajectoryError("formal acquisition has no attempt repair")
    if mode == "smoke" and smoke_attempt not in (1, 2):
        raise GateIAngularTrajectoryError("smoke permits attempt 1 or one fresh retry")
    manifest_path = root / model_key / (
        "smoke_membership_manifest.json" if mode == "smoke"
        else "formal_membership_manifest.json"
    )
    manifest = load_json(manifest_path)
    members, membership_hash = _select_members(manifest, mode, shard_id)
    records, source_manifest, pool_manifest = _pool_and_manifests(card)
    by_ordinal = {i: record for i, record in enumerate(records)}
    if mode == "smoke":
        output = root / model_key / "smoke" / ("attempt-%04d" % smoke_attempt)
    else:
        output = root / model_key / "formal" / ("shard-%04d" % int(shard_id))
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "acquisition_status.json"
    forward_calls = 0
    sanitized: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    try:
        runtime = _load_runtime(model)
        torch = runtime[0]
        for member in members:
            ordinal = int(member["ordinal"])
            pool_record = by_ordinal[ordinal]
            if _identity_tuple(pool_record) != _identity_tuple(member):
                raise GateIAngularTrajectoryError("membership identity mismatch")
            record, one_evidence = _acquire_one(
                pool_record, source_manifest, pool_manifest, model_key, model, runtime
            )
            forward_calls += 1
            sanitized.append(record)
            evidence.append(one_evidence)
        records_path = output / "sanitized_angular_distance_records.jsonl"
        records_sha = write_new_jsonl(records_path, sanitized)
        receipt = {
            "schema_version": ACQUISITION_SCHEMA,
            "created_at_utc": utc_now(),
            "status": "PASS",
            "mode": mode,
            "model_key": model_key,
            "shard_id": shard_id,
            "smoke_attempt": smoke_attempt if mode == "smoke" else None,
            "record_count": len(sanitized),
            "forward_calls": forward_calls,
            "unique_identity_count": len({_identity_tuple(r) for r in sanitized}),
            "subject_count": len({r["subject"] for r in sanitized}),
            "membership_sha256": membership_hash,
            "records_sha256": records_sha,
            "implementation_sha256": implementation_hashes(),
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "transformers": __import__("transformers").__version__,
                "dtype": model["dtype"],
                "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "model_class": runtime[2].__class__.__name__,
                "native_model_class": runtime[2].model.__class__.__name__,
                "final_norm_class": runtime[3].__class__.__name__,
                "model_snapshot": model["snapshot"],
            },
            "per_identity_evidence": evidence,
            "closure": {
                "final_norm_prehook_calls_per_forward": sorted(
                    {item["final_norm_prehook_calls"] for item in evidence}
                ),
                "final_norm_closure_max_abs": max(
                    item["final_norm_closure_max_abs"] for item in evidence
                ),
                "boundary_counts": sorted({r["boundary_count"] for r in sanitized}),
                "transition_counts": sorted({r["transition_count"] for r in sanitized}),
                "all_angles_finite_in_unit_interval": True,
                "unknown_fields_rejected": True,
                "forbidden_fields_absent": True,
                "loop_insertions": 0,
            },
            "information_barrier": {
                "test_read": False,
                "gold_read": False,
                "outcome_read": False,
                "accuracy_computed": False,
            },
        }
        write_new_json(output / "acquisition_receipt.json", receipt)
        write_new_json(status_path, {"status": "PASS", "forward_calls": forward_calls})
        return receipt
    except Exception as exc:
        if not status_path.exists():
            write_new_json(
                status_path,
                {
                    "status": "FAIL",
                    "forward_calls": forward_calls,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "created_at_utc": utc_now(),
                },
            )
        raise


def _validate_receipt_records(
    receipt_path: Path, records_path: Path, model: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    receipt = load_json(receipt_path)
    records = load_jsonl(records_path)
    if receipt.get("status") != "PASS" or receipt.get("record_count") != len(records):
        raise GateIAngularTrajectoryError("acquisition receipt count/status mismatch")
    if receipt.get("forward_calls") != len(records):
        raise GateIAngularTrajectoryError("forward count mismatch")
    if receipt.get("records_sha256") != file_sha256(records_path):
        raise GateIAngularTrajectoryError("records hash mismatch")
    return receipt, [validate_sanitized_record(record, model) for record in records]


def seal_smoke(card_path: Path, run_root: Path, model_key: str, attempt: int = 1) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    model = card["models"][model_key]
    base = root / model_key / "smoke" / ("attempt-%04d" % attempt)
    receipt, records = _validate_receipt_records(
        base / "acquisition_receipt.json",
        base / "sanitized_angular_distance_records.jsonl",
        model,
    )
    manifest = load_json(root / model_key / "smoke_membership_manifest.json")
    identities = [_identity_tuple(r) for r in records]
    expected = [_identity_tuple(r) for r in manifest["records"]]
    if identities != expected or len(records) != 4 or len(set(identities)) != 4:
        raise GateIAngularTrajectoryError("smoke identity closure mismatch")
    if len({r["subject"] for r in records}) < 2:
        raise GateIAngularTrajectoryError("smoke spans fewer than two subjects")
    payload = {
        "schema_version": SEAL_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "mode": "smoke",
        "model_key": model_key,
        "attempt": attempt,
        "record_count": 4,
        "forward_calls": receipt["forward_calls"],
        "unique_identity_count": 4,
        "subject_count": len({r["subject"] for r in records}),
        "boundary_count": model["decoder_layers"] + 1,
        "transition_count": model["decoder_layers"],
        "answer_position": "last_non_padding_token",
        "final_norm_prehook_calls_per_forward": 1,
        "final_norm_closure_max_abs": receipt["closure"]["final_norm_closure_max_abs"],
        "loop_insertions": 0,
        "all_angles_finite_in_unit_interval": True,
        "forbidden_fields_absent": True,
        "records_sha256": file_sha256(base / "sanitized_angular_distance_records.jsonl"),
        "acquisition_receipt_sha256": file_sha256(base / "acquisition_receipt.json"),
    }
    write_new_json(root / model_key / "smoke_seal_receipt.json", payload)
    return payload


def seal_formal(card_path: Path, run_root: Path, model_key: str) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    model = card["models"][model_key]
    manifest = load_json(root / model_key / "formal_membership_manifest.json")
    all_records: List[Dict[str, Any]] = []
    receipts = []
    for shard_id in range(int(manifest["shard_count"])):
        base = root / model_key / "formal" / ("shard-%04d" % shard_id)
        receipt, records = _validate_receipt_records(
            base / "acquisition_receipt.json",
            base / "sanitized_angular_distance_records.jsonl",
            model,
        )
        receipts.append(receipt)
        all_records.extend(records)
    all_records.sort(key=lambda record: next(
        int(m["ordinal"]) for shard in manifest["shards"] for m in shard["records"]
        if _identity_tuple(m) == _identity_tuple(record)
    ))
    pool_records, _, _ = _pool_and_manifests(card)
    actual = [_identity_tuple(r) for r in all_records]
    expected = [_identity_tuple(r) for r in pool_records]
    if actual != expected or len(actual) != 1531 or len(set(actual)) != 1531:
        raise GateIAngularTrajectoryError("formal canonical membership/order mismatch")
    if len({r["subject"] for r in all_records}) != 57:
        raise GateIAngularTrajectoryError("formal subject closure mismatch")
    output = root / model_key / "sanitized_angular_distance_records.jsonl"
    records_sha = write_new_jsonl(output, all_records)
    payload = {
        "schema_version": SEAL_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "mode": "formal",
        "model_key": model_key,
        "record_count": len(all_records),
        "forward_calls": sum(int(r["forward_calls"]) for r in receipts),
        "unique_identity_count": len(set(actual)),
        "subject_count": len({r["subject"] for r in all_records}),
        "missing": 0,
        "duplicate": 0,
        "extra": 0,
        "partial": 0,
        "boundary_count": model["decoder_layers"] + 1,
        "transition_count": model["decoder_layers"],
        "loop_insertions": 0,
        "records_sha256": records_sha,
        "shard_receipt_sha256": [
            file_sha256(
                root / model_key / "formal" / ("shard-%04d" % shard)
                / "acquisition_receipt.json"
            )
            for shard in range(int(manifest["shard_count"]))
        ],
        "information_barrier": {
            "test_read": False,
            "gold_read": False,
            "outcome_read": False,
            "accuracy_computed": False,
        },
    }
    write_new_json(root / model_key / "formal_seal_receipt.json", payload)
    return payload


def subject_stratified_bootstrap(
    records: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> List[List[float]]:
    if not records:
        raise GateIAngularTrajectoryError("bootstrap needs records")
    layers = len(records[0]["adjacent_angular_distance"])
    by_subject: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if len(record["adjacent_angular_distance"]) != layers:
            raise GateIAngularTrajectoryError("bootstrap transition shape mismatch")
        by_subject[str(record["subject"])].append(record)
    rng = random.Random(int(seed))
    curves: List[List[float]] = []
    for _ in range(int(replicates)):
        sums = [0.0] * layers
        count = 0
        for subject in sorted(by_subject):
            group = by_subject[subject]
            for _draw in range(len(group)):
                record = group[rng.randrange(len(group))]
                count += 1
                for layer, value in enumerate(record["adjacent_angular_distance"]):
                    sums[layer] += float(value)
        curves.append([value / count for value in sums])
    return curves


def aggregate_records(
    records: Sequence[Mapping[str, Any]], bootstrap_curves: Sequence[Sequence[float]]
) -> List[Dict[str, Any]]:
    layers = len(records[0]["adjacent_angular_distance"])
    if any(len(curve) != layers for curve in bootstrap_curves):
        raise GateIAngularTrajectoryError("aggregate bootstrap shape mismatch")
    result = []
    for layer in range(layers):
        values = [float(record["adjacent_angular_distance"][layer]) for record in records]
        boot = [float(curve[layer]) for curve in bootstrap_curves]
        result.append(
            {
                "transition_index": layer,
                "normalized_depth": layer / layers,
                "n": len(values),
                "subject_count": len({record["subject"] for record in records}),
                "mean": statistics.fmean(values),
                "sample_sd_ddof_1": statistics.stdev(values),
                "median": linear_quantile(values, 0.5),
                "q25": linear_quantile(values, 0.25),
                "q75": linear_quantile(values, 0.75),
                "bootstrap_ci95_low": linear_quantile(boot, 0.025),
                "bootstrap_ci95_high": linear_quantile(boot, 0.975),
            }
        )
    return result


def _quadratic_shape(x: Sequence[float], y: Sequence[float]) -> Tuple[bool, List[float]]:
    if len(x) != len(y) or len(x) < 3:
        return False, []
    try:
        import numpy as np
    except Exception as exc:
        raise GateIAngularTrajectoryError("NumPy is required for Kneedle shape closure") from exc
    coefficients = [float(v) for v in np.polyfit(x, y, 2)]
    a, b, _ = coefficients
    derivatives = [2.0 * a * float(value) + b for value in x]
    valid = 2.0 * a > 0.0 and all(value < 0.0 for value in derivatives)
    return valid, coefficients


def kneedle_direction(
    mean_curve: Sequence[float],
    bootstrap_curves: Sequence[Sequence[float]],
    *,
    reverse: bool,
    locator_factory: Optional[Callable[..., Any]] = None,
    minimum_frequency: float = 0.8,
) -> Dict[str, Any]:
    values = list(reversed(mean_curve)) if reverse else list(mean_curve)
    layers = len(values)
    if layers < 3:
        return {"status": "NO_STABLE_KNEE_SHAPE_ASSUMPTION", "reason": "shape_invalid"}
    x = [index / layers for index in range(layers)]
    shape_ok, coefficients = _quadratic_shape(x, values)
    decreasing_fraction = sum(
        1 for left, right in zip(values[:-1], values[1:]) if right < left
    ) / (layers - 1)
    base = {
        "direction": "reverse" if reverse else "forward",
        "quadratic_coefficients": coefficients,
        "raw_decreasing_step_fraction": decreasing_fraction,
    }
    if not shape_ok:
        return {**base, "status": "NO_STABLE_KNEE_SHAPE_ASSUMPTION", "reason": "shape_invalid"}
    if locator_factory is None:
        from kneed import KneeLocator
        locator_factory = KneeLocator
    locator = locator_factory(
        x, values, curve="convex", direction="decreasing",
        interp_method="polynomial", polynomial_degree=2, online=True, S=1.0
    )
    knees = sorted(float(value) for value in (getattr(locator, "all_knees", set()) or set()))
    grid = [index for index, value in enumerate(x) if any(abs(value - knee) <= 1e-12 for knee in knees)]
    if len(knees) != 1 or len(grid) != 1:
        return {
            **base,
            "status": "NO_STABLE_KNEE_NON_UNIQUE",
            "reason": "non_unique_or_off_grid",
            "all_knees": knees,
        }
    point_index = grid[0]
    mapped_transition = layers - 1 - point_index if reverse else point_index
    histogram: Counter[int] = Counter()
    for curve in bootstrap_curves:
        replica = list(reversed(curve)) if reverse else list(curve)
        replica_shape, _ = _quadratic_shape(x, replica)
        if not replica_shape:
            continue
        candidate = locator_factory(
            x, replica, curve="convex", direction="decreasing",
            interp_method="polynomial", polynomial_degree=2, online=True, S=1.0
        )
        candidate_knees = sorted(
            float(value) for value in (getattr(candidate, "all_knees", set()) or set())
        )
        candidate_grid = [
            index for index, value in enumerate(x)
            if any(abs(value - knee) <= 1e-12 for knee in candidate_knees)
        ]
        if len(candidate_knees) == 1 and len(candidate_grid) == 1:
            mapped = layers - 1 - candidate_grid[0] if reverse else candidate_grid[0]
            histogram[mapped] += 1
    frequency = histogram[mapped_transition] / len(bootstrap_curves)
    result = {
        **base,
        "all_knees": knees,
        "point_grid_index": point_index,
        "mapped_transition": mapped_transition,
        "candidate_boundary": (
            mapped_transition if reverse else mapped_transition + 1
        ),
        "bootstrap_winner_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "bootstrap_frequency": frequency,
        "bootstrap_denominator": len(bootstrap_curves),
    }
    if frequency < minimum_frequency:
        return {
            **result,
            "status": "NO_STABLE_KNEE_UNSTABLE",
            "reason": "bootstrap_frequency_below_threshold",
        }
    return {**result, "status": "STABLE_KNEE", "reason": "stable"}


def analyze_kneedle(
    aggregate: Sequence[Mapping[str, Any]],
    bootstrap_curves: Sequence[Sequence[float]],
    locator_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    mean = [float(row["mean"]) for row in aggregate]
    forward = kneedle_direction(
        mean, bootstrap_curves, reverse=False, locator_factory=locator_factory
    )
    reverse = kneedle_direction(
        mean, bootstrap_curves, reverse=True, locator_factory=locator_factory
    )
    valid_interval = (
        forward["status"] == "STABLE_KNEE"
        and reverse["status"] == "STABLE_KNEE"
        and int(forward["candidate_boundary"]) < int(reverse["candidate_boundary"])
    )
    return {
        "forward": forward,
        "reverse": reverse,
        "model_status": "STABLE_RECURSIVE_BLOCK" if valid_interval else "NO_STABLE_KNEE",
        "descriptive_recursive_block": (
            [forward["candidate_boundary"], reverse["candidate_boundary"]]
            if valid_interval else None
        ),
    }


def figure_source_rows(
    aggregates: Mapping[str, Sequence[Mapping[str, Any]]],
    models: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if set(aggregates) != set(models):
        raise GateIAngularTrajectoryError("figure source model membership mismatch")
    rows = []
    for model_key in ("qwen3_1p7b", "qwen3_4b"):
        expected = int(models[model_key]["decoder_layers"])
        values = list(aggregates[model_key])
        if len(values) != expected:
            raise GateIAngularTrajectoryError("figure source transition membership mismatch")
        for row in values:
            rows.append(
                {
                    "model_key": model_key,
                    "model_repo": models[model_key]["repo"],
                    "transition_index": row["transition_index"],
                    "normalized_depth": row["normalized_depth"],
                    "mean": row["mean"],
                    "bootstrap_ci95_low": row["bootstrap_ci95_low"],
                    "bootstrap_ci95_high": row["bootstrap_ci95_high"],
                }
            )
    return rows


AGGREGATE_FIELDS = (
    "transition_index", "normalized_depth", "n", "subject_count", "mean",
    "sample_sd_ddof_1", "median", "q25", "q75",
    "bootstrap_ci95_low", "bootstrap_ci95_high",
)
SOURCE_FIELDS = (
    "model_key", "model_repo", "transition_index", "normalized_depth", "mean",
    "bootstrap_ci95_low", "bootstrap_ci95_high",
)


def _plot_with_gnuplot(
    root: Path,
    card: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    kneedle: Mapping[str, Any],
) -> Dict[str, str]:
    plot_dir = root / "plot_inputs"
    plot_dir.mkdir(parents=True, exist_ok=False)
    data_paths = {}
    for key in ("qwen3_1p7b", "qwen3_4b"):
        path = plot_dir / (key + ".dat")
        rows = [row for row in source_rows if row["model_key"] == key]
        text = "".join(
            "%d %.17g %.17g %.17g %.17g\n"
            % (
                row["transition_index"], row["normalized_depth"], row["mean"],
                row["bootstrap_ci95_low"], row["bootstrap_ci95_high"],
            )
            for row in rows
        )
        write_new_text(path, text)
        data_paths[key] = path
    base = card["figure"]["base"]
    script = plot_dir / "plot.gp"
    svg = root / (base + ".svg")
    pdf = root / (base + ".pdf")
    png = root / (base + ".png")
    annotations = []
    raw_arrows = []
    normalized_arrows = []
    arrow_id = 1
    for index, key in enumerate(("qwen3_1p7b", "qwen3_4b"), start=1):
        status = kneedle[key]["model_status"]
        if status == "NO_STABLE_KNEE":
            annotations.append(
                "set label %d '%s: NO_STABLE_KNEE' at screen 0.55,screen %.3f tc rgb '%s' front"
                % (index, key.replace("_", " "), 0.93 - index * 0.045, card["models"][key]["color"])
            )
        for direction in ("forward", "reverse"):
            result = kneedle[key][direction]
            if result["status"] == "STABLE_KNEE":
                transition = int(result["mapped_transition"])
                color = card["models"][key]["color"]
                raw_arrows.append(
                    "set arrow %d from %d,graph 0 to %d,graph 1 nohead dt 3 lw 1 lc rgb '%s'"
                    % (arrow_id, transition, transition, color)
                )
                layers = int(card["models"][key]["decoder_layers"])
                normalized_arrows.append(
                    "set arrow %d from %.17g,graph 0 to %.17g,graph 1 nohead dt 3 lw 1 lc rgb '%s'"
                    % (arrow_id, transition / layers, transition / layers, color)
                )
                arrow_id += 1
    coefficients = {
        key: kneedle[key]["forward"]["quadratic_coefficients"]
        for key in ("qwen3_1p7b", "qwen3_4b")
    }
    for key, value in coefficients.items():
        if len(value) != 3:
            raise GateIAngularTrajectoryError("figure requires degree-2 fit coefficients")
    a17, b17, c17 = coefficients["qwen3_1p7b"]
    a4, b4, c4 = coefficients["qwen3_4b"]
    plot_body = """
set multiplot layout 1,2 margins 0.09,0.98,0.16,0.92 spacing 0.10,0.06
set border linewidth 1.0
set tics out nomirror
set grid ytics lc rgb '#DDDDDD' lw 0.6
set key top right font ',8'
set ylabel 'Adjacent angular distance / pi'
set xlabel 'Transition index l'
set title 'A  Raw transition index' left
set autoscale x
unset arrow
f17i(x)=%.17g*(x/28.0)**2+%.17g*(x/28.0)+%.17g
f4i(x)=%.17g*(x/36.0)**2+%.17g*(x/36.0)+%.17g
%s
plot '%s' using 1:4:5 with filledcurves lc rgb '#F0D9B7' notitle, \
     '%s' using 1:3 with linespoints lw 1.5 pt 6 ps 0.55 lc rgb '#C98022' title 'Qwen3-1.7B', \
     '%s' using 1:4:5 with filledcurves lc rgb '#C9DDF0' notitle, \
     '%s' using 1:3 with linespoints lw 1.5 pt 6 ps 0.55 lc rgb '#3775BA' title 'Qwen3-4B', \
     f17i(x) with lines dt 2 lw 1 lc rgb '#C98022' notitle, \
     f4i(x) with lines dt 2 lw 1 lc rgb '#3775BA' notitle
unset ylabel
set xlabel 'Normalized depth l/L'
set xrange [0:1]
set title 'B  Normalized depth' left
unset arrow
f17n(x)=%.17g*x**2+%.17g*x+%.17g
f4n(x)=%.17g*x**2+%.17g*x+%.17g
%s
plot '%s' using 2:4:5 with filledcurves lc rgb '#F0D9B7' notitle, \
     '%s' using 2:3 with linespoints lw 1.5 pt 6 ps 0.55 lc rgb '#C98022' title 'Qwen3-1.7B', \
     '%s' using 2:4:5 with filledcurves lc rgb '#C9DDF0' notitle, \
     '%s' using 2:3 with linespoints lw 1.5 pt 6 ps 0.55 lc rgb '#3775BA' title 'Qwen3-4B', \
     f17n(x) with lines dt 2 lw 1 lc rgb '#C98022' notitle, \
     f4n(x) with lines dt 2 lw 1 lc rgb '#3775BA' notitle
unset multiplot
""" % (
        a17, b17, c17, a4, b4, c4, "\n".join(raw_arrows),
        data_paths["qwen3_1p7b"], data_paths["qwen3_1p7b"],
        data_paths["qwen3_4b"], data_paths["qwen3_4b"],
        a17, b17, c17, a4, b4, c4, "\n".join(normalized_arrows),
        data_paths["qwen3_1p7b"], data_paths["qwen3_1p7b"],
        data_paths["qwen3_4b"], data_paths["qwen3_4b"],
    )
    common = "\n".join(
        [
            "set encoding utf8",
            "set datafile separator whitespace",
            "set style fill transparent solid 0.25 noborder",
            "set termoption enhanced",
            *annotations,
        ]
    )
    commands = [
        "set terminal svg size 720,340 dynamic enhanced font 'Helvetica,9' background rgb 'white'",
        "set output '%s'" % svg,
        plot_body,
        "set terminal pdfcairo size 7.2in,3.4in enhanced font 'Helvetica,9'",
        "set output '%s'" % pdf,
        plot_body,
        "set terminal pngcairo size 4320,2040 enhanced font 'Helvetica,54'",
        "set output '%s'" % png,
        plot_body,
    ]
    write_new_text(script, common + "\n" + "\n".join(commands) + "\n")
    completed = subprocess.run(
        ["gnuplot", str(script)], check=False, text=True, capture_output=True
    )
    if completed.returncode != 0:
        raise GateIAngularTrajectoryError("gnuplot failed: %s" % completed.stderr)
    result = {}
    for path in (svg, pdf, png):
        if not path.is_file() or path.stat().st_size == 0:
            raise GateIAngularTrajectoryError("figure output absent/empty: %s" % path)
        result[path.name] = file_sha256(path)
    return result


def analyze_data(card_path: Path, run_root: Path) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    try:
        import kneed
    except Exception as exc:
        raise GateIAngularTrajectoryError("kneed==0.8.5 is required") from exc
    if str(getattr(kneed, "__version__", "")) != "0.8.5":
        raise GateIAngularTrajectoryError("kneed version differs from 0.8.5")
    for key in card["models"]:
        seal = load_json(root / key / "formal_seal_receipt.json")
        if seal.get("status") != "PASS":
            raise GateIAngularTrajectoryError("analysis requires both formal seals")
    aggregates: Dict[str, List[Dict[str, Any]]] = {}
    bootstraps: Dict[str, List[List[float]]] = {}
    knees = {}
    aggregate_hashes = {}
    bootstrap_digests = {}
    for key, model in card["models"].items():
        records = [
            validate_sanitized_record(r, model)
            for r in load_jsonl(root / key / "sanitized_angular_distance_records.jsonl")
        ]
        boot = subject_stratified_bootstrap(
            records, int(card["bootstrap"]["replicates"]), int(model["bootstrap_seed"])
        )
        aggregate = aggregate_records(records, boot)
        bootstraps[key] = boot
        aggregates[key] = aggregate
        aggregate_path = root / ("angular_distance_by_transition_%s.csv" % key)
        aggregate_hashes[key] = write_new_csv(aggregate_path, AGGREGATE_FIELDS, aggregate)
        bootstrap_digests[key] = sha256_bytes(canonical_json_bytes(boot))
        knees[key] = analyze_kneedle(aggregate, boot)
    knee_payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "created_at_utc": utc_now(),
        "implementation": "kneed.KneeLocator",
        "required_version": "0.8.5",
        "models": knees,
    }
    knee_hash = write_new_json(root / "angular_distance_kneedle_summary.json", knee_payload)
    source_rows = figure_source_rows(aggregates, card["models"])
    source_hash = write_new_csv(
        root / card["figure"]["source_data"], SOURCE_FIELDS, source_rows
    )
    payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "stage": "I-3A_SCIENTIFIC_DATA_ANALYSIS",
        "aggregate_sha256": aggregate_hashes,
        "bootstrap_digest_sha256": bootstrap_digests,
        "kneedle_sha256": knee_hash,
        "kneedle": knees,
        "figure_source_data_sha256": source_hash,
        "plot_executed": False,
    }
    write_new_json(root / "angular_distance_analysis_receipt.json", payload)
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_figure_source_data(
    path: Path, models: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    persisted = _read_csv(path)
    rows: List[Dict[str, Any]] = []
    for text in persisted:
        if set(text) != set(SOURCE_FIELDS):
            raise GateIAngularTrajectoryError("figure source fields mismatch")
        row = {
            "model_key": text["model_key"],
            "model_repo": text["model_repo"],
            "transition_index": int(text["transition_index"]),
            "normalized_depth": float(text["normalized_depth"]),
            "mean": float(text["mean"]),
            "bootstrap_ci95_low": float(text["bootstrap_ci95_low"]),
            "bootstrap_ci95_high": float(text["bootstrap_ci95_high"]),
        }
        if not all(
            math.isfinite(row[field])
            for field in (
                "normalized_depth",
                "mean",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            )
        ):
            raise GateIAngularTrajectoryError("figure source contains non-finite scalar")
        rows.append(row)
    expected = [
        (key, model["repo"], index, index / int(model["decoder_layers"]))
        for key, model in models.items()
        for index in range(int(model["decoder_layers"]))
    ]
    observed = [
        (
            row["model_key"],
            row["model_repo"],
            row["transition_index"],
            row["normalized_depth"],
        )
        for row in rows
    ]
    if observed != expected:
        raise GateIAngularTrajectoryError("figure source membership/order mismatch")
    return rows


def verify_data(card_path: Path, run_root: Path) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    try:
        import kneed
    except Exception as exc:
        raise GateIAngularTrajectoryError("verifier requires kneed==0.8.5") from exc
    if str(getattr(kneed, "__version__", "")) != "0.8.5":
        raise GateIAngularTrajectoryError("verifier kneed version differs from 0.8.5")
    analysis = load_json(root / "angular_distance_analysis_receipt.json")
    if (
        analysis.get("status") != "PASS"
        or analysis.get("stage") != "I-3A_SCIENTIFIC_DATA_ANALYSIS"
        or analysis.get("plot_executed") is not False
        or "figure_sha256" in analysis
    ):
        raise GateIAngularTrajectoryError("data analysis receipt violates stage split")
    recomputed_aggregates = {}
    recomputed_knees = {}
    bootstrap_digests = {}
    aggregate_hashes = {}
    for key, model in card["models"].items():
        records = [
            validate_sanitized_record(r, model)
            for r in load_jsonl(root / key / "sanitized_angular_distance_records.jsonl")
        ]
        if len(records) != 1531 or len({_identity_tuple(r) for r in records}) != 1531:
            raise GateIAngularTrajectoryError("verifier formal identity closure mismatch")
        boot = subject_stratified_bootstrap(
            records, int(card["bootstrap"]["replicates"]), int(model["bootstrap_seed"])
        )
        aggregate = aggregate_records(records, boot)
        recomputed_aggregates[key] = aggregate
        recomputed_knees[key] = analyze_kneedle(aggregate, boot)
        bootstrap_digests[key] = sha256_bytes(canonical_json_bytes(boot))
        aggregate_path = root / ("angular_distance_by_transition_%s.csv" % key)
        aggregate_hashes[key] = file_sha256(aggregate_path)
        persisted = _read_csv(aggregate_path)
        if len(persisted) != len(aggregate):
            raise GateIAngularTrajectoryError("verifier aggregate row count mismatch")
        for text, row in zip(persisted, aggregate):
            for field in AGGREGATE_FIELDS:
                if field in {"transition_index", "n", "subject_count"}:
                    if int(text[field]) != int(row[field]):
                        raise GateIAngularTrajectoryError("verifier aggregate integer mismatch")
                elif float(text[field]) != float(row[field]):
                    raise GateIAngularTrajectoryError("verifier aggregate scalar mismatch")
    persisted_knees = load_json(root / "angular_distance_kneedle_summary.json")["models"]
    if persisted_knees != recomputed_knees:
        raise GateIAngularTrajectoryError("verifier Kneedle mismatch")
    source_rows = figure_source_rows(recomputed_aggregates, card["models"])
    source_path = root / card["figure"]["source_data"]
    persisted_source = _read_csv(source_path)
    if len(source_rows) != len(persisted_source):
        raise GateIAngularTrajectoryError("verifier figure source membership mismatch")
    for text, row in zip(persisted_source, source_rows):
        for field in SOURCE_FIELDS:
            if field in {"model_key", "model_repo"}:
                if text[field] != row[field]:
                    raise GateIAngularTrajectoryError("verifier figure source text mismatch")
            elif float(text[field]) != float(row[field]):
                raise GateIAngularTrajectoryError("verifier figure source scalar mismatch")
    if aggregate_hashes != analysis["aggregate_sha256"]:
        raise GateIAngularTrajectoryError("data verifier aggregate hashes mismatch")
    if bootstrap_digests != analysis["bootstrap_digest_sha256"]:
        raise GateIAngularTrajectoryError("data verifier bootstrap digests mismatch")
    if file_sha256(root / "angular_distance_kneedle_summary.json") != analysis["kneedle_sha256"]:
        raise GateIAngularTrajectoryError("data verifier Kneedle hash mismatch")
    if file_sha256(source_path) != analysis["figure_source_data_sha256"]:
        raise GateIAngularTrajectoryError("data verifier source-data hash mismatch")
    payload = {
        "schema_version": DATA_VERIFIER_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "stage": "I-3A_DATA_VERIFICATION",
        "fresh_process_pid": os.getpid(),
        "analysis_sha256": file_sha256(root / "angular_distance_analysis_receipt.json"),
        "record_counts": {"qwen3_1p7b": 1531, "qwen3_4b": 1531},
        "subject_counts": {"qwen3_1p7b": 57, "qwen3_4b": 57},
        "transition_counts": {"qwen3_1p7b": 28, "qwen3_4b": 36},
        "aggregate_sha256": aggregate_hashes,
        "bootstrap_digest_sha256": bootstrap_digests,
        "kneedle": recomputed_knees,
        "kneedle_sha256": file_sha256(root / "angular_distance_kneedle_summary.json"),
        "figure_source_data_sha256": file_sha256(source_path),
        "closed_world_record_schema": True,
        "forbidden_fields_absent": True,
        "plot_executed": False,
        "information_barrier": {
            "test_read": False,
            "gold_read": False,
            "outcome_read": False,
            "accuracy_computed": False,
            "loop_insertions": 0,
        },
    }
    write_new_json(root / "angular_distance_data_verifier_receipt.json", payload)
    return payload


def _validate_plot_script_state(script_text: str) -> None:
    blocks = script_text.split("set title 'A  Raw transition index' left")[1:]
    if len(blocks) != 3:
        raise GateIAngularTrajectoryError("figure script must contain three render blocks")
    for block in blocks:
        if "set title 'B  Normalized depth' left" not in block:
            raise GateIAngularTrajectoryError("figure script Panel B is missing")
        panel_a, panel_b = block.split("set title 'B  Normalized depth' left", 1)
        autoscale = panel_a.find("set autoscale x")
        unset = panel_a.find("unset arrow")
        plot = panel_a.find("plot ")
        arrows = [
            index
            for index in (
                panel_a.find("set arrow %d" % arrow_id)
                for arrow_id in range(1, 9)
            )
            if index >= 0
        ]
        if not (0 <= autoscale < unset < plot):
            raise GateIAngularTrajectoryError("Panel A x/arrow reset order mismatch")
        if any(not unset < index < plot for index in arrows):
            raise GateIAngularTrajectoryError("Panel A raw arrow order mismatch")
        if "set xrange [0:1]" not in panel_a:
            raise GateIAngularTrajectoryError("Panel B normalized xrange setup is missing")
        if panel_b.find("unset arrow") < 0 or panel_b.find("unset arrow") > panel_b.find("plot "):
            raise GateIAngularTrajectoryError("Panel B normalized arrow reset order mismatch")


def plot_figures(card_path: Path, run_root: Path) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    analysis_path = root / "angular_distance_analysis_receipt.json"
    data_verifier_path = root / "angular_distance_data_verifier_receipt.json"
    analysis = load_json(analysis_path)
    data_verifier = load_json(data_verifier_path)
    if data_verifier.get("status") != "PASS":
        raise GateIAngularTrajectoryError("plotting requires I-3A data verifier PASS")
    if data_verifier.get("analysis_sha256") != file_sha256(analysis_path):
        raise GateIAngularTrajectoryError("plotting data-analysis receipt hash mismatch")
    source_path = root / card["figure"]["source_data"]
    knee_path = root / "angular_distance_kneedle_summary.json"
    if file_sha256(source_path) != data_verifier["figure_source_data_sha256"]:
        raise GateIAngularTrajectoryError("plotting source-data hash mismatch")
    if file_sha256(knee_path) != data_verifier["kneedle_sha256"]:
        raise GateIAngularTrajectoryError("plotting Kneedle hash mismatch")
    source_rows = _load_figure_source_data(source_path, card["models"])
    knees = load_json(knee_path)["models"]
    if knees != data_verifier["kneedle"]:
        raise GateIAngularTrajectoryError("plotting Kneedle payload mismatch")
    figure_hashes = _plot_with_gnuplot(root, card, source_rows, knees)
    plot_script_path = root / "plot_inputs" / "plot.gp"
    _validate_plot_script_state(plot_script_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": FIGURE_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "stage": "I-3B_PRESENTATION",
        "data_verifier_sha256": file_sha256(data_verifier_path),
        "figure_source_data_sha256": file_sha256(source_path),
        "kneedle_sha256": file_sha256(knee_path),
        "plot_script_sha256": file_sha256(plot_script_path),
        "figure_sha256": figure_hashes,
        "raw_record_inputs_consumed": False,
    }
    write_new_json(root / "angular_distance_figure_receipt.json", payload)
    return payload


def verify_figure(card_path: Path, run_root: Path) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    data_verifier_path = root / "angular_distance_data_verifier_receipt.json"
    figure_receipt_path = root / "angular_distance_figure_receipt.json"
    data_verifier = load_json(data_verifier_path)
    figure_receipt = load_json(figure_receipt_path)
    if data_verifier.get("status") != "PASS" or figure_receipt.get("status") != "PASS":
        raise GateIAngularTrajectoryError("figure verifier requires I-3A/I-3B PASS")
    if figure_receipt.get("data_verifier_sha256") != file_sha256(data_verifier_path):
        raise GateIAngularTrajectoryError("figure verifier data receipt mismatch")
    source_path = root / card["figure"]["source_data"]
    knee_path = root / "angular_distance_kneedle_summary.json"
    _load_figure_source_data(source_path, card["models"])
    if figure_receipt.get("figure_source_data_sha256") != file_sha256(source_path):
        raise GateIAngularTrajectoryError("figure verifier source-data mismatch")
    if figure_receipt.get("kneedle_sha256") != file_sha256(knee_path):
        raise GateIAngularTrajectoryError("figure verifier Kneedle mismatch")
    plot_script_path = root / "plot_inputs" / "plot.gp"
    _validate_plot_script_state(plot_script_path.read_text(encoding="utf-8"))
    if figure_receipt.get("plot_script_sha256") != file_sha256(plot_script_path):
        raise GateIAngularTrajectoryError("figure verifier plot-script mismatch")
    base = card["figure"]["base"]
    figure_hashes = {}
    for suffix in ("svg", "pdf", "png"):
        path = root / (base + "." + suffix)
        if not path.is_file() or path.stat().st_size == 0:
            raise GateIAngularTrajectoryError("figure output missing/empty")
        figure_hashes[path.name] = file_sha256(path)
    if figure_hashes != figure_receipt.get("figure_sha256"):
        raise GateIAngularTrajectoryError("figure verifier hashes mismatch")
    payload = {
        "schema_version": VERIFIER_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "stage": "I-3B_FIGURE_VERIFICATION",
        "fresh_process_pid": os.getpid(),
        "data_verifier_sha256": file_sha256(data_verifier_path),
        "figure_receipt_sha256": file_sha256(figure_receipt_path),
        "record_counts": data_verifier["record_counts"],
        "subject_counts": data_verifier["subject_counts"],
        "transition_counts": data_verifier["transition_counts"],
        "bootstrap_digest_sha256": data_verifier["bootstrap_digest_sha256"],
        "kneedle": data_verifier["kneedle"],
        "figure_source_data_sha256": file_sha256(source_path),
        "figure_sha256": figure_hashes,
        "plot_script_state": "PASS",
        "raw_record_inputs_consumed_by_plotter": False,
        "closed_world_record_schema": data_verifier["closed_world_record_schema"],
        "forbidden_fields_absent": data_verifier["forbidden_fields_absent"],
        "information_barrier": data_verifier["information_barrier"],
    }
    write_new_json(root / "angular_distance_verifier_receipt.json", payload)
    return payload


def record_scheduler(
    run_root: Path, model_key: str, stage: str, job_id: str,
    state: Optional[str] = None, resources: Optional[str] = None
) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    if stage not in {"smoke", "formal"}:
        raise GateIAngularTrajectoryError("invalid scheduler stage")
    path = root / "scheduler" / ("%s_%s_%s.json" % (model_key, stage, job_id))
    payload = {
        "schema_version": "loopscope.phase5.gate-i-scheduler-receipt.v1",
        "created_at_utc": utc_now(),
        "model_key": model_key,
        "stage": stage,
        "job_id": str(job_id),
        "state": state,
        "resources": resources,
    }
    write_new_json(path, payload)
    return payload


def write_launchers(
    card_path: Path,
    run_root: Path,
    stage_root: Path,
    implementation_commit: str,
    partition: str,
) -> Dict[str, str]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    if len(implementation_commit) != 40:
        raise GateIAngularTrajectoryError("implementation commit must be exact")
    launch_dir = root / "launchers"
    launch_dir.mkdir(parents=True, exist_ok=False)
    common = """
set -euo pipefail
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT
STAGE_ROOT=%s
RUN_ROOT=%s
CARD=$STAGE_ROOT/configs/loopscope/phase5_angular_trajectory_gate_i_card.json
RUNNER=(env PYTHONPATH=$STAGE_ROOT/src:%s/deps/site-packages %s/bin/python $STAGE_ROOT/scripts/loopscope/run_qwen_phase5_gate_i.py)
""" % (stage_root, root, root, card["runtime_venv"])
    paths = {}
    for model_key in card["models"]:
        smoke = launch_dir / ("%s_smoke.sbatch" % model_key)
        smoke_text = """#!/bin/bash
#SBATCH --job-name=gati-%s-smoke
#SBATCH --partition=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=%s/logs/%%x-%%j.out
#SBATCH --error=%s/logs/%%x-%%j.err
mkdir -p %s/logs
%s "${RUNNER[@]}" acquire --card "$CARD" --run-root "$RUN_ROOT" --model-key %s --mode smoke --smoke-attempt 1
""" % (model_key, partition, root, root, root, common, model_key)
        write_new_text(smoke, smoke_text, executable=True)
        paths[smoke.name] = file_sha256(smoke)
        formal = launch_dir / ("%s_formal.sbatch" % model_key)
        formal_text = """#!/bin/bash
#SBATCH --job-name=gati-%s-formal
#SBATCH --partition=%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=%s/logs/%%x-%%A_%%a.out
#SBATCH --error=%s/logs/%%x-%%A_%%a.err
mkdir -p %s/logs
%s "${RUNNER[@]}" acquire --card "$CARD" --run-root "$RUN_ROOT" --model-key %s --mode formal --shard-id "$SLURM_ARRAY_TASK_ID"
""" % (model_key, partition, root, root, root, common, model_key)
        write_new_text(formal, formal_text, executable=True)
        paths[formal.name] = file_sha256(formal)
    payload = {
        "schema_version": "loopscope.phase5.gate-i-launcher-receipt.v1",
        "created_at_utc": utc_now(),
        "implementation_commit": implementation_commit,
        "partition": partition,
        "stage_root": str(stage_root),
        "launcher_sha256": paths,
    }
    write_new_json(root / "launcher_receipt.json", payload)
    return paths


def finalize(card_path: Path, run_root: Path) -> Dict[str, Any]:
    root = assert_run_root(run_root, must_exist=True)
    card = load_card(card_path)
    verifier = load_json(root / "angular_distance_verifier_receipt.json")
    if verifier.get("status") != "PASS":
        raise GateIAngularTrajectoryError("finalize requires verifier PASS")
    analysis = load_json(root / "angular_distance_analysis_receipt.json")
    data_verifier = load_json(root / "angular_distance_data_verifier_receipt.json")
    figure_receipt = load_json(root / "angular_distance_figure_receipt.json")
    if data_verifier.get("status") != "PASS" or figure_receipt.get("status") != "PASS":
        raise GateIAngularTrajectoryError("finalize requires I-3A and I-3B receipts PASS")
    lines = [
        "# LoopScope 第五阶段 Gate I：相邻层角距离诊断",
        "",
        "本诊断仅描述两个冻结 Qwen3-Base × MMLU validation-1531 cell 在最终非 padding",
        "预答案 token 上的 raw residual-stream 相邻层旋转；不读取 test/gold/outcome，",
        "不计算 accuracy，不运行 loop，也不修改任何历史 selector/window。",
        "",
        "## 闭合结果",
        "",
        "- Qwen3-1.7B：1531 条、57 subjects、28 transitions、1531 次 native forward。",
        "- Qwen3-4B：1531 条、57 subjects、36 transitions、1531 次 native forward。",
        "- FinalNorm prehook 每 forward 恰好一次，捕获 raw B_L，并与 post-FinalNorm hidden 闭合。",
        "- 所有角距离均有限且位于 [0,1]；独立 verifier：PASS。",
        "",
        "## 描述性 Kneedle",
        "",
    ]
    for key in ("qwen3_1p7b", "qwen3_4b"):
        knee = analysis["kneedle"][key]
        lines.append(
            "- %s：model_status=%s；forward=%s；reverse=%s。"
            % (
                key,
                knee["model_status"],
                knee["forward"]["status"],
                knee["reverse"]["status"],
            )
        )
    lines.extend(
        [
            "",
            "Kneedle 仅为 descriptive candidate boundary；`NO_STABLE_KNEE` 是合同允许的",
            "描述性结果，不构成 selector、eligibility 或历史窗口变更。",
            "",
        ]
    )
    report = root / "angular_distance_summary_zh.md"
    report_sha = write_new_text(report, "\n".join(lines))
    artifact_names = [
        "qwen3_1p7b/sanitized_angular_distance_records.jsonl",
        "qwen3_4b/sanitized_angular_distance_records.jsonl",
        "angular_distance_by_transition_qwen3_1p7b.csv",
        "angular_distance_by_transition_qwen3_4b.csv",
        "angular_distance_kneedle_summary.json",
        card["figure"]["source_data"],
        card["figure"]["base"] + ".svg",
        card["figure"]["base"] + ".pdf",
        card["figure"]["base"] + ".png",
        "angular_distance_summary_zh.md",
        "angular_distance_analysis_receipt.json",
        "angular_distance_data_verifier_receipt.json",
        "angular_distance_figure_receipt.json",
        "angular_distance_verifier_receipt.json",
    ]
    artifacts = {name: file_sha256(root / name) for name in artifact_names}
    payload = {
        "schema_version": FINAL_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "READY_FOR_GAT_I_FINAL_AUDIT",
        "executor_thread_id": EXECUTOR_THREAD,
        "planning_thread_id": PLANNING_THREAD,
        "run_root": str(root),
        "card_sha256": file_sha256(card_path),
        "implementation_sha256": implementation_hashes(),
        "artifacts_sha256": artifacts,
        "report_sha256": report_sha,
        "analysis_sha256": file_sha256(root / "angular_distance_analysis_receipt.json"),
        "data_verifier_sha256": file_sha256(
            root / "angular_distance_data_verifier_receipt.json"
        ),
        "figure_receipt_sha256": file_sha256(root / "angular_distance_figure_receipt.json"),
        "verifier_sha256": file_sha256(root / "angular_distance_verifier_receipt.json"),
        "information_barrier": verifier["information_barrier"],
        "decision_requested": "PASS / PASS_WITH_FIXES / BLOCK",
    }
    write_new_json(root / "phase5_gate_i_manifest_receipt.json", payload)
    return payload


def dry_run(card_path: Path) -> Dict[str, Any]:
    card = load_card(card_path)
    return {
        "status": "DRY_RUN_PASS",
        "card_sha256": file_sha256(card_path),
        "models": {
            key: {
                "revision": model["revision"],
                "dtype": model["dtype"],
                "boundaries": model["decoder_layers"] + 1,
                "transitions": model["decoder_layers"],
            }
            for key, model in card["models"].items()
        },
        "population": card["pool"]["record_count"],
        "subjects": card["pool"]["subject_count"],
        "forbidden": ["test", "gold", "outcome", "accuracy", "loop"],
    }
