"""Outcome-blind Phase 5 Gate C sealed MMLU acquisition.

The producer may mechanically project stable sample identities from the in-memory
lm-eval payload.  The sealer never parses ``results.json`` and can close only
scheduler, recipe, identity, file-stat, and SHA256 evidence.  Scientific outcome
values remain sealed for a separately authorized Gate D.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase3_schema import ordered_identity_sha256
from tflt.loopscope.phase5_schema import canonical_json_bytes, file_sha256


GATE = "C"
EXECUTOR_THREAD_ID = "019f884d-e5c0-7750-bc8e-7735c7984bb9"
PLANNING_THREAD_ID = "019f8604-4717-7be2-8bf8-9d4a26a3d7f7"
ADMITTED_COMMIT = "0d7e33d7b095fcae4b5dd7fbc14fdb21f2410205"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
WORKSPACE = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope"
)
AUTHORIZED_RUN_ROOT = WORKSPACE / "runs/phase5-gate-c-20260722T053058Z"
GATE_B_ROOT = WORKSPACE / "runs/phase5-gate-b-repair-b36-native-20260722T035452Z"
PHASE3_C_ROOT = WORKSPACE / "runs/phase3-p3c-20260716T064601Z"

MODEL_REPO = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
CARD_PATH = Path("configs/loopscope/phase5_card.json")
CARD_FILE_SHA256 = "8d25996f994d131ea0fd9ca78f83c90680a67c41c7c460f7e4e8b4ae5ea42a73"
CARD_MANIFEST_SHA256 = "75eed3e30c623f5eedd63e84e93bfd3e4de50cfb490b23f3a2d31683f9947143"
TRAJECTORY_PATH = GATE_B_ROOT / "formal/validation1531_phase5_trajectories.jsonl"
TRAJECTORY_FILE_SHA256 = "aa4894e632f557ddc4614e88d06ec1fc4c525463be7531aef69fa3c28671edf9"
SELECTOR_PATH = GATE_B_ROOT / "selector/selector_freeze.json"
SELECTOR_FILE_SHA256 = "7c42f0e22ce2caa5784edb33dcfc271e31291229e107b478928fd77181cf1a0b"
SELECTOR_MANIFEST_SHA256 = "4c66ef75ee8efc492245b341ee79c0147a3ba3b973a0de569ef6ca5dcb858d97"
PANEL_PATH = GATE_B_ROOT / "selector/outcome_panel.json"
PANEL_FILE_SHA256 = "3ea83d12e5185edffbf01d5b916612e616d3a7cf41881601fa7b8931918be0e9"
PANEL_MANIFEST_SHA256 = "f28ce84d54e69f2179209f276ecd69f39c3e94359eac4f2aed84b09167be4961"
GATE_B_RECEIPT_PATH = GATE_B_ROOT / "verification/gate_b_receipt.json"
GATE_B_RECEIPT_FILE_SHA256 = "bd720065beb2cd7387322346aafdb06e55c6c0131896d6df10164da0e427d941"
TEST_MANIFEST_PATH = PHASE3_C_ROOT / "test14042_identity_content_manifest.json"
TEST_METADATA_PATH = PHASE3_C_ROOT / "test14042_identity_content_metadata.jsonl"
TEST_MANIFEST_FILE_SHA256 = "59500340aa64d91810cd5a5d40f3d96ded5618ab5508543b31eea93050e7f3e4"
TEST_MANIFEST_INTERNAL_SHA256 = "ec494ff6dd55b4398cd8151eed4609916e8d4c272df4583a4a1574cab3c25386"
TEST_METADATA_FILE_SHA256 = "3c281447cc7618e58c292a20c13b73d8f42f081692a6927372cf1ee1eebf84df"
TEST_ORDERED_IDENTITY_SHA256 = "2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532"

# Ordered authorization: (cell_id, role, window).
CELLS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("baseline_no_loop", "baseline", None),
    ("fixed_15_18", "fixed_comparator", "15:18"),
    ("blind_high1_4_7", "blind_high", "4:7"),
    ("blind_high2_5_8", "blind_high", "5:8"),
    ("blind_high3_13_16", "blind_high", "13:16"),
    ("blind_low1_8_11", "blind_low", "8:11"),
    ("blind_low2_9_12", "blind_low", "9:12"),
    ("blind_low3_6_9", "blind_low", "6:9"),
)
SMOKE_IDENTITIES: Tuple[Tuple[str, str, str], ...] = (
    ("mmlu_abstract_algebra", "0", "307538e6db7f0f40546171e82b7606ccfa2b51dca7d2fdf65a94c350b7413ab1"),
    ("mmlu_abstract_algebra", "1", "dd14a588c453fd5ce4a77194d428f7ad76cb67b7939ae751bcb2ce9d0e4c48bd"),
    ("mmlu_anatomy", "0", "260435a901c78f8f4fb792cb2893f23dc444053829742d2397f3974fe8deb4c2"),
    ("mmlu_anatomy", "1", "edca44324847f85192304489cf7f6021744efddf49294af461c85335da3eed31"),
)
IMPLEMENTATION_PATHS = (
    "src/tflt/loopscope/phase5_outcome.py",
    "scripts/loopscope/run_qwen4base_phase5_gate_c.py",
    "tests/test_loopscope_phase5_gate_c.py",
)
MANIFEST_NAMES = {
    "smoke": "phase5_gate_c_smoke_launch_manifest.json",
    "formal": "phase5_gate_c_formal_launch_manifest.json",
}
SUBMISSION_NAMES = {
    "smoke": "phase5_gate_c_smoke_submission.json",
    "formal": "phase5_gate_c_formal_submission.json",
}
SEAL_NAMES = {
    "smoke": "phase5_gate_c_smoke_seal.json",
    "formal": "phase5_gate_c_completion_receipt.json",
}
SACCT_FIELDS = (
    "JobID", "State", "ExitCode", "ElapsedRaw", "AllocTRES", "Partition", "NodeList"
)


class Phase5OutcomeError(ValueError):
    """Fail-closed Gate C contract or provenance error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise Phase5OutcomeError("git provenance command failed: %s" % completed.stderr.strip())
    return completed.stdout.strip()


def git_provenance(expected_commit: str, *, remote_required: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if remote_required and root != REMOTE_REPO:
        raise Phase5OutcomeError("remote action is outside the dedicated LoopScope clone")
    observed = {
        "repo": str(root),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "origin_loopscope": _git(root, "rev-parse", "origin/loopscope"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }
    if (
        observed["branch"] != "loopscope"
        or observed["commit"] != expected_commit
        or observed["origin_loopscope"] != expected_commit
        or observed["dirty"]
    ):
        raise Phase5OutcomeError("Git provenance is not clean/exact at the expected pushed commit")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ADMITTED_COMMIT, expected_commit],
        check=False,
    ).returncode:
        raise Phase5OutcomeError("implementation commit is not descended from Gate B admission")
    return observed


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            Phase5OutcomeError("non-finite JSON constant: %s" % value)
        ))
    if not isinstance(value, dict):
        raise Phase5OutcomeError("expected JSON object: %s" % path)
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(body).hexdigest()


def _write_new_text(path: Path, value: str, *, executable: bool = False) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value.encode("utf-8")
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return hashlib.sha256(body).hexdigest()


def _attach_manifest(value: Dict[str, Any]) -> Dict[str, Any]:
    value["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def _verify_manifest(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise Phase5OutcomeError("internal manifest SHA256 mismatch")


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise Phase5OutcomeError("missing Gate C implementation path: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def _load_test_metadata() -> List[Dict[str, Any]]:
    if file_sha256(TEST_METADATA_PATH) != TEST_METADATA_FILE_SHA256:
        raise Phase5OutcomeError("test metadata byte hash differs")
    records = []
    with TEST_METADATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append({
                "identity": dict(row["identity"]),
                "subject": str(row["subject"]),
                "split": str(row["split"]),
            })
    if len(records) != 14042 or len({row["subject"] for row in records}) != 57:
        raise Phase5OutcomeError("test metadata does not close 14042/57")
    if ordered_identity_sha256([row["identity"] for row in records]) != TEST_ORDERED_IDENTITY_SHA256:
        raise Phase5OutcomeError("test metadata ordered identity hash differs")
    return records


def validate_frozen_inputs() -> Dict[str, Any]:
    expected = {
        repository_root() / CARD_PATH: CARD_FILE_SHA256,
        TRAJECTORY_PATH: TRAJECTORY_FILE_SHA256,
        SELECTOR_PATH: SELECTOR_FILE_SHA256,
        PANEL_PATH: PANEL_FILE_SHA256,
        GATE_B_RECEIPT_PATH: GATE_B_RECEIPT_FILE_SHA256,
        TEST_MANIFEST_PATH: TEST_MANIFEST_FILE_SHA256,
        TEST_METADATA_PATH: TEST_METADATA_FILE_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise Phase5OutcomeError("frozen upstream byte hash differs: %s" % path)
    card = _read_json(repository_root() / CARD_PATH)
    selector = _read_json(SELECTOR_PATH)
    panel = _read_json(PANEL_PATH)
    test_manifest = _read_json(TEST_MANIFEST_PATH)
    if card.get("manifest_sha256") != CARD_MANIFEST_SHA256:
        raise Phase5OutcomeError("Phase 5 card internal hash differs")
    if selector.get("manifest_sha256") != SELECTOR_MANIFEST_SHA256:
        raise Phase5OutcomeError("selector freeze internal hash differs")
    if selector.get("window_decision") != "ABSTAIN_NO_POINT_ELIGIBLE":
        raise Phase5OutcomeError("Gate B frozen selector decision differs")
    if panel.get("manifest_sha256") != PANEL_MANIFEST_SHA256:
        raise Phase5OutcomeError("outcome panel internal hash differs")
    if panel.get("unique_cells") != ["baseline", "15:18", "4:7", "5:8", "13:16", "8:11", "9:12", "6:9"]:
        raise Phase5OutcomeError("outcome panel membership/order differs")
    if test_manifest.get("manifest_sha256") != TEST_MANIFEST_INTERNAL_SHA256:
        raise Phase5OutcomeError("test identity manifest internal hash differs")
    if (
        test_manifest.get("record_count") != 14042
        or test_manifest.get("subject_count") != 57
        or test_manifest.get("ordered_identity_sha256") != TEST_ORDERED_IDENTITY_SHA256
    ):
        raise Phase5OutcomeError("test identity manifest closure differs")
    _load_test_metadata()
    return {"card": card, "selector": selector, "panel": panel, "test_manifest": test_manifest}


def _cell_eval_argv(
    mode: str, output_dir: Path, window: Optional[str], batch_size: str
) -> List[str]:
    tasks = "mmlu_abstract_algebra,mmlu_anatomy" if mode == "smoke" else "mmlu"
    argv = [
        "python", "-m", "tflt.eval_runner", "--model", MODEL_REPO,
        "--revision", MODEL_REVISION, "--tasks", tasks,
        "--output-dir", str(output_dir), "--num-fewshot", "5",
        "--batch-size", batch_size, "--dtype", "bfloat16",
    ]
    if mode == "smoke":
        argv.extend(("--limit", "2"))
    if window is not None:
        argv.extend((
            "--loop", "--window", window, "--k", "3",
            "--iteration-mode", "block", "--strategy", "euler",
            "--alpha", "1.0", "--beta", "0.0",
            "--cache-strategy", "first", "--decode-mode", "bypass",
        ))
    return argv


def _cell_argv(mode: str, run_root: Path, index: int, expected_commit: str) -> List[str]:
    return [
        "python", "scripts/loopscope/run_qwen4base_phase5_gate_c.py", "run-cell",
        "--mode", mode, "--run-root", str(run_root), "--cell-index", str(index),
        "--expected-commit", expected_commit,
    ]


def build_launch_manifest(
    *, mode: str, run_root: Path, expected_commit: str, partition: str, gres: str,
    qos: Optional[str], account: Optional[str], array_throttle: int,
    time_limit: str, batch_size: str, created_at_utc: str,
) -> Dict[str, Any]:
    if mode not in {"smoke", "formal"}:
        raise Phase5OutcomeError("mode must be smoke or formal")
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise Phase5OutcomeError("Gate C run root differs from authorization")
    if not partition or not gres or not time_limit or not batch_size:
        raise Phase5OutcomeError("scheduler and batch parameters must be explicit")
    if not isinstance(array_throttle, int) or not 1 <= array_throttle <= 8:
        raise Phase5OutcomeError("array throttle must be in 1..8")
    cells = []
    for index, (cell_id, role, window) in enumerate(CELLS):
        cell_root = AUTHORIZED_RUN_ROOT / mode / ("cell-%02d-%s" % (index, cell_id))
        cells.append({
            "index": index, "cell_id": cell_id, "role": role, "window": window,
            "loop_enabled": window is not None, "cell_root": str(cell_root),
            "eval_output_dir": str(cell_root / "eval"),
            "argv": _cell_argv(mode, AUTHORIZED_RUN_ROOT, index, expected_commit),
            "eval_argv": _cell_eval_argv(mode, cell_root / "eval", window, batch_size),
            "automatic_retry": False,
        })
    value: Dict[str, Any] = {
        "schema_version": "loopscope.phase5.gate-c-%s-launch.v1" % mode,
        "artifact_role": "phase5_gate_c_frozen_eight_cell_%s" % mode,
        "gate": GATE, "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID, "created_at_utc": created_at_utc,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git": {"branch": "loopscope", "commit": expected_commit, "dirty": False},
        "implementation_sha256": implementation_hashes(),
        "frozen_inputs": {
            "admitted_commit": ADMITTED_COMMIT,
            "trajectory_file_sha256": TRAJECTORY_FILE_SHA256,
            "selector_file_sha256": SELECTOR_FILE_SHA256,
            "selector_manifest_sha256": SELECTOR_MANIFEST_SHA256,
            "panel_file_sha256": PANEL_FILE_SHA256,
            "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
            "gate_b_receipt_file_sha256": GATE_B_RECEIPT_FILE_SHA256,
            "card_file_sha256": CARD_FILE_SHA256,
            "card_manifest_sha256": CARD_MANIFEST_SHA256,
            "test_manifest_file_sha256": TEST_MANIFEST_FILE_SHA256,
            "test_manifest_internal_sha256": TEST_MANIFEST_INTERNAL_SHA256,
            "test_ordered_identity_sha256": TEST_ORDERED_IDENTITY_SHA256,
        },
        "recipe": {
            "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION, "dtype": "bfloat16",
            "dataset": "cais/mmlu", "dataset_revision": DATASET_REVISION,
            "task_group": "mmlu", "split": "test", "num_fewshot": 5,
            "lm_eval_version": "0.4.11", "metric": "acc,none",
            "apply_chat_template": False, "fewshot_as_multiturn": False,
            "batch_size": batch_size, "limit": 2 if mode == "smoke" else None,
            "window_width": 4, "k": 3, "iteration_mode": "block",
            "strategy_cli": "euler", "strategy_kernel": "damped_euler_alias",
            "alpha": 1.0, "beta": 0.0, "step_size": 1.0 / 3.0,
            "total_horizon": 1.0, "cache_strategy": "first",
            "decode_mode": "bypass", "log_samples": True,
        },
        "population": {
            "test_manifest_path": str(TEST_MANIFEST_PATH),
            "test_metadata_path": str(TEST_METADATA_PATH),
            "required_count": 4 if mode == "smoke" else 14042,
            "required_subject_count": 2 if mode == "smoke" else 57,
            "ordered_identity_sha256": (
                ordered_identity_sha256([
                    {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}
                    for task, doc_id, doc_hash in SMOKE_IDENTITIES
                ]) if mode == "smoke" else TEST_ORDERED_IDENTITY_SHA256
            ),
            "smoke_identities": [
                {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}
                for task, doc_id, doc_hash in SMOKE_IDENTITIES
            ] if mode == "smoke" else None,
        },
        "scheduler": {
            "partition": partition, "gres": gres, "qos": qos, "account": account,
            "array": "0-7%%%d" % array_throttle, "array_throttle": array_throttle,
            "cpus_per_task": 8, "memory": "64G", "time_limit": time_limit,
            "requeue": False,
        },
        "cells": cells, "cell_count": 8,
        "information_barrier": {
            "producer_identity_projection_only": True,
            "sealer_results_parse_authorized": False,
            "accuracy_or_gain_computation_authorized": False,
            "gate_d_unseal_authorized": False,
        },
        "outcome_values_consumed": False,
    }
    _attach_manifest(value)
    validate_launch_manifest(value, mode=mode)
    return value


def validate_launch_manifest(value: Mapping[str, Any], *, mode: str) -> None:
    _verify_manifest(value)
    if (
        value.get("gate") != GATE
        or value.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or value.get("planning_thread_id") != PLANNING_THREAD_ID
        or value.get("run_root") != str(AUTHORIZED_RUN_ROOT)
        or value.get("cell_count") != 8
        or value.get("outcome_values_consumed") is not False
    ):
        raise Phase5OutcomeError("Gate C launch identity differs")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 8:
        raise Phase5OutcomeError("Gate C requires exactly eight cells")
    observed = [(cell.get("cell_id"), cell.get("role"), cell.get("window")) for cell in cells]
    if observed != list(CELLS) or [cell.get("index") for cell in cells] != list(range(8)):
        raise Phase5OutcomeError("Gate C cell membership/order differs")
    recipe = value.get("recipe", {})
    expected = {
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION, "dtype": "bfloat16",
        "dataset_revision": DATASET_REVISION, "task_group": "mmlu", "split": "test",
        "num_fewshot": 5, "lm_eval_version": "0.4.11", "metric": "acc,none",
        "window_width": 4, "k": 3, "iteration_mode": "block",
        "strategy_cli": "euler", "strategy_kernel": "damped_euler_alias",
        "alpha": 1.0, "beta": 0.0, "cache_strategy": "first", "decode_mode": "bypass",
    }
    if any(recipe.get(key) != item for key, item in expected.items()):
        raise Phase5OutcomeError("Gate C frozen recipe drift")
    if mode == "smoke":
        if recipe.get("limit") != 2 or value.get("population", {}).get("required_count") != 4:
            raise Phase5OutcomeError("Gate C smoke population differs")
    elif mode == "formal":
        if recipe.get("limit") is not None or value.get("population", {}).get("required_count") != 14042:
            raise Phase5OutcomeError("Gate C formal population differs")
    else:
        raise Phase5OutcomeError("unknown Gate C mode")
    for cell in cells:
        window = cell["window"]
        expected_eval = _cell_eval_argv(
            mode, Path(cell["eval_output_dir"]), window, str(recipe["batch_size"])
        )
        if cell.get("eval_argv") != expected_eval or cell.get("automatic_retry") is not False:
            raise Phase5OutcomeError("Gate C cell argv/retry drift")
        if window is None:
            if cell.get("loop_enabled") is not False or "--loop" in cell["eval_argv"]:
                raise Phase5OutcomeError("baseline must use the native no-loop path")
        else:
            start, end = (int(item) for item in window.split(":"))
            if end - start + 1 != 4 or cell.get("loop_enabled") is not True:
                raise Phase5OutcomeError("only frozen width-4 loop cells are allowed")


def _common_environment() -> str:
    return """set -euo pipefail
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_ENDPOINT || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONPATH={repo}/src
cd {repo}
""".format(repo=shlex.quote(str(REMOTE_REPO)))


def _launcher_text(cell: Mapping[str, Any]) -> str:
    root = shlex.quote(str(cell["cell_root"]))
    command = shlex.join([str(AUDITED_VENV / "bin/python"), *cell["argv"][1:]])
    return (
        "#!/usr/bin/env bash\n" + _common_environment()
        + "if [[ -e %s ]]; then echo 'refusing existing Gate C cell output' >&2; exit 73; fi\n" % root
        + "exec %s\n" % command
    )


def _array_runner_text(mode: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'case "${SLURM_ARRAY_TASK_ID:?}" in']
    for index in range(8):
        path = AUTHORIZED_RUN_ROOT / mode / "launch" / ("cell-%02d.sh" % index)
        lines.append("  %d) exec %s ;;" % (index, shlex.quote(str(path))))
    lines.extend(["  *) echo 'invalid Gate C array task id' >&2; exit 64 ;;", "esac", ""])
    return "\n".join(lines)


def _submit_text(mode: str, manifest: Mapping[str, Any]) -> str:
    scheduler = manifest["scheduler"]
    args = [
        "/opt/slurm/bin/sbatch", "--parsable", "--job-name=loopscope-p5c-%s" % mode,
        "--partition=%s" % scheduler["partition"], "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=%d" % scheduler["cpus_per_task"],
        "--mem=%s" % scheduler["memory"], "--time=%s" % scheduler["time_limit"],
        "--array=%s" % scheduler["array"], "--no-requeue",
        "--output=%s" % (AUTHORIZED_RUN_ROOT / mode / ("slurm/p5c-%s-%%A_%%a.out" % mode)),
        "--error=%s" % (AUTHORIZED_RUN_ROOT / mode / ("slurm/p5c-%s-%%A_%%a.err" % mode)),
    ]
    if scheduler.get("qos"):
        args.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        args.append("--account=%s" % scheduler["account"])
    args.append(str(AUTHORIZED_RUN_ROOT / mode / "launch/array_runner.sh"))
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(args)


def _materialize_launch(mode: str, manifest: Mapping[str, Any]) -> Dict[str, str]:
    root = AUTHORIZED_RUN_ROOT / mode
    (root / "launch").mkdir(parents=True, exist_ok=False)
    (root / "slurm").mkdir(parents=True, exist_ok=False)
    hashes = {MANIFEST_NAMES[mode]: _write_new_json(root / MANIFEST_NAMES[mode], manifest)}
    for cell in manifest["cells"]:
        path = root / "launch" / ("cell-%02d.sh" % cell["index"])
        hashes[str(path.relative_to(root))] = _write_new_text(path, _launcher_text(cell), executable=True)
    hashes["launch/array_runner.sh"] = _write_new_text(
        root / "launch/array_runner.sh", _array_runner_text(mode), executable=True
    )
    hashes["launch/submit.sh"] = _write_new_text(
        root / "launch/submit.sh", _submit_text(mode, manifest), executable=True
    )
    return hashes


def prepare_smoke(
    *, run_root: Path, expected_commit: str, partition: str, gres: str,
    qos: Optional[str], account: Optional[str], array_throttle: int,
    time_limit: str, batch_size: str,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT or Path(run_root).exists():
        raise Phase5OutcomeError("authorized write-once Gate C root is wrong or exists")
    git = git_provenance(expected_commit, remote_required=True)
    validate_frozen_inputs()
    manifest = build_launch_manifest(
        mode="smoke", run_root=run_root, expected_commit=expected_commit,
        partition=partition, gres=gres, qos=qos, account=account,
        array_throttle=array_throttle, time_limit=time_limit, batch_size=batch_size,
        created_at_utc=utc_now(),
    )
    Path(run_root).mkdir(parents=False, exist_ok=False)
    hashes = _materialize_launch("smoke", manifest)
    receipt = _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-smoke-freeze.v1",
        "gate": GATE, "git": git, "created_at_utc": utc_now(),
        "launch_manifest_sha256": hashes[MANIFEST_NAMES["smoke"]],
        "launch_manifest_internal_sha256": manifest["manifest_sha256"],
        "launcher_sha256": hashes, "outcome_values_consumed": False,
        "status": "SMOKE_PREPARED",
    })
    _write_new_json(AUTHORIZED_RUN_ROOT / "phase5_gate_c_preoutcome_freeze.json", receipt)
    return receipt


def freeze_formal(
    *, run_root: Path, expected_commit: str, partition: str, gres: str,
    qos: Optional[str], account: Optional[str], array_throttle: int,
    time_limit: str, batch_size: str,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise Phase5OutcomeError("formal freeze root differs from authorization")
    git_provenance(expected_commit, remote_required=True)
    smoke = _read_json(AUTHORIZED_RUN_ROOT / SEAL_NAMES["smoke"])
    _verify_manifest(smoke)
    if smoke.get("status") != "NON_SCIENTIFIC_LIMIT_SMOKE_PASS" or smoke.get("cell_count") != 8:
        raise Phase5OutcomeError("formal launch requires exact eight-cell smoke PASS")
    if (AUTHORIZED_RUN_ROOT / "formal").exists():
        raise FileExistsError("formal launch is already frozen")
    validate_frozen_inputs()
    manifest = build_launch_manifest(
        mode="formal", run_root=run_root, expected_commit=expected_commit,
        partition=partition, gres=gres, qos=qos, account=account,
        array_throttle=array_throttle, time_limit=time_limit, batch_size=batch_size,
        created_at_utc=utc_now(),
    )
    hashes = _materialize_launch("formal", manifest)
    receipt = _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-formal-freeze.v1",
        "gate": GATE, "created_at_utc": utc_now(),
        "launch_manifest_sha256": hashes[MANIFEST_NAMES["formal"]],
        "launch_manifest_internal_sha256": manifest["manifest_sha256"],
        "smoke_seal_file_sha256": file_sha256(AUTHORIZED_RUN_ROOT / SEAL_NAMES["smoke"]),
        "launcher_sha256": hashes, "outcome_values_consumed": False,
        "status": "FORMAL_FROZEN",
    })
    _write_new_json(AUTHORIZED_RUN_ROOT / "phase5_gate_c_formal_freeze.json", receipt)
    return receipt


def record_submission(*, mode: str, run_root: Path, job_id: str) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT or not re.fullmatch(r"[0-9]+", str(job_id)):
        raise Phase5OutcomeError("submission root/job id differs")
    manifest_path = AUTHORIZED_RUN_ROOT / mode / MANIFEST_NAMES[mode]
    manifest = _read_json(manifest_path)
    validate_launch_manifest(manifest, mode=mode)
    receipt = _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-%s-submission.v1" % mode,
        "gate": GATE, "mode": mode, "job_id": str(job_id), "submitted_at_utc": utc_now(),
        "launch_manifest_file_sha256": file_sha256(manifest_path),
        "launch_manifest_internal_sha256": manifest["manifest_sha256"],
        "automatic_retry": False, "outcome_values_consumed": False,
    })
    _write_new_json(AUTHORIZED_RUN_ROOT / mode / SUBMISSION_NAMES[mode], receipt)
    return receipt


def _identity_key(identity: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(identity["task"]), str(identity["doc_id"]), str(identity["doc_hash"])


def project_identity_only(
    result: Mapping[str, Any], expected_records: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Mechanically allowlist identity fields; never access score/outcome keys."""

    samples = result.get("samples")
    if not isinstance(samples, Mapping):
        raise Phase5OutcomeError("lm-eval result lacks logged samples")
    observed = set()
    for task, rows in samples.items():
        if not isinstance(task, str) or not isinstance(rows, list):
            raise Phase5OutcomeError("lm-eval sample mapping is malformed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise Phase5OutcomeError("lm-eval sample row is malformed")
            key = (task, str(row.get("doc_id", "")), str(row.get("doc_hash", "")))
            if not key[1] or not re.fullmatch(r"[0-9a-f]{64}", key[2]) or key in observed:
                raise Phase5OutcomeError("lm-eval stable identity is invalid/duplicate")
            observed.add(key)
    expected = [_identity_key(row["identity"]) for row in expected_records]
    if observed != set(expected) or len(observed) != len(expected):
        raise Phase5OutcomeError(
            "lm-eval identity projection is incomplete; missing=%d extra=%d"
            % (len(set(expected) - observed), len(observed - set(expected)))
        )
    return [
        {"ordinal": index, "identity": dict(row["identity"]),
         "subject": row["subject"], "split": row["split"]}
        for index, row in enumerate(expected_records)
    ]


def _smoke_prompts() -> List[Dict[str, str]]:
    """Render the four frozen test prompts without retaining target labels."""

    from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend

    collected = LmEvalMMLURendererBackend().collect(
        task_names=("mmlu_abstract_algebra", "mmlu_anatomy"),
        dataset_revision=DATASET_REVISION,
        target_split="test",
        fewshot_split="dev",
        seed=1234,
        max_targets_per_task=2,
    )
    records = collected.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise Phase5OutcomeError("exact smoke renderer did not return four prompts")
    observed = [
        (str(row.get("task_name")), str(row.get("target_doc_index")), str(row.get("target_doc_sha256")))
        for row in records
    ]
    if observed != list(SMOKE_IDENTITIES):
        raise Phase5OutcomeError("exact smoke renderer identity/hash differs")
    return [
        {"identity": "%s:%s:%s" % identity, "text": str(row["text"]),
         "render_sha256": str(row["render_sha256"])}
        for identity, row in zip(SMOKE_IDENTITIES, records)
    ]


def _audit_command(
    cell: Mapping[str, Any], output_dir: Path, prompts: Sequence[Mapping[str, str]]
) -> List[str]:
    argv = [
        str(AUDITED_VENV / "bin/python"), "-m", "tflt.cli", "audit-loop-effect",
        "--model", MODEL_REPO, "--revision", MODEL_REVISION,
        "--output-dir", str(output_dir), "--window", str(cell["window"]),
        "--k", "3", "--iteration-mode", "block", "--strategy", "euler",
        "--alpha", "1.0", "--beta", "0.0", "--cache-strategy", "first",
        "--decode-mode", "bypass", "--dtype", "bfloat16", "--device", "cuda",
    ]
    for prompt in prompts:
        argv.extend(("--prompt", str(prompt["text"])))
    return argv


def _validate_loop_audit(report: Mapping[str, Any], window: str) -> Dict[str, Any]:
    config = report.get("loop_config", {})
    if (
        config.get("window") != window or config.get("k") != 3
        or config.get("iteration_mode") != "block" or config.get("strategy") != "euler"
        or config.get("alpha") != 1.0 or config.get("beta") != 0.0
        or config.get("cache_strategy") != "first" or config.get("decode_mode") != "bypass"
    ):
        raise Phase5OutcomeError("loop smoke audit recipe differs")
    prompts = report.get("prompts")
    if report.get("overall_decision", {}).get("code") != "loop_effective_logits_changed" or not isinstance(prompts, list):
        raise Phase5OutcomeError("loop smoke audit did not prove effective logits")
    for prompt in prompts:
        decision = prompt.get("decision", {})
        if (
            decision.get("operator_body_calls") != 3
            or decision.get("restore_allclose") is not True
            or decision.get("code") != "loop_effective_logits_changed"
        ):
            raise Phase5OutcomeError("loop smoke audit structural closure differs")
    return {
        "overall_decision": "loop_effective_logits_changed",
        "prompt_count": len(prompts),
        "operator_body_calls_per_prompt": [3 for _ in prompts],
        "restore_allclose_all_prompts": True,
    }


def run_cell(*, mode: str, run_root: Path, cell_index: int, expected_commit: str) -> None:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT or cell_index not in range(8):
        raise Phase5OutcomeError("run-cell root/index differs from authorization")
    git = git_provenance(expected_commit, remote_required=True)
    manifest_path = AUTHORIZED_RUN_ROOT / mode / MANIFEST_NAMES[mode]
    manifest = _read_json(manifest_path)
    validate_launch_manifest(manifest, mode=mode)
    validate_frozen_inputs()
    cell = manifest["cells"][cell_index]
    cell_root = Path(cell["cell_root"])
    cell_root.mkdir(parents=False, exist_ok=False)
    _write_new_json(cell_root / "producer_command.json", {
        "mode": mode, "cell": {key: cell[key] for key in (
            "index", "cell_id", "role", "window", "loop_enabled", "eval_argv"
        )}, "git": git, "launch_manifest_sha256": manifest["manifest_sha256"],
        "outcome_values_consumed": False,
    })
    smoke_prompts = _smoke_prompts() if mode == "smoke" else []
    if mode == "smoke" and cell["loop_enabled"]:
        audit_dir = cell_root / "audit"
        completed = subprocess.run(_audit_command(cell, audit_dir, smoke_prompts), check=False)
        if completed.returncode:
            raise Phase5OutcomeError("loop-effect audit failed with exit %d" % completed.returncode)
        structural = _validate_loop_audit(_read_json(audit_dir / "audit_report.json"), str(cell["window"]))
    elif mode == "smoke":
        structural = {
            "overall_decision": "native_baseline_no_loop_activity",
            "prompt_count": 4, "operator_body_calls_per_prompt": [0, 0, 0, 0],
            "restore_allclose_all_prompts": True,
        }
    else:
        structural = {"formal_loop_audit_not_persisted": True}
    _write_new_json(cell_root / "structural_receipt.json", _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-cell-structural.v1",
        "mode": mode, "cell_id": cell["cell_id"], "window": cell["window"],
        "loop_enabled": cell["loop_enabled"], "structural": structural,
        "fixed_smoke_prompt_identity": [row["identity"] for row in smoke_prompts],
        "fixed_smoke_prompt_render_sha256": [row["render_sha256"] for row in smoke_prompts],
        "outcome_values_consumed": False,
    }))

    eval_argv = [str(AUDITED_VENV / "bin/python"), *cell["eval_argv"][1:]]
    completed = subprocess.run(eval_argv, check=False)
    if completed.returncode:
        raise Phase5OutcomeError("lm-eval producer failed with exit %d" % completed.returncode)
    eval_dir = Path(cell["eval_output_dir"])
    results_path = eval_dir / "results.json"
    command_path = eval_dir / "command_args.json"
    revision_path = eval_dir / "model_revision.json"
    for path in (results_path, command_path, revision_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise Phase5OutcomeError("producer artifact is absent/empty/symlinked: %s" % path)
    all_records = _load_test_metadata()
    if mode == "smoke":
        expected_keys = set(SMOKE_IDENTITIES)
        expected_records = [row for row in all_records if _identity_key(row["identity"]) in expected_keys]
        if [_identity_key(row["identity"]) for row in expected_records] != list(SMOKE_IDENTITIES):
            raise Phase5OutcomeError("frozen smoke identity order differs from test metadata")
    else:
        expected_records = all_records
    # Producer-only allowlist projection.  The result object is discarded immediately.
    result = _read_json(results_path)
    projected = project_identity_only(result, expected_records)
    del result
    identity = _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-cell-identity.v1",
        "mode": mode, "cell_id": cell["cell_id"], "role": cell["role"],
        "window": cell["window"], "loop_enabled": cell["loop_enabled"],
        "record_count": len(projected),
        "subject_count": len({row["subject"] for row in projected}),
        "ordered_identity_sha256": ordered_identity_sha256([row["identity"] for row in projected]),
        "records": projected, "projection_allowlist": [
            "task", "doc_id", "doc_hash", "subject", "split", "canonical_ordinal"
        ], "outcome_fields_accessed": False, "outcome_values_consumed": False,
    })
    identity_path = cell_root / "identity_sidecar.json"
    _write_new_json(identity_path, identity)
    completion = _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-cell-completion.v1",
        "gate": GATE, "mode": mode, "cell_id": cell["cell_id"],
        "results": {"path": str(results_path), "size_bytes": results_path.stat().st_size,
                    "file_sha256": file_sha256(results_path)},
        "command_args_file_sha256": file_sha256(command_path),
        "model_revision_file_sha256": file_sha256(revision_path),
        "identity_sidecar_file_sha256": file_sha256(identity_path),
        "structural_receipt_file_sha256": file_sha256(cell_root / "structural_receipt.json"),
        "exit_state": "PRODUCER_COMPLETED", "outcome_values_consumed": False,
    })
    _write_new_json(cell_root / "producer_completion.json", completion)


def query_sacct(job_id: str) -> List[Dict[str, str]]:
    completed = subprocess.run(
        ["/opt/slurm/bin/sacct", "-j", str(job_id), "--array", "--noheader", "--parsable2",
         "--format=" + ",".join(SACCT_FIELDS)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise Phase5OutcomeError("sacct failed: %s" % completed.stderr.strip())
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) > len(SACCT_FIELDS) and values[-1] == "":
            values = values[:-1]
        if len(values) != len(SACCT_FIELDS):
            raise Phase5OutcomeError("unexpected sacct field count")
        rows.append(dict(zip(SACCT_FIELDS, values)))
    return rows


def validate_scheduler_rows(rows: Sequence[Mapping[str, Any]], job_id: str) -> List[Dict[str, str]]:
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise Phase5OutcomeError("Slurm job id must be decimal")
    pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(str(job_id)))
    tasks = {}
    for raw in rows:
        match = pattern.fullmatch(str(raw.get("JobID", "")))
        if not match:
            continue
        index = int(match.group(1))
        if index in tasks:
            raise Phase5OutcomeError("scheduler rows contain duplicate array task")
        tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    if set(tasks) != set(range(8)):
        raise Phase5OutcomeError("sealed completion requires exact eight scheduler tasks")
    for index, row in tasks.items():
        state = row["State"].split()[0].rstrip("+")
        if state != "COMPLETED" or row["ExitCode"] != "0:0":
            raise Phase5OutcomeError(
                "scheduler task %d is not COMPLETED/0:0: %s %s"
                % (index, row["State"], row["ExitCode"])
            )
    return [tasks[index] for index in range(8)]


def _gpu_count(alloc_tres: str) -> int:
    matches = re.findall(r"(?:^|,)gres/gpu(?::[^=,]+)?=([0-9]+)(?:,|$)", alloc_tres)
    if len(matches) > 1:
        raise Phase5OutcomeError("ambiguous GPU allocation TRES")
    return int(matches[0]) if matches else 0


def _close_cells(mode: str, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries = []
    for cell in manifest["cells"]:
        root = Path(cell["cell_root"])
        completion_path = root / "producer_completion.json"
        identity_path = root / "identity_sidecar.json"
        structural_path = root / "structural_receipt.json"
        for path in (completion_path, identity_path, structural_path):
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise Phase5OutcomeError("cell closure artifact is absent/empty/symlinked")
        completion = _read_json(completion_path)
        identity = _read_json(identity_path)
        structural = _read_json(structural_path)
        _verify_manifest(completion)
        _verify_manifest(identity)
        _verify_manifest(structural)
        if (
            completion.get("cell_id") != cell["cell_id"]
            or completion.get("exit_state") != "PRODUCER_COMPLETED"
            or identity.get("cell_id") != cell["cell_id"]
            or identity.get("record_count") != manifest["population"]["required_count"]
            or identity.get("subject_count") != manifest["population"]["required_subject_count"]
            or identity.get("ordered_identity_sha256") != manifest["population"]["ordered_identity_sha256"]
            or identity.get("outcome_fields_accessed") is not False
            or identity.get("outcome_values_consumed") is not False
        ):
            raise Phase5OutcomeError("cell identity/completion closure differs")
        results = completion.get("results", {})
        results_path = Path(str(results.get("path", "")))
        if (
            results_path != Path(cell["eval_output_dir"]) / "results.json"
            or not results_path.is_file() or results_path.is_symlink()
            or results_path.stat().st_size != results.get("size_bytes")
            or file_sha256(results_path) != results.get("file_sha256")
        ):
            raise Phase5OutcomeError("sealed results stat/SHA differs")
        if mode == "smoke":
            info = structural.get("structural", {})
            expected_calls = 0 if not cell["loop_enabled"] else 3
            if any(value != expected_calls for value in info.get("operator_body_calls_per_prompt", [])):
                raise Phase5OutcomeError("smoke body-call closure differs")
            if info.get("restore_allclose_all_prompts") is not True:
                raise Phase5OutcomeError("smoke restore closure differs")
            if cell["loop_enabled"] and info.get("overall_decision") != "loop_effective_logits_changed":
                raise Phase5OutcomeError("smoke loop-effective decision differs")
        identity_file_sha256 = file_sha256(identity_path)
        entries.append({
            "index": cell["index"], "cell_id": cell["cell_id"], "role": cell["role"],
            "window": cell["window"], "results": dict(results),
            "identity_sidecar": {"path": str(identity_path),
                                 "file_sha256": identity_file_sha256,
                                 "manifest_sha256": identity["manifest_sha256"]},
            "producer_completion_file_sha256": file_sha256(completion_path),
            "structural_receipt_file_sha256": file_sha256(structural_path),
        })
    return entries


def seal_stage(
    *, mode: str, run_root: Path, expected_commit: str, job_id: str,
    scheduler_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise Phase5OutcomeError("seal root differs from authorization")
    git = git_provenance(expected_commit, remote_required=True)
    output_path = AUTHORIZED_RUN_ROOT / SEAL_NAMES[mode]
    if output_path.exists():
        raise FileExistsError("sealed stage receipt already exists")
    manifest_path = AUTHORIZED_RUN_ROOT / mode / MANIFEST_NAMES[mode]
    manifest = _read_json(manifest_path)
    validate_launch_manifest(manifest, mode=mode)
    submission = _read_json(AUTHORIZED_RUN_ROOT / mode / SUBMISSION_NAMES[mode])
    _verify_manifest(submission)
    if submission.get("job_id") != str(job_id):
        raise Phase5OutcomeError("submission/job identity differs")
    terminal = validate_scheduler_rows(
        list(scheduler_rows) if scheduler_rows is not None else query_sacct(job_id), job_id
    )
    entries = _close_cells(mode, manifest)
    total_gpu_seconds = 0
    for row in terminal:
        try:
            elapsed = int(row["ElapsedRaw"])
        except ValueError as exc:
            raise Phase5OutcomeError("Slurm ElapsedRaw is invalid") from exc
        total_gpu_seconds += elapsed * _gpu_count(row["AllocTRES"])
    receipt = _attach_manifest({
        "schema_version": "loopscope.phase5.gate-c-%s-seal.v1" % mode,
        "artifact_role": "gate_c_%s_identity_scheduler_stat_sha_closure" % mode,
        "gate": GATE, "mode": mode, "created_at_utc": utc_now(), "git": git,
        "job_id": str(job_id), "scheduler_rows": terminal,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_file_sha256": file_sha256(manifest_path),
        "launch_manifest_internal_sha256": manifest["manifest_sha256"],
        "cell_count": 8, "cells": entries,
        "identity_closure": {
            "record_count_per_cell": manifest["population"]["required_count"],
            "subject_count_per_cell": manifest["population"]["required_subject_count"],
            "ordered_identity_sha256_all_cells": manifest["population"]["ordered_identity_sha256"],
            "missing_duplicate_extra": 0,
        },
        "panel_card_test_manifest_closure": True,
        "resource": {
            "total_gpu_seconds": total_gpu_seconds,
            "total_gpu_hours": total_gpu_seconds / 3600.0,
        },
        "outcome_values_consumed": False,
        "accuracy_or_gain_computed": False,
        "results_payload_parsed_by_sealer": False,
        "gate_d_unseal_performed": False,
        "status": (
            "NON_SCIENTIFIC_LIMIT_SMOKE_PASS"
            if mode == "smoke" else "SEALED_COMPLETE_READY_FOR_PLANNING_AUDIT"
        ),
    })
    _write_new_json(output_path, receipt)
    if mode == "formal":
        _write_new_json(AUTHORIZED_RUN_ROOT / "phase5_gate_c_resource_receipt.json", _attach_manifest({
            "schema_version": "loopscope.phase5.gate-c-resource.v1",
            "gate": GATE, "job_id": str(job_id), "scheduler_rows": terminal,
            "total_gpu_seconds": total_gpu_seconds,
            "total_gpu_hours": total_gpu_seconds / 3600.0,
            "outcome_values_consumed": False,
        }))
    return receipt


__all__ = [
    "ADMITTED_COMMIT", "AUTHORIZED_RUN_ROOT", "CELLS", "Phase5OutcomeError",
    "SMOKE_IDENTITIES", "build_launch_manifest", "freeze_formal", "prepare_smoke",
    "project_identity_only", "record_submission", "run_cell", "seal_stage",
    "validate_frozen_inputs", "validate_launch_manifest", "validate_scheduler_rows",
]
