"""Targeted variable-width full evaluation for LoopScope Phase 3 Gate P3-G.

The sidecar keeps three summary artifacts only:

* p3g_manifest.json freezes one four-cell limit=5 structural smoke and one
  four-cell no-limit full array.
* p3g_targeted_full_analysis.json is written only after exact 4/4 terminal
  completion and outcome-free identity/recipe closure for every full cell.
* p3g_verifier_receipt.json independently reloads the sealed sources and
  recomputes the paired statistics and frozen label.

Smoke accuracy is never extracted or compared.  Comparator outcomes are not
opened until all four full candidates have passed the terminal and identity
seal.  The evaluator, loop implementation, frozen cards, and historical
artifacts remain read-only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase3_p3c import (
    MODEL_REVISION,
    TEST_MANIFEST_NAME,
    TEST_METADATA_NAME,
    _load_outcome_cell,
    _phase1_ordered_sample_identity,
    _validate_cell_revision,
    _verify_result_sample_content,
    load_strict_json,
    load_strict_jsonl,
    validate_test_metadata_records,
)
from tflt.loopscope.phase3_p3e import (
    SACCT_FIELDS,
    _file_metadata,
    _git,
    _index_bytes,
    _quantile,
    exact_mcnemar_p,
    query_sacct,
    utc_now,
    write_new_json,
    write_new_text,
)
from tflt.loopscope.phase3_schema import (
    PHASE3_CARD_BYTE_SHA256,
    attach_manifest_sha256,
    canonical_json_bytes,
    file_sha256,
    load_phase3_card,
    verify_manifest_sha256,
)


GATE = "P3-G"
EXECUTOR_THREAD_ID = "019f6b0a-2979-7662-b380-10bd3ed4b352"
PLANNING_THREAD_ID = "019f670d-24ca-7ad0-b92e-64438aa04ef1"
AUTHORIZED_BASE_COMMIT = "b7e28ea1ab9ec3221bad909f4346963d7e91153f"

REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3g-20260716T130132Z"
)
P3C_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3c-20260716T064601Z"
)
TARGETED_CARD_PATH = REMOTE_REPO / (
    "configs/loopscope/phase3_variable_width_targeted_full_card.json"
)
PHASE3_CARD_PATH = REMOTE_REPO / "configs/loopscope/phase3_card.json"

TARGETED_CARD_SHA256 = "ccc4a285388966dba1043cd78028c535d8dba45e88fb99d9d535135ed8022acf"
EXPECTED_LEGACY_IDENTITY_SHA256 = (
    "872c9f6bd7f40e3c6fa40b13cc862033379ac0a9cbc439a1593653945f3d521d"
)
EXPECTED_CANONICAL_IDENTITY_SHA256 = (
    "2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532"
)
EXPECTED_DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"

CANDIDATE_WINDOWS = ("7:9", "2:9", "13:15", "3:7")
COMPARATOR_ORDER = ("baseline", "12:15")

MANIFEST_NAME = "p3g_manifest.json"
ANALYSIS_NAME = "p3g_targeted_full_analysis.json"
VERIFIER_NAME = "p3g_verifier_receipt.json"

IMPLEMENTATION_RELATIVE_PATHS = (
    "src/tflt/loopscope/phase3_p3g.py",
    "scripts/loopscope/run_qwen17_phase3_p3g.py",
    "tests/test_loopscope_phase3_p3g.py",
)


class P3GError(ValueError):
    """Fail-closed P3-G contract, provenance, seal, or statistics violation."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise P3GError("P3-G implementation path is missing: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def git_provenance(expected_commit: str, *, require_remote_root: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if require_remote_root and root != REMOTE_REPO:
        raise P3GError("P3-G remote action is outside the dedicated clone")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "refs/remotes/origin/loopscope")
    dirty = bool(_git(root, "status", "--porcelain"))
    if branch != "loopscope":
        raise P3GError("P3-G requires branch loopscope")
    if head != str(expected_commit) or origin != str(expected_commit):
        raise P3GError("P3-G HEAD/origin differs from expected implementation commit")
    if dirty:
        raise P3GError("P3-G requires a clean working tree")
    changed = tuple(
        line
        for line in _git(
            root,
            "diff",
            "--name-only",
            "%s..%s" % (AUTHORIZED_BASE_COMMIT, expected_commit),
        ).splitlines()
        if line
    )
    if set(changed) != set(IMPLEMENTATION_RELATIVE_PATHS):
        raise P3GError("P3-G implementation commit changed paths outside exact authorization")
    return {
        "root": str(root),
        "branch": branch,
        "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
        "commit": head,
        "origin_loopscope": origin,
        "dirty": False,
        "changed_paths": list(changed),
    }


def assert_authorized_paths(
    targeted_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    p3g_root: Path,
) -> None:
    exact = {
        Path(targeted_card_path).resolve(): TARGETED_CARD_PATH,
        Path(phase3_card_path).resolve(): PHASE3_CARD_PATH,
        Path(p3c_root).resolve(): P3C_ROOT,
        Path(p3g_root).resolve(): AUTHORIZED_RUN_ROOT,
    }
    for actual, expected in exact.items():
        if actual != expected:
            raise P3GError("P3-G path differs from exact authorization: %s" % actual)


def load_targeted_card(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if file_sha256(path) != TARGETED_CARD_SHA256:
        raise P3GError("P3-G targeted-full card SHA256 mismatch")
    card = load_strict_json(path)
    if (
        card.get("schema_version")
        != "loopscope.phase3.variable_width_targeted_full_card.v1"
        or card.get("card") != "H3_VARIABLE_WIDTH_TARGETED_FULL_V1"
        or card.get("status") != "PLANNING_FROZEN"
        or tuple(card.get("targeted_cells", {}).get("ordered_windows", ()))
        != CANDIDATE_WINDOWS
        or card.get("targeted_cells", {}).get("count") != 4
    ):
        raise P3GError("P3-G targeted card identity/candidate order differs")
    parent = card.get("parent", {})
    if (
        parent.get("phase3_card_sha256") != PHASE3_CARD_BYTE_SHA256
        or parent.get("p3f_audit_decision") != "PASS"
    ):
        raise P3GError("P3-G parent card/admission differs")
    recipe = card.get("model_task_recipe", {})
    exact_recipe = {
        "model": "Qwen/Qwen3-1.7B-Base",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": "cais/mmlu",
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "task_group": "mmlu",
        "split": "test",
        "ordered_sample_count": 14042,
        "subject_count": 57,
        "num_fewshot": 5,
        "fewshot_split": "dev",
        "dtype": "float16",
        "k": 2,
        "fixed_horizon": True,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "step_size": 0.5,
        "total_horizon": 1.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "only_window_may_vary": True,
        "full_limit": None,
    }
    if any(recipe.get(key) != value for key, value in exact_recipe.items()):
        raise P3GError("P3-G frozen scientific recipe differs")
    identity = card.get("identity", {})
    if (
        identity.get("ordered_sample_count") != 14042
        or identity.get("subject_count") != 57
        or identity.get("phase1_legacy_ordered_identity_sha256")
        != EXPECTED_LEGACY_IDENTITY_SHA256
        or identity.get("gold_free_canonical_ordered_identity_sha256")
        != EXPECTED_CANONICAL_IDENTITY_SHA256
    ):
        raise P3GError("P3-G frozen identity differs")
    execution = card.get("execution", {})
    if (
        Path(str(execution.get("repo", ""))).resolve() != REMOTE_REPO
        or Path(str(execution.get("venv", ""))).resolve() != AUDITED_VENV
        or Path(str(execution.get("run_root", ""))).resolve() != AUTHORIZED_RUN_ROOT
        or execution.get("cache_policy")
        != "offline existing caches only; no download or mutation"
    ):
        raise P3GError("P3-G frozen execution paths/cache policy differ")
    bootstrap = card.get("analysis", {}).get("bootstrap", {})
    if (
        bootstrap.get("method") != "joint_paired_sample_bootstrap"
        or bootstrap.get("replicates") != 2000
        or bootstrap.get("seed") != 20260710
        or bootstrap.get("prng") != "python_stdlib_random.Random"
        or bootstrap.get("random_api") != "randrange(14042)"
    ):
        raise P3GError("P3-G frozen bootstrap differs")
    labels = card.get("analysis", {}).get("labels", {})
    if tuple(card.get("analysis", {}).get("label_priority", ())) != (
        "TARGETED_VARIABLE_WIDTH_STRONG_IMPROVEMENT",
        "TARGETED_VARIABLE_WIDTH_POINT_IMPROVEMENT",
        "TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT",
    ) or set(labels) != {
        "TARGETED_VARIABLE_WIDTH_STRONG_IMPROVEMENT",
        "TARGETED_VARIABLE_WIDTH_POINT_IMPROVEMENT",
        "TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT",
    }:
        raise P3GError("P3-G frozen labels differ")
    return card


def _eval_argv(output_dir: Path, window: str, limit: Optional[int]) -> List[str]:
    argv = [
        "python",
        "-m",
        "tflt.eval_runner",
        "--model",
        "qwen3-1.7b-base",
        "--revision",
        MODEL_REVISION,
        "--tasks",
        "mmlu",
        "--output-dir",
        str(output_dir),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    argv.extend(
        [
            "--num-fewshot",
            "5",
            "--batch-size",
            "auto",
            "--dtype",
            "float16",
            "--loop",
            "--window",
            window,
            "--k",
            "2",
            "--iteration-mode",
            "block",
            "--strategy",
            "damped_euler",
            "--alpha",
            "1.0",
            "--beta",
            "0.0",
            "--cache-strategy",
            "last",
            "--decode-mode",
            "bypass",
        ]
    )
    return argv


def _audit_argv(output_dir: Path, window: str) -> List[str]:
    return [
        "python",
        "-m",
        "tflt.cli",
        "audit-loop-effect",
        "--model",
        "qwen3-1.7b-base",
        "--revision",
        MODEL_REVISION,
        "--output-dir",
        str(output_dir),
        "--window",
        window,
        "--k",
        "2",
        "--iteration-mode",
        "block",
        "--strategy",
        "damped_euler",
        "--alpha",
        "1.0",
        "--beta",
        "0.0",
        "--cache-strategy",
        "last",
        "--decode-mode",
        "bypass",
        "--dtype",
        "float16",
        "--device",
        "cuda",
        "--prompt",
        "Question: Which planet is known as the Red Planet?\n"
        "A. Earth\nB. Mars\nC. Jupiter\nD. Venus\nAnswer:",
        "--prompt",
        "In one sentence, explain why reproducibility matters in science.",
    ]


def build_manifest(
    *,
    run_root: Path,
    expected_commit: str,
    partition: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    smoke_time_limit: str,
    full_time_limit: str,
    created_at_utc: str,
    hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P3GError("P3-G manifest run root differs from authorization")
    if not partition or not smoke_time_limit or not full_time_limit:
        raise P3GError("P3-G scheduler partition/time limits must be explicit")
    if (
        isinstance(array_throttle, bool)
        or not isinstance(array_throttle, int)
        or array_throttle < 1
        or array_throttle > 4
    ):
        raise P3GError("P3-G array throttle must be an integer in 1..4")
    smoke_cells = []
    full_cells = []
    for index, window in enumerate(CANDIDATE_WINDOWS):
        token = window.replace(":", "-")
        smoke_root = Path(run_root) / ("smoke/cell-%02d-window-%s" % (index, token))
        full_root = Path(run_root) / ("full/cell-%02d-window-%s" % (index, token))
        smoke_cells.append(
            {
                "index": index,
                "window": window,
                "cell_root": str(smoke_root),
                "audit_output_dir": str(smoke_root / "audit"),
                "limit_output_dir": str(smoke_root / "limit"),
                "launcher": "launch/smoke/cell-%02d.sh" % index,
                "audit_argv": _audit_argv(smoke_root / "audit", window),
                "limit_argv": _eval_argv(smoke_root / "limit", window, 5),
                "accuracy_values_may_not_be_accessed": True,
            }
        )
        full_cells.append(
            {
                "index": index,
                "window": window,
                "output_dir": str(full_root),
                "launcher": "launch/full/cell-%02d.sh" % index,
                "argv": _eval_argv(full_root, window, None),
                "automatic_retry": False,
            }
        )
    scheduler_common = {
        "partition": partition,
        "qos": qos,
        "account": account,
        "array": "0-3%%%d" % array_throttle,
        "initial_throttle": array_throttle,
        "cpus_per_task": 8,
        "memory": "64G",
        "gres": "gpu:a40:1",
        "requeue": False,
    }
    manifest: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3g-manifest.v1",
        "artifact_role": "p3g_exact_four_cell_smoke_and_full_preoutcome_freeze",
        "gate": GATE,
        "created_at_utc": created_at_utc,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(Path(run_root)),
        "git": {
            "repo": str(REMOTE_REPO),
            "branch": "loopscope",
            "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
            "implementation_commit": str(expected_commit),
            "dirty": False,
        },
        "implementation_sha256": dict(hashes or implementation_hashes()),
        "frozen_inputs": {
            "targeted_card_file_sha256": TARGETED_CARD_SHA256,
            "phase3_card_file_sha256": PHASE3_CARD_BYTE_SHA256,
            "candidate_order": list(CANDIDATE_WINDOWS),
            "baseline_results_sha256": (
                "b390880ae8f85d655c86b7b81a138bc12b99b61cf26d22520a8d372bdeafcc25"
            ),
            "best_12_15_results_sha256": (
                "bbcb5940fbefd9ec24402fd3269cf31763c13fa73661c1137c16f372544036d1"
            ),
            "legacy_ordered_identity_sha256": EXPECTED_LEGACY_IDENTITY_SHA256,
            "canonical_ordered_identity_sha256": EXPECTED_CANONICAL_IDENTITY_SHA256,
        },
        "recipe": {
            "model_repo": "Qwen/Qwen3-1.7B-Base",
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dataset": "cais/mmlu",
            "dataset_revision": EXPECTED_DATASET_REVISION,
            "task": "mmlu",
            "split": "test",
            "sample_count": 14042,
            "subject_count": 57,
            "num_fewshot": 5,
            "dtype": "float16",
            "k": 2,
            "fixed_horizon": True,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "step_size": 0.5,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "only_window_varies": True,
        },
        "smoke": {
            "cell_count": 4,
            "window_order": list(CANDIDATE_WINDOWS),
            "limit": 5,
            "cells": smoke_cells,
            "scheduler": dict(scheduler_common, time_limit=smoke_time_limit),
            "inspection_scope": [
                "terminal_state",
                "recipe",
                "task_and_sample_structure",
                "loop_effect",
            ],
            "accuracy_values_consumed": False,
        },
        "full": {
            "cell_count": 4,
            "window_order": list(CANDIDATE_WINDOWS),
            "limit": None,
            "cells": full_cells,
            "scheduler": dict(scheduler_common, time_limit=full_time_limit),
            "sealed_completion": {
                "required_terminal_tasks": 4,
                "required_state": "COMPLETED",
                "required_exit_code": "0:0",
                "required_tasks_per_cell": 57,
                "required_samples_per_cell": 14042,
                "partial_outcome_viewing": False,
            },
            "automatic_retry": False,
        },
        "analysis": {
            "comparators": ["baseline", "12:15"],
            "bootstrap_method": "joint_paired_sample_bootstrap",
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20260710,
            "joint_index_reuse_all_eight_contrasts": True,
            "exact_mcnemar_comparator": "12:15",
            "holm_family_size": 4,
            "holm_familywise_alpha": 0.05,
            "candidate_rank_order": ["accuracy_descending", "card_order_ascending"],
        },
        "required_summary_artifacts": [MANIFEST_NAME, ANALYSIS_NAME, VERIFIER_NAME],
        "outcome_values_consumed": False,
    }
    attach_manifest_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    verify_manifest_sha256(manifest)
    if (
        manifest.get("schema_version") != "loopscope.phase3.p3g-manifest.v1"
        or manifest.get("gate") != GATE
        or manifest.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or Path(str(manifest.get("run_root", ""))).resolve() != AUTHORIZED_RUN_ROOT
        or manifest.get("outcome_values_consumed") is not False
        or tuple(manifest.get("required_summary_artifacts", ()))
        != (MANIFEST_NAME, ANALYSIS_NAME, VERIFIER_NAME)
    ):
        raise P3GError("P3-G manifest identity/summary contract differs")
    if set(manifest.get("implementation_sha256", {})) != set(
        IMPLEMENTATION_RELATIVE_PATHS
    ):
        raise P3GError("P3-G implementation hash set differs")
    for stage_name in ("smoke", "full"):
        stage = manifest.get(stage_name, {})
        cells = stage.get("cells")
        if (
            stage.get("cell_count") != 4
            or tuple(stage.get("window_order", ())) != CANDIDATE_WINDOWS
            or not isinstance(cells, list)
            or len(cells) != 4
            or [cell.get("index") for cell in cells] != list(range(4))
            or tuple(cell.get("window") for cell in cells) != CANDIDATE_WINDOWS
        ):
            raise P3GError("P3-G %s matrix differs from exact four cells" % stage_name)
    if manifest["smoke"].get("limit") != 5:
        raise P3GError("P3-G smoke must remain limit=5")
    for cell in manifest["smoke"]["cells"]:
        root = AUTHORIZED_RUN_ROOT / (
            "smoke/cell-%02d-window-%s"
            % (cell["index"], cell["window"].replace(":", "-"))
        )
        if (
            Path(cell["cell_root"]).resolve() != root
            or cell["audit_argv"] != _audit_argv(root / "audit", cell["window"])
            or cell["limit_argv"] != _eval_argv(root / "limit", cell["window"], 5)
            or cell.get("accuracy_values_may_not_be_accessed") is not True
        ):
            raise P3GError("P3-G smoke cell differs")
    if manifest["full"].get("limit") is not None:
        raise P3GError("P3-G full must not use a scientific limit")
    for cell in manifest["full"]["cells"]:
        root = AUTHORIZED_RUN_ROOT / (
            "full/cell-%02d-window-%s"
            % (cell["index"], cell["window"].replace(":", "-"))
        )
        if (
            Path(cell["output_dir"]).resolve() != root
            or cell["argv"] != _eval_argv(root, cell["window"], None)
            or "--limit" in cell["argv"]
            or cell.get("automatic_retry") is not False
        ):
            raise P3GError("P3-G full cell differs")
    for stage_name in ("smoke", "full"):
        scheduler = manifest[stage_name].get("scheduler", {})
        if (
            scheduler.get("initial_throttle") not in (1, 2, 3, 4)
            or scheduler.get("array")
            != "0-3%%%d" % scheduler.get("initial_throttle")
            or scheduler.get("cpus_per_task") != 8
            or scheduler.get("memory") != "64G"
            or scheduler.get("gres") != "gpu:a40:1"
            or scheduler.get("requeue") is not False
        ):
            raise P3GError("P3-G scheduler contract differs")


def _common_environment(expected_commit: str, manifest_file_sha256: str) -> str:
    return """set -euo pipefail
repo_root={repo}
venv={venv}
run_root={run_root}
[[ "$(git -C "$repo_root" rev-parse --show-toplevel)" == "$repo_root" ]]
[[ "$(git -C "$repo_root" symbolic-ref --short HEAD)" == "loopscope" ]]
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "{commit}" ]]
[[ "$(git -C "$repo_root" rev-parse origin/loopscope)" == "{commit}" ]]
[[ -z "$(git -C "$repo_root" status --porcelain)" ]]
printf "%s  %s\n" "{manifest_sha}" "$run_root/{manifest_name}" | sha256sum -c -
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export UV_CACHE_DIR=/hpc2hdd/home/xhuang225/shared/uv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
unset HF_ENDPOINT || true
export PYTHONPATH="$repo_root/src"
cd "$repo_root"
""".format(
        repo=shlex.quote(str(REMOTE_REPO)),
        venv=shlex.quote(str(AUDITED_VENV)),
        run_root=shlex.quote(str(AUTHORIZED_RUN_ROOT)),
        commit=str(expected_commit),
        manifest_sha=str(manifest_file_sha256),
        manifest_name=MANIFEST_NAME,
    )


def _command(argv: Sequence[str]) -> str:
    return "%s %s" % (
        shlex.quote(str(AUDITED_VENV / "bin/python")),
        shlex.join([str(value) for value in argv[1:]]),
    )


def _smoke_launcher_text(
    cell: Mapping[str, Any], expected_commit: str, manifest_file_sha256: str
) -> str:
    cell_root = Path(str(cell["cell_root"]))
    claim = Path(str(cell_root) + ".claim")
    return (
        "#!/usr/bin/env bash\n"
        + _common_environment(expected_commit, manifest_file_sha256)
        + "cell_root=%s\n" % shlex.quote(str(cell_root))
        + "claim=%s\n" % shlex.quote(str(claim))
        + "if [[ -e \"$cell_root\" || -e \"$claim\" ]]; then\n"
        + "  echo 'refusing to reuse P3-G smoke cell/claim' >&2\n"
        + "  exit 73\n"
        + "fi\n"
        + "mkdir \"$claim\"\n"
        + "mkdir \"$cell_root\"\n"
        + _command(cell["audit_argv"])
        + "\n"
        + "exec "
        + _command(cell["limit_argv"])
        + "\n"
    )


def _full_launcher_text(
    cell: Mapping[str, Any], expected_commit: str, manifest_file_sha256: str
) -> str:
    output_dir = Path(str(cell["output_dir"]))
    claim = Path(str(output_dir) + ".claim")
    return (
        "#!/usr/bin/env bash\n"
        + _common_environment(expected_commit, manifest_file_sha256)
        + "output_dir=%s\n" % shlex.quote(str(output_dir))
        + "claim=%s\n" % shlex.quote(str(claim))
        + "if [[ -e \"$output_dir\" || -e \"$claim\" ]]; then\n"
        + "  echo 'refusing to reuse P3-G full cell/claim' >&2\n"
        + "  exit 73\n"
        + "fi\n"
        + "mkdir \"$claim\"\n"
        + "exec "
        + _command(cell["argv"])
        + "\n"
    )


def _runner_text(stage: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'case "$SLURM_ARRAY_TASK_ID" in',
    ]
    for index in range(4):
        path = AUTHORIZED_RUN_ROOT / ("launch/%s/cell-%02d.sh" % (stage, index))
        lines.append("  %d) exec %s ;;" % (index, shlex.quote(str(path))))
    lines.extend(["  *) echo 'invalid P3-G array task id' >&2; exit 64 ;;", "esac", ""])
    return "\n".join(lines)


def _submit_text(stage: str, scheduler: Mapping[str, Any]) -> str:
    argv = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=loopscope-p3g-%s" % stage,
        "--partition=%s" % scheduler["partition"],
        "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=%d" % scheduler["cpus_per_task"],
        "--mem=%s" % scheduler["memory"],
        "--time=%s" % scheduler["time_limit"],
        "--array=%s" % scheduler["array"],
        "--no-requeue",
        "--output=%s"
        % (AUTHORIZED_RUN_ROOT / ("slurm/%s/loopscope-p3g-%s-%%A_%%a.out" % (stage, stage))),
        "--error=%s"
        % (AUTHORIZED_RUN_ROOT / ("slurm/%s/loopscope-p3g-%s-%%A_%%a.err" % (stage, stage))),
    ]
    if scheduler.get("qos"):
        argv.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        argv.append("--account=%s" % scheduler["account"])
    argv.append(str(AUTHORIZED_RUN_ROOT / ("launch/%s/array_runner.sh" % stage)))
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(argv)


def prepare_run(
    *,
    targeted_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    p3g_root: Path,
    expected_commit: str,
    partition: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    smoke_time_limit: str,
    full_time_limit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    del argv
    assert_authorized_paths(
        targeted_card_path, phase3_card_path, p3c_root, p3g_root
    )
    if Path(p3g_root).exists():
        raise FileExistsError("authorized P3-G write-once run root already exists")
    git_provenance(expected_commit, require_remote_root=True)
    targeted = load_targeted_card(targeted_card_path)
    load_phase3_card(phase3_card_path)
    test_manifest = load_strict_json(Path(p3c_root) / TEST_MANIFEST_NAME)
    verify_manifest_sha256(test_manifest)
    if (
        test_manifest.get("ordered_identity_sha256")
        != targeted["identity"]["gold_free_canonical_ordered_identity_sha256"]
        or test_manifest.get("dataset")
        != {
            "repo": "cais/mmlu",
            "revision": EXPECTED_DATASET_REVISION,
            "split": "test",
        }
        or test_manifest.get("record_count") != 14042
        or test_manifest.get("subject_count") != 57
        or test_manifest.get("outcome_values_read") is not False
    ):
        raise P3GError("P3-G preoutcome canonical test metadata differs")
    manifest = build_manifest(
        run_root=p3g_root,
        expected_commit=expected_commit,
        partition=partition,
        qos=qos,
        account=account,
        array_throttle=array_throttle,
        smoke_time_limit=smoke_time_limit,
        full_time_limit=full_time_limit,
        created_at_utc=utc_now(),
    )
    root = Path(p3g_root)
    root.mkdir()
    for relative in (
        "launch",
        "launch/smoke",
        "launch/full",
        "slurm",
        "slurm/smoke",
        "slurm/full",
        "smoke",
        "full",
    ):
        (root / relative).mkdir()
    manifest_file_sha = write_new_json(root / MANIFEST_NAME, manifest)
    for cell in manifest["smoke"]["cells"]:
        write_new_text(
            root / cell["launcher"],
            _smoke_launcher_text(cell, expected_commit, manifest_file_sha),
            executable=True,
        )
    for cell in manifest["full"]["cells"]:
        write_new_text(
            root / cell["launcher"],
            _full_launcher_text(cell, expected_commit, manifest_file_sha),
            executable=True,
        )
    for stage in ("smoke", "full"):
        write_new_text(
            root / ("launch/%s/array_runner.sh" % stage),
            _runner_text(stage),
            executable=True,
        )
        write_new_text(
            root / ("launch/%s/submit.sh" % stage),
            _submit_text(stage, manifest[stage]["scheduler"]),
            executable=True,
        )
    return manifest


def validate_scheduler_rows(
    rows: Sequence[Mapping[str, Any]], job_id: str
) -> List[Dict[str, str]]:
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise P3GError("P3-G Slurm job id must be decimal")
    task_pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(str(job_id)))
    tasks = {}
    for raw in rows:
        formatted = str(raw.get("JobID", ""))
        match = task_pattern.fullmatch(formatted)
        if not match:
            continue
        index = int(match.group(1))
        if index in tasks:
            raise P3GError("P3-G scheduler rows contain a duplicate array task")
        tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    if set(tasks) != set(range(4)):
        raise P3GError("P3-G seal requires the exact four Slurm array tasks")
    for index, row in tasks.items():
        state = row["State"].split()[0].rstrip("+")
        if state != "COMPLETED" or row["ExitCode"] != "0:0":
            raise P3GError(
                "P3-G task %d is not COMPLETED/0:0: %s %s"
                % (index, row["State"], row["ExitCode"])
            )
    return [tasks[index] for index in range(4)]


def _validate_eval_command(
    command: Mapping[str, Any],
    *,
    window: str,
    output_dir: Path,
    limit: Optional[int],
) -> None:
    exact = {
        "model": "qwen3-1.7b-base",
        "revision": MODEL_REVISION,
        "tasks": "mmlu",
        "num_fewshot": 5,
        "dtype": "float16",
        "limit": limit,
        "first_n": None,
        "batch_size": "auto",
        "k": 2,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "output_dir": str(output_dir),
        "loop": True,
        "window": window,
    }
    for key, expected in exact.items():
        if command.get(key) != expected:
            raise P3GError("P3-G eval command mismatch at %s for %s" % (key, window))


def _smoke_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Mapping):
        raise P3GError("P3-G smoke results lacks inline task sample structure")
    task_names = sorted(
        str(name)
        for name in raw_samples
        if str(name) == "mmlu" or str(name).startswith("mmlu_")
    )
    if len(task_names) != 57:
        raise P3GError("P3-G smoke must contain exactly 57 MMLU tasks")
    identities = []
    for task in task_names:
        rows = raw_samples[task]
        if not isinstance(rows, list) or len(rows) != 5:
            raise P3GError("P3-G smoke must contain exactly five samples per task")
        for row in rows:
            if not isinstance(row, Mapping):
                raise P3GError("P3-G smoke sample structure is malformed")
            doc_id = str(row.get("doc_id", ""))
            doc_hash = str(row.get("doc_hash", ""))
            if not doc_id.isdigit() or str(int(doc_id)) != doc_id:
                raise P3GError("P3-G smoke doc_id is not canonical decimal")
            if not re.fullmatch(r"[0-9a-f]{64}", doc_hash):
                raise P3GError("P3-G smoke doc_hash is invalid")
            identities.append((task, doc_id, doc_hash))
    return {
        "task_count": 57,
        "sample_count": 285,
        "ordered_structural_identity_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _validate_audit_report(payload: Mapping[str, Any], window: str) -> Dict[str, Any]:
    if payload.get("loop_config", {}).get("window") != window:
        raise P3GError("P3-G smoke audit window differs")
    overall = payload.get("overall_decision", {})
    if overall.get("code") != "loop_effective_logits_changed":
        raise P3GError("P3-G smoke audit did not prove loop-effective logits")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise P3GError("P3-G smoke audit has no prompt evidence")
    for prompt in prompts:
        decision = prompt.get("decision", {})
        if (
            decision.get("operator_body_calls") != 2
            or decision.get("restore_allclose") is not True
            or decision.get("code") != "loop_effective_logits_changed"
        ):
            raise P3GError("P3-G smoke audit prompt closure differs")
    return {
        "overall_decision": "loop_effective_logits_changed",
        "prompt_count": len(prompts),
        "operator_body_calls_per_prompt": [2 for _ in prompts],
        "restore_allclose_all_prompts": True,
    }


def verify_smoke(
    *,
    targeted_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    p3g_root: Path,
    expected_commit: str,
    job_id: str,
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(
        targeted_card_path, phase3_card_path, p3c_root, p3g_root
    )
    git = git_provenance(expected_commit, require_remote_root=True)
    load_targeted_card(targeted_card_path)
    load_phase3_card(phase3_card_path)
    root = Path(p3g_root)
    manifest = load_strict_json(root / MANIFEST_NAME)
    validate_manifest(manifest)
    terminal = validate_scheduler_rows(
        list(sacct_rows) if sacct_rows is not None else query_sacct(job_id), job_id
    )
    entries = []
    structural_hashes = []
    for cell, scheduler in zip(manifest["smoke"]["cells"], terminal):
        cell_root = Path(cell["cell_root"])
        audit_path = cell_root / "audit/audit_report.json"
        limit_root = cell_root / "limit"
        command_path = limit_root / "command_args.json"
        revision_path = limit_root / "model_revision.json"
        results_path = limit_root / "results.json"
        for path in (audit_path, command_path, revision_path, results_path):
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise P3GError("P3-G smoke structural artifact is missing/empty/symlinked")
        audit = _validate_audit_report(load_strict_json(audit_path), cell["window"])
        _validate_eval_command(
            load_strict_json(command_path),
            window=cell["window"],
            output_dir=limit_root,
            limit=5,
        )
        _validate_cell_revision(load_strict_json(revision_path))
        structure = _smoke_identity(load_strict_json(results_path))
        structural_hashes.append(structure["ordered_structural_identity_sha256"])
        entries.append(
            {
                "index": cell["index"],
                "window": cell["window"],
                "scheduler": dict(scheduler),
                "audit": audit,
                "task_count": structure["task_count"],
                "sample_count": structure["sample_count"],
                "ordered_structural_identity_sha256": structure[
                    "ordered_structural_identity_sha256"
                ],
                "command_args_file_sha256": file_sha256(command_path),
                "model_revision_file_sha256": file_sha256(revision_path),
                "results_file_sha256": file_sha256(results_path),
            }
        )
    if len(set(structural_hashes)) != 1:
        raise P3GError("P3-G smoke identities differ across candidates")
    return {
        "schema_version": "loopscope.phase3.p3g-smoke-stdout.v1",
        "gate": GATE,
        "git": git,
        "job_id": str(job_id),
        "terminal_completed_0_0": 4,
        "cells": entries,
        "same_57_task_285_sample_structure_all_cells": True,
        "loop_effective_all_cells": True,
        "accuracy_values_accessed": False,
        "accuracy_values_compared": False,
        "status": "STRUCTURAL_SMOKE_PASS",
    }


def _load_test_context(
    phase3_card_path: Path, p3c_root: Path, targeted_card: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    phase3_card = load_phase3_card(phase3_card_path)
    records = validate_test_metadata_records(
        load_strict_jsonl(Path(p3c_root) / TEST_METADATA_NAME),
        phase3_card,
        expected_ordered_identity_sha256=EXPECTED_CANONICAL_IDENTITY_SHA256,
    )
    manifest = load_strict_json(Path(p3c_root) / TEST_MANIFEST_NAME)
    verify_manifest_sha256(manifest)
    if (
        manifest.get("metadata_file_sha256")
        != file_sha256(Path(p3c_root) / TEST_METADATA_NAME)
        or manifest.get("ordered_identity_sha256")
        != targeted_card["identity"]["gold_free_canonical_ordered_identity_sha256"]
        or manifest.get("record_count") != 14042
        or manifest.get("subject_count") != 57
        or manifest.get("dataset")
        != {
            "repo": "cais/mmlu",
            "revision": EXPECTED_DATASET_REVISION,
            "split": "test",
        }
        or manifest.get("outcome_values_read") is not False
    ):
        raise P3GError("P3-G canonical test context differs")
    return phase3_card, records, manifest


def _expected_by_pair_id(
    test_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for record in test_records:
        identity = record["identity"]
        pair_id = "%s:%s" % (identity["task"], identity["doc_id"])
        if pair_id in result:
            raise P3GError("P3-G canonical test context has duplicate evaluator pair ids")
        result[pair_id] = record
    return result


def _inspect_full_identity(
    cell: Mapping[str, Any],
    test_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    output_dir = Path(str(cell["output_dir"])).resolve()
    command_path = output_dir / "command_args.json"
    revision_path = output_dir / "model_revision.json"
    results_path = output_dir / "results.json"
    for path in (command_path, revision_path, results_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise P3GError("P3-G full artifact is missing/empty/symlinked: %s" % path)
    _validate_eval_command(
        load_strict_json(command_path),
        window=str(cell["window"]),
        output_dir=output_dir,
        limit=None,
    )
    _validate_cell_revision(load_strict_json(revision_path))
    payload = load_strict_json(results_path)
    legacy = _phase1_ordered_sample_identity(payload)
    if legacy != {
        "task_count": 57,
        "sample_count": 14042,
        "ordered_identity_sha256": EXPECTED_LEGACY_IDENTITY_SHA256,
    }:
        raise P3GError("P3-G full legacy identity does not close 57/14,042")
    _verify_result_sample_content(payload, _expected_by_pair_id(test_records))
    sidecars = sorted(output_dir.glob("samples_*.jsonl"), key=lambda path: path.name)
    entry = {
        "cell": str(cell["window"]),
        "window": str(cell["window"]),
        "output_dir": str(output_dir),
        "job_id": None,
        "command_args": _file_metadata(command_path),
        "model_revision": _file_metadata(revision_path),
        "results": _file_metadata(results_path),
        "sample_sidecars": [_file_metadata(path) for path in sidecars],
        "task_count": 57,
        "sample_count": 14042,
        "phase1_ordered_sample_identity_sha256": EXPECTED_LEGACY_IDENTITY_SHA256,
        "canonical_test_identity_exact": True,
        "renderer_source_doc_hash_exact": True,
    }
    del payload
    gc.collect()
    return entry


def _sealed_full_sources(
    *,
    manifest: Mapping[str, Any],
    phase3_card_path: Path,
    p3c_root: Path,
    p3g_root: Path,
    targeted_card: Mapping[str, Any],
    job_id: str,
    sacct_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(p3g_root)
    terminal = validate_scheduler_rows(
        list(sacct_rows) if sacct_rows is not None else query_sacct(job_id), job_id
    )
    expected_dirs = {
        Path(cell["output_dir"]).resolve() for cell in manifest["full"]["cells"]
    }
    actual_dirs = {
        path.resolve()
        for path in (root / "full").iterdir()
        if path.is_dir() and not path.name.endswith(".claim")
    }
    if actual_dirs != expected_dirs:
        raise P3GError("P3-G full output directory set has missing/extra cells")
    for cell in manifest["full"]["cells"]:
        for name in ("command_args.json", "model_revision.json", "results.json"):
            path = Path(cell["output_dir"]) / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise P3GError("P3-G 4/4 seal lacks a required non-empty artifact")
    _, test_records, test_manifest = _load_test_context(
        phase3_card_path, p3c_root, targeted_card
    )
    entries = []
    for cell, scheduler in zip(manifest["full"]["cells"], terminal):
        entry = _inspect_full_identity(cell, test_records)
        entry["job_id"] = "%s_%d" % (job_id, cell["index"])
        entry["scheduler"] = dict(scheduler)
        entries.append(entry)
    return terminal, entries, test_records, test_manifest


def _load_comparator_entries(
    p3c_root: Path, targeted_card: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    allowlist = load_strict_json(
        Path(p3c_root) / "historical13_source_allowlist_receipt.json"
    )
    verify_manifest_sha256(allowlist)
    entries = {str(entry.get("cell")): dict(entry) for entry in allowlist.get("cells", ())}
    if not {"baseline", "12:15"}.issubset(entries):
        raise P3GError("P3-G comparator allowlist lacks baseline or 12:15")
    card_comparators = targeted_card["comparators"]
    expected = {
        "baseline": card_comparators["baseline"],
        "12:15": card_comparators["current_best_12_15"],
    }
    for name in COMPARATOR_ORDER:
        entry = entries[name]
        results = entry.get("results", {})
        if (
            Path(str(results.get("path", ""))).resolve()
            != Path(str(expected[name]["results_path"])).resolve()
            or results.get("file_sha256") != expected[name]["results_sha256"]
        ):
            raise P3GError("P3-G comparator path/hash differs for %s" % name)
    return {name: entries[name] for name in COMPARATOR_ORDER}


def _load_correctness_bundle(
    *,
    candidate_entries: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    test_manifest: Mapping[str, Any],
    p3c_root: Path,
    targeted_card: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, Any]]:
    comparator_entries = _load_comparator_entries(p3c_root, targeted_card)
    maps = {}
    summaries = {"comparators": {}, "candidates": {}}
    for name in COMPARATOR_ORDER:
        correctness, summary = _load_outcome_cell(
            comparator_entries[name], test_records, test_manifest
        )
        maps[name] = correctness
        summaries["comparators"][name] = summary
    baseline_order = list(maps["baseline"])
    if list(maps["12:15"]) != baseline_order or len(baseline_order) != 14042:
        raise P3GError("P3-G comparator canonical identity order differs")
    expected_accuracy = {
        "baseline": targeted_card["comparators"]["baseline"]["known_accuracy_percent"],
        "12:15": targeted_card["comparators"]["current_best_12_15"][
            "known_accuracy_percent"
        ],
    }
    for name in COMPARATOR_ORDER:
        observed = sum(maps[name].values()) * 100.0 / 14042.0
        if not math.isclose(
            observed, float(expected_accuracy[name]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise P3GError("P3-G comparator known accuracy mismatch for %s" % name)
    for entry in candidate_entries:
        window = str(entry["cell"])
        correctness, summary = _load_outcome_cell(entry, test_records, test_manifest)
        if list(correctness) != baseline_order:
            raise P3GError("P3-G candidate canonical identity order differs")
        maps[window] = correctness
        summaries["candidates"][window] = summary
    if tuple(window for window in CANDIDATE_WINDOWS if window in maps) != CANDIDATE_WINDOWS:
        raise P3GError("P3-G candidate correctness membership differs")
    return maps, summaries


def holm_step_down_four(
    raw_p_by_window: Mapping[str, float], *, alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    if set(raw_p_by_window) != set(CANDIDATE_WINDOWS):
        raise P3GError("P3-G Holm family must be the exact four candidates")
    order_index = {window: index for index, window in enumerate(CANDIDATE_WINDOWS)}
    ranked = sorted(
        CANDIDATE_WINDOWS,
        key=lambda window: (float(raw_p_by_window[window]), order_index[window]),
    )
    running = 0.0
    result = {}
    for rank, window in enumerate(ranked, start=1):
        raw = float(raw_p_by_window[window])
        if not math.isfinite(raw) or raw < 0.0 or raw > 1.0:
            raise P3GError("P3-G Holm raw p-value is invalid")
        adjusted = max(running, min(1.0, (5 - rank) * raw))
        running = adjusted
        result[window] = {
            "raw_p": raw,
            "adjusted_p": adjusted,
            "significant": adjusted < alpha,
            "holm_rank": rank,
        }
    return {window: result[window] for window in CANDIDATE_WINDOWS}


def joint_paired_bootstrap(
    contrasts: Mapping[str, Sequence[int]],
    *,
    replicates: int,
    seed: int,
    force_python: bool = False,
) -> Dict[str, Any]:
    expected_order = tuple(
        "%s_vs_%s" % (window, comparator)
        for window in CANDIDATE_WINDOWS
        for comparator in COMPARATOR_ORDER
    )
    if tuple(contrasts) != expected_order:
        raise P3GError("P3-G bootstrap contrast order must be exact candidate/card order")
    lengths = {len(values) for values in contrasts.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
        raise P3GError("P3-G bootstrap contrasts must have one positive sample count")
    count = next(iter(lengths))
    rows = []
    for name in expected_order:
        row = []
        for value in contrasts[name]:
            if isinstance(value, bool) or int(value) != value or int(value) not in (-1, 0, 1):
                raise P3GError("P3-G paired difference must be -1, 0, or 1")
            row.append(int(value))
        rows.append(row)
    if not isinstance(replicates, int) or replicates < 1:
        raise P3GError("P3-G bootstrap replicates must be positive")
    rng = random.Random(int(seed))
    stream_hash = hashlib.sha256()
    estimates = {name: [] for name in expected_order}
    numpy_module = None
    if not force_python and count * replicates * len(expected_order) >= 1_000_000:
        try:
            import numpy as numpy_module  # type: ignore
        except ImportError:
            numpy_module = None
    if numpy_module is not None:
        matrix = numpy_module.asarray(rows, dtype=numpy_module.int8)
        for _ in range(replicates):
            indices = [rng.randrange(count) for _ in range(count)]
            stream_hash.update(_index_bytes(indices))
            sampled_counts = numpy_module.bincount(indices, minlength=count)
            points = matrix.dot(sampled_counts) / float(count)
            for name, point in zip(expected_order, points.tolist()):
                estimates[name].append(float(point))
    else:
        for _ in range(replicates):
            indices = [rng.randrange(count) for _ in range(count)]
            stream_hash.update(_index_bytes(indices))
            for name, row in zip(expected_order, rows):
                estimates[name].append(
                    math.fsum(row[index] for index in indices) / float(count)
                )
    result = {}
    for name in expected_order:
        values = estimates[name]
        result[name] = {
            "point_delta_accuracy_fraction": math.fsum(contrasts[name]) / float(count),
            "bootstrap_mean_fraction": math.fsum(values) / float(replicates),
            "percentile_95_ci_fraction": [
                _quantile(values, 0.025),
                _quantile(values, 0.975),
            ],
        }
    return {
        "method": "joint_paired_sample_bootstrap",
        "sample_count": count,
        "replicates": replicates,
        "seed": int(seed),
        "prng": "python_stdlib_random.Random",
        "random_api": "randrange(%d)" % count,
        "iteration_order": "replicate-major then draw-major over canonical test identity order",
        "contrast_order": list(expected_order),
        "index_stream_sha256": stream_hash.hexdigest(),
        "joint_index_reuse_all_eight_contrasts": True,
        "contrasts": result,
    }


def _paired_counts(
    reference: Sequence[bool], candidate: Sequence[bool]
) -> Dict[str, int]:
    if len(reference) != len(candidate) or not reference:
        raise P3GError("P3-G paired correctness vectors differ in length")
    counts = {
        "right_to_right": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "wrong_to_wrong": 0,
    }
    for before, after in zip(reference, candidate):
        if before and after:
            counts["right_to_right"] += 1
        elif not before and after:
            counts["wrong_to_right"] += 1
        elif before and not after:
            counts["right_to_wrong"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    return counts


def classify_targeted_label(best_row: Mapping[str, Any]) -> Tuple[str, Dict[str, bool]]:
    delta = float(best_row["versus_12_15"]["delta_accuracy_fraction"])
    lower = float(best_row["versus_12_15"]["paired_ci95_fraction"][0])
    adjusted = float(best_row["versus_12_15"]["holm_adjusted_mcnemar_p"])
    checks = {
        "best_candidate_point_delta_positive": delta > 0.0,
        "best_candidate_ci95_lower_positive": lower > 0.0,
        "best_candidate_holm_adjusted_mcnemar_below_0_05": adjusted < 0.05,
    }
    if all(checks.values()):
        label = "TARGETED_VARIABLE_WIDTH_STRONG_IMPROVEMENT"
    elif checks["best_candidate_point_delta_positive"]:
        label = "TARGETED_VARIABLE_WIDTH_POINT_IMPROVEMENT"
    else:
        label = "TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT"
    return label, checks


def _resource_accounting(
    terminal: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
    job_id: str,
) -> Dict[str, Any]:
    elapsed = []
    gpu_seconds = 0
    for row in terminal:
        try:
            seconds = int(row["ElapsedRaw"])
        except (KeyError, ValueError) as exc:
            raise P3GError("P3-G Slurm ElapsedRaw is not an integer") from exc
        elapsed.append(seconds)
        match = re.search(r"(?:^|,)gres/gpu=([0-9]+)(?:,|$)", row.get("AllocTRES", ""))
        gpu_seconds += seconds * (int(match.group(1)) if match else 1)
    return {
        "job_id": str(job_id),
        "array": manifest["full"]["scheduler"]["array"],
        "initial_array_throttle": manifest["full"]["scheduler"]["initial_throttle"],
        "throttle_history": [
            {
                "kind": "initial_submission",
                "array": manifest["full"]["scheduler"]["array"],
                "science_manifest_unchanged": True,
            }
        ],
        "terminal_task_count": 4,
        "completed_0_0": 4,
        "elapsed_raw_seconds_sum": sum(elapsed),
        "gpu_hours_allocated_from_elapsed": gpu_seconds / 3600.0,
        "partitions": sorted({row.get("Partition", "") for row in terminal}),
        "nodes": sorted({row.get("NodeList", "") for row in terminal}),
        "alloc_tres": sorted({row.get("AllocTRES", "") for row in terminal}),
        "tasks": [dict(row) for row in terminal],
        "scientific_argv_config_hash_unchanged": True,
    }


def _compute_core(
    maps: Mapping[str, Mapping[str, bool]],
    summaries: Mapping[str, Any],
    candidate_entries: Sequence[Mapping[str, Any]],
    terminal: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
    job_id: str,
) -> Dict[str, Any]:
    identity_order = list(maps["baseline"])
    values = {
        name: [bool(maps[name][key]) for key in identity_order]
        for name in COMPARATOR_ORDER + CANDIDATE_WINDOWS
    }
    contrasts = {}
    for window in CANDIDATE_WINDOWS:
        for comparator in COMPARATOR_ORDER:
            name = "%s_vs_%s" % (window, comparator)
            contrasts[name] = [
                int(candidate) - int(reference)
                for reference, candidate in zip(values[comparator], values[window])
            ]
    bootstrap = joint_paired_bootstrap(
        contrasts, replicates=2000, seed=20260710
    )
    raw_p = {}
    transitions = {}
    accuracies = {}
    for window in CANDIDATE_WINDOWS:
        counts = _paired_counts(values["12:15"], values[window])
        transitions[window] = counts
        raw_p[window] = exact_mcnemar_p(
            counts["wrong_to_right"], counts["right_to_wrong"]
        )
        accuracies[window] = sum(values[window]) / 14042.0
    holm = holm_step_down_four(raw_p)
    card_order = {window: index for index, window in enumerate(CANDIDATE_WINDOWS)}
    ranking = sorted(
        CANDIDATE_WINDOWS,
        key=lambda window: (-accuracies[window], card_order[window]),
    )
    oracle_accuracy = accuracies[ranking[0]]
    oracle_windows = [
        window for window in CANDIDATE_WINDOWS if accuracies[window] == oracle_accuracy
    ]
    rows = []
    for window in CANDIDATE_WINDOWS:
        comparisons = {}
        for comparator in COMPARATOR_ORDER:
            contrast = bootstrap["contrasts"]["%s_vs_%s" % (window, comparator)]
            interval = contrast["percentile_95_ci_fraction"]
            comparisons[comparator] = {
                "delta_accuracy_fraction": contrast["point_delta_accuracy_fraction"],
                "delta_accuracy_pp": contrast["point_delta_accuracy_fraction"] * 100.0,
                "paired_ci95_fraction": list(interval),
                "paired_ci95_pp": [float(value) * 100.0 for value in interval],
            }
        rows.append(
            {
                "window": window,
                "card_order": card_order[window] + 1,
                "accuracy_fraction": accuracies[window],
                "accuracy_percent": accuracies[window] * 100.0,
                "observed_rank": ranking.index(window) + 1,
                "regret_to_observed_candidate_oracle_fraction": (
                    oracle_accuracy - accuracies[window]
                ),
                "regret_to_observed_candidate_oracle_pp": (
                    oracle_accuracy - accuracies[window]
                )
                * 100.0,
                "versus_baseline": comparisons["baseline"],
                "versus_12_15": dict(
                    comparisons["12:15"],
                    paired_flip_counts=transitions[window],
                    exact_mcnemar_p=raw_p[window],
                    holm_adjusted_mcnemar_p=holm[window]["adjusted_p"],
                    holm_significant=holm[window]["significant"],
                    holm_rank=holm[window]["holm_rank"],
                ),
                "source": summaries["candidates"][window],
            }
        )
    by_window = {row["window"]: row for row in rows}
    label, label_checks = classify_targeted_label(by_window[ranking[0]])
    candidate_source = {
        str(entry["cell"]): {
            "output_dir": entry["output_dir"],
            "results_file_sha256": entry["results"]["file_sha256"],
            "sample_sidecar_file_sha256": {
                Path(value["path"]).name: value["file_sha256"]
                for value in entry["sample_sidecars"]
            },
        }
        for entry in candidate_entries
    }
    return {
        "sample_count": 14042,
        "subject_count": 57,
        "identity_closure": {
            "phase1_legacy_ordered_identity_sha256": EXPECTED_LEGACY_IDENTITY_SHA256,
            "canonical_ordered_identity_sha256": EXPECTED_CANONICAL_IDENTITY_SHA256,
            "all_four_candidates_and_both_comparators_exact": True,
            "identity_intersection_fallback_used": False,
        },
        "comparators": {
            name: {
                "accuracy_fraction": sum(values[name]) / 14042.0,
                "accuracy_percent": sum(values[name]) * 100.0 / 14042.0,
                "source": summaries["comparators"][name],
            }
            for name in COMPARATOR_ORDER
        },
        "candidate_sources": candidate_source,
        "bootstrap": bootstrap,
        "candidates": rows,
        "observed_candidate_ranking": ranking,
        "observed_candidate_oracle": {
            "windows": oracle_windows,
            "unique": len(oracle_windows) == 1,
            "accuracy_fraction": oracle_accuracy,
            "accuracy_percent": oracle_accuracy * 100.0,
        },
        "holm_family": {
            "method": "Holm step-down",
            "family": ["%s_vs_12:15" % window for window in CANDIDATE_WINDOWS],
            "family_size": 4,
            "familywise_alpha": 0.05,
        },
        "frozen_science_label": label,
        "frozen_label_checks": label_checks,
        "best_candidate": ranking[0],
        "resource_accounting": _resource_accounting(terminal, manifest, job_id),
        "full_seal": {
            "job_id": str(job_id),
            "terminal_completed_0_0": 4,
            "task_count_per_cell": 57,
            "sample_count_per_cell": 14042,
            "recipe_revision_renderer_closure": True,
            "partial_outcome_viewing": False,
            "all_identity_checks_completed_before_first_correctness_access": True,
        },
    }


def analyze_full(
    *,
    targeted_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    p3g_root: Path,
    expected_commit: str,
    job_id: str,
    argv: Sequence[str],
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(
        targeted_card_path, phase3_card_path, p3c_root, p3g_root
    )
    git = git_provenance(expected_commit, require_remote_root=True)
    root = Path(p3g_root)
    if (root / ANALYSIS_NAME).exists() or (root / VERIFIER_NAME).exists():
        raise FileExistsError("P3-G analysis/verifier is write-once")
    targeted = load_targeted_card(targeted_card_path)
    manifest = load_strict_json(root / MANIFEST_NAME)
    validate_manifest(manifest)
    if manifest["git"]["implementation_commit"] != expected_commit:
        raise P3GError("P3-G manifest implementation commit differs")
    terminal, entries, test_records, test_manifest = _sealed_full_sources(
        manifest=manifest,
        phase3_card_path=phase3_card_path,
        p3c_root=p3c_root,
        p3g_root=p3g_root,
        targeted_card=targeted,
        job_id=job_id,
        sacct_rows=sacct_rows,
    )
    maps, summaries = _load_correctness_bundle(
        candidate_entries=entries,
        test_records=test_records,
        test_manifest=test_manifest,
        p3c_root=p3c_root,
        targeted_card=targeted,
    )
    core = _compute_core(maps, summaries, entries, terminal, manifest, job_id)
    analysis: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3g-targeted-full-analysis.v1",
        "artifact_role": "p3g_one_shot_four_candidate_paired_analysis",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
        "targeted_card_file_sha256": TARGETED_CARD_SHA256,
        "phase3_card_file_sha256": PHASE3_CARD_BYTE_SHA256,
        "p3g_manifest_file_sha256": file_sha256(root / MANIFEST_NAME),
        "p3g_manifest_internal_sha256": manifest["manifest_sha256"],
        "analysis_run_count": 1,
        "analysis_rerun_authorized": False,
        "result": core,
        "status": "TARGETED_FULL_ANALYSIS_COMPLETE",
    }
    attach_manifest_sha256(analysis)
    write_new_json(root / ANALYSIS_NAME, analysis)
    del maps
    gc.collect()
    return analysis


def verify_full(
    *,
    targeted_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    p3g_root: Path,
    expected_commit: str,
    job_id: str,
    argv: Sequence[str],
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(
        targeted_card_path, phase3_card_path, p3c_root, p3g_root
    )
    git = git_provenance(expected_commit, require_remote_root=True)
    root = Path(p3g_root)
    if (root / VERIFIER_NAME).exists():
        raise FileExistsError("P3-G verifier is write-once")
    analysis = load_strict_json(root / ANALYSIS_NAME)
    verify_manifest_sha256(analysis)
    if (
        analysis.get("analysis_run_count") != 1
        or analysis.get("analysis_rerun_authorized") is not False
        or analysis.get("status") != "TARGETED_FULL_ANALYSIS_COMPLETE"
    ):
        raise P3GError("P3-G analysis one-shot contract differs")
    targeted = load_targeted_card(targeted_card_path)
    manifest = load_strict_json(root / MANIFEST_NAME)
    validate_manifest(manifest)
    terminal, entries, test_records, test_manifest = _sealed_full_sources(
        manifest=manifest,
        phase3_card_path=phase3_card_path,
        p3c_root=p3c_root,
        p3g_root=p3g_root,
        targeted_card=targeted,
        job_id=job_id,
        sacct_rows=sacct_rows,
    )
    maps, summaries = _load_correctness_bundle(
        candidate_entries=entries,
        test_records=test_records,
        test_manifest=test_manifest,
        p3c_root=p3c_root,
        targeted_card=targeted,
    )
    recomputed = _compute_core(maps, summaries, entries, terminal, manifest, job_id)
    if canonical_json_bytes(recomputed) != canonical_json_bytes(analysis.get("result")):
        raise P3GError("P3-G verifier recomputation differs from analysis")
    verifier: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3g-verifier.v1",
        "artifact_role": "p3g_independent_source_reload_and_statistical_recompute",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "argv": list(argv),
        "analysis_file_sha256": file_sha256(root / ANALYSIS_NAME),
        "analysis_internal_sha256": analysis["manifest_sha256"],
        "p3g_manifest_file_sha256": file_sha256(root / MANIFEST_NAME),
        "p3g_manifest_internal_sha256": manifest["manifest_sha256"],
        "verification": {
            "full_4_of_4_terminal_rechecked": True,
            "all_six_identity_maps_reloaded_exact": True,
            "candidate_accuracy_recomputed": True,
            "eight_joint_paired_bootstrap_contrasts_recomputed": True,
            "bootstrap_index_stream_sha256": recomputed["bootstrap"][
                "index_stream_sha256"
            ],
            "paired_flip_counts_recomputed": True,
            "four_exact_mcnemar_recomputed": True,
            "four_test_holm_recomputed": True,
            "observed_ranking_and_regret_recomputed": True,
            "frozen_label_recomputed": True,
            "frozen_science_label": recomputed["frozen_science_label"],
        },
        "implementation_sha256": implementation_hashes(),
        "status": "PASS",
    }
    attach_manifest_sha256(verifier)
    write_new_json(root / VERIFIER_NAME, verifier)
    del maps
    gc.collect()
    return verifier


__all__ = [
    "ANALYSIS_NAME",
    "AUTHORIZED_RUN_ROOT",
    "CANDIDATE_WINDOWS",
    "MANIFEST_NAME",
    "P3GError",
    "VERIFIER_NAME",
    "analyze_full",
    "build_manifest",
    "classify_targeted_label",
    "holm_step_down_four",
    "joint_paired_bootstrap",
    "prepare_run",
    "validate_manifest",
    "validate_scheduler_rows",
    "verify_full",
    "verify_smoke",
]
