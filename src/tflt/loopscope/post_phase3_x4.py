"""Post-Phase3 X4: repeat decoder layer 15 at fixed-horizon K=2/3/4.

This sidecar reuses the audited Phase 3 evaluation, identity, and paired-outcome
helpers without changing the loop implementation or evaluator.  It freezes one
three-cell limit=5 structural smoke and one three-cell full array.  Full outcome
values remain sealed until all three tasks are COMPLETED/0:0 and every cell has
closed the exact model, recipe, and 14,042-sample identity contract.
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


GATE = "POST-P3-X4"
EXECUTOR_THREAD_ID = "019f6bb3-8f65-7ea1-a2f9-425a7959680f"
PLANNING_THREAD_ID = "019f670d-24ca-7ad0-b92e-64438aa04efc1"
AUTHORIZED_BASE_COMMIT = "684aff7a58bd62b6dd893e24689f3bfb4833f075"
FIXED_ANCESTOR_COMMIT = "4f59bd93eca4da3cbf458a93508f91c5b23912bc"

REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-posthoc-x4-layer15-k234-20260717T001249Z"
)
P3C_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3c-20260716T064601Z"
)
X4_CARD_PATH = REMOTE_REPO / "configs/loopscope/post_phase3_x4_layer15_k234_card.json"
PHASE3_CARD_PATH = REMOTE_REPO / "configs/loopscope/phase3_card.json"

EXPECTED_LEGACY_IDENTITY_SHA256 = (
    "872c9f6bd7f40e3c6fa40b13cc862033379ac0a9cbc439a1593653945f3d521d"
)
EXPECTED_CANONICAL_IDENTITY_SHA256 = (
    "2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532"
)
EXPECTED_DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
BASELINE_RESULTS_SHA256 = (
    "b390880ae8f85d655c86b7b81a138bc12b99b61cf26d22520a8d372bdeafcc25"
)
CONTEXT_RESULTS_SHA256 = (
    "bbcb5940fbefd9ec24402fd3269cf31763c13fa73661c1137c16f372544036d1"
)

WINDOW = "15:15"
K_ORDER = (2, 3, 4)
K_LABELS = tuple("K%d" % value for value in K_ORDER)
COMPARATOR_ORDER = ("baseline", "12:15@K2")

MANIFEST_NAME = "x4_manifest.json"
ANALYSIS_NAME = "x4_layer15_k234_analysis.json"
VERIFIER_NAME = "x4_layer15_k234_verifier.json"

IMPLEMENTATION_RELATIVE_PATHS = (
    "configs/loopscope/post_phase3_x4_layer15_k234_card.json",
    "src/tflt/loopscope/post_phase3_x4.py",
    "scripts/loopscope/run_qwen17_post_phase3_x4.py",
    "tests/test_loopscope_post_phase3_x4.py",
)


class X4Error(ValueError):
    """Fail-closed X4 contract, provenance, seal, or statistics violation."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise X4Error("X4 implementation path is missing: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def git_provenance(expected_commit: str, *, require_remote_root: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if require_remote_root and root != REMOTE_REPO:
        raise X4Error("X4 remote action is outside the dedicated clone")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "refs/remotes/origin/loopscope")
    dirty = bool(_git(root, "status", "--porcelain"))
    if branch != "loopscope":
        raise X4Error("X4 requires branch loopscope")
    if head != str(expected_commit) or origin != str(expected_commit):
        raise X4Error("X4 HEAD/origin differs from expected implementation commit")
    if dirty:
        raise X4Error("X4 requires a clean working tree")
    ancestor = _git(root, "merge-base", "--is-ancestor", FIXED_ANCESTOR_COMMIT, head)
    if ancestor:
        raise X4Error("unexpected merge-base output")
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
        raise X4Error("X4 implementation commit changed paths outside exact authorization")
    return {
        "root": str(root),
        "branch": branch,
        "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
        "fixed_ancestor_commit": FIXED_ANCESTOR_COMMIT,
        "commit": head,
        "origin_loopscope": origin,
        "dirty": False,
        "changed_paths": list(changed),
    }


def assert_authorized_paths(
    x4_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    x4_root: Path,
) -> None:
    exact = {
        Path(x4_card_path).resolve(): X4_CARD_PATH,
        Path(phase3_card_path).resolve(): PHASE3_CARD_PATH,
        Path(p3c_root).resolve(): P3C_ROOT,
        Path(x4_root).resolve(): AUTHORIZED_RUN_ROOT,
    }
    for actual, expected in exact.items():
        if actual != expected:
            raise X4Error("X4 path differs from exact authorization: %s" % actual)


def load_x4_card(path: Path) -> Dict[str, Any]:
    card = load_strict_json(path)
    if (
        card.get("schema_version")
        != "loopscope.post_phase3.x4_layer15_k234_card.v1"
        or card.get("card") != "POST_PHASE3_X4_LAYER15_K234_V1"
        or card.get("status") != "PLANNING_HANDOFF_MATERIALIZED"
        or card.get("authorized_executor_thread") != EXECUTOR_THREAD_ID
        or card.get("planning_audit_thread") != PLANNING_THREAD_ID
        or card.get("authorized_base_commit") != AUTHORIZED_BASE_COMMIT
        or card.get("fixed_ancestor_commit") != FIXED_ANCESTOR_COMMIT
    ):
        raise X4Error("X4 card identity or authority differs")
    cells = card.get("frozen_cells", {})
    if (
        cells.get("window") != WINDOW
        or cells.get("single_decoder_layer_body") is not True
        or tuple(cells.get("ordered_k", ())) != K_ORDER
        or cells.get("cell_count") != 3
        or cells.get("no_substitution_reordering_or_addition") is not True
    ):
        raise X4Error("X4 frozen cell matrix differs")
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
        "fixed_horizon": True,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "step_size_rule": "alpha_divided_by_k",
        "total_horizon": 1.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "only_k_may_vary": True,
        "full_limit": None,
    }
    if any(recipe.get(key) != value for key, value in exact_recipe.items()):
        raise X4Error("X4 frozen scientific recipe differs")
    identity = card.get("identity", {})
    if (
        identity.get("ordered_sample_count") != 14042
        or identity.get("subject_count") != 57
        or identity.get("phase1_legacy_ordered_identity_sha256")
        != EXPECTED_LEGACY_IDENTITY_SHA256
        or identity.get("gold_free_canonical_ordered_identity_sha256")
        != EXPECTED_CANONICAL_IDENTITY_SHA256
    ):
        raise X4Error("X4 identity contract differs")
    comparators = card.get("comparators", {})
    if (
        comparators.get("baseline", {}).get("results_sha256")
        != BASELINE_RESULTS_SHA256
        or comparators.get("context_12_15_k2", {}).get("results_sha256")
        != CONTEXT_RESULTS_SHA256
        or comparators.get("rerun_comparators") is not False
    ):
        raise X4Error("X4 comparator contract differs")
    execution = card.get("execution", {})
    if (
        Path(str(execution.get("repo", ""))).resolve() != REMOTE_REPO
        or Path(str(execution.get("venv", ""))).resolve() != AUDITED_VENV
        or Path(str(execution.get("run_root", ""))).resolve() != AUTHORIZED_RUN_ROOT
        or execution.get("cache_policy")
        != "offline existing caches only; no download or mutation"
    ):
        raise X4Error("X4 execution paths or cache policy differ")
    bootstrap = card.get("analysis", {}).get("bootstrap", {})
    if (
        bootstrap.get("method") != "joint_paired_sample_bootstrap"
        or bootstrap.get("replicates") != 2000
        or bootstrap.get("seed") != 20260710
        or bootstrap.get("prng") != "python_stdlib_random.Random"
        or bootstrap.get("random_api") != "randrange(14042)"
    ):
        raise X4Error("X4 bootstrap contract differs")
    return card


def _eval_argv(output_dir: Path, k: int, limit: Optional[int]) -> List[str]:
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
            WINDOW,
            "--k",
            str(k),
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


def _audit_argv(output_dir: Path, k: int) -> List[str]:
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
        WINDOW,
        "--k",
        str(k),
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
    card_file_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise X4Error("X4 manifest run root differs from authorization")
    if not partition or not smoke_time_limit or not full_time_limit:
        raise X4Error("X4 scheduler partition/time limits must be explicit")
    if (
        isinstance(array_throttle, bool)
        or not isinstance(array_throttle, int)
        or array_throttle < 1
        or array_throttle > 3
    ):
        raise X4Error("X4 array throttle must be an integer in 1..3")
    smoke_cells = []
    full_cells = []
    for index, k in enumerate(K_ORDER):
        token = "k%d" % k
        smoke_root = Path(run_root) / (
            "smoke/cell-%02d-window-15-15-%s" % (index, token)
        )
        full_root = Path(run_root) / (
            "full/cell-%02d-window-15-15-%s" % (index, token)
        )
        smoke_cells.append(
            {
                "index": index,
                "label": "K%d" % k,
                "window": WINDOW,
                "k": k,
                "step_size": 1.0 / float(k),
                "cell_root": str(smoke_root),
                "audit_output_dir": str(smoke_root / "audit"),
                "limit_output_dir": str(smoke_root / "limit"),
                "launcher": "launch/smoke/cell-%02d.sh" % index,
                "audit_argv": _audit_argv(smoke_root / "audit", k),
                "limit_argv": _eval_argv(smoke_root / "limit", k, 5),
                "accuracy_values_may_not_be_accessed": True,
            }
        )
        full_cells.append(
            {
                "index": index,
                "label": "K%d" % k,
                "window": WINDOW,
                "k": k,
                "step_size": 1.0 / float(k),
                "output_dir": str(full_root),
                "launcher": "launch/full/cell-%02d.sh" % index,
                "argv": _eval_argv(full_root, k, None),
                "automatic_retry": False,
            }
        )
    scheduler_common = {
        "partition": partition,
        "qos": qos,
        "account": account,
        "array": "0-2%%%d" % array_throttle,
        "initial_throttle": array_throttle,
        "cpus_per_task": 8,
        "memory": "64G",
        "gres": "gpu:a40:1",
        "requeue": False,
    }
    manifest: Dict[str, Any] = {
        "schema_version": "loopscope.post_phase3.x4-manifest.v1",
        "artifact_role": "x4_exact_layer15_k234_smoke_and_full_preoutcome_freeze",
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
            "x4_card_file_sha256": card_file_sha256
            or file_sha256(repository_root() / IMPLEMENTATION_RELATIVE_PATHS[0]),
            "phase3_card_file_sha256": PHASE3_CARD_BYTE_SHA256,
            "window": WINDOW,
            "k_order": list(K_ORDER),
            "baseline_results_sha256": BASELINE_RESULTS_SHA256,
            "context_12_15_results_sha256": CONTEXT_RESULTS_SHA256,
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
            "window": WINDOW,
            "single_decoder_layer_body": True,
            "k_order": list(K_ORDER),
            "fixed_horizon": True,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "step_size_rule": "alpha_divided_by_k",
            "total_horizon": 1.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "only_k_varies": True,
        },
        "smoke": {
            "cell_count": 3,
            "k_order": list(K_ORDER),
            "limit": 5,
            "cells": smoke_cells,
            "scheduler": dict(scheduler_common, time_limit=smoke_time_limit),
            "inspection_scope": [
                "terminal_state",
                "recipe",
                "task_and_sample_structure",
                "single_layer_loop_effect",
            ],
            "accuracy_values_consumed": False,
        },
        "full": {
            "cell_count": 3,
            "k_order": list(K_ORDER),
            "limit": None,
            "cells": full_cells,
            "scheduler": dict(scheduler_common, time_limit=full_time_limit),
            "sealed_completion": {
                "required_terminal_tasks": 3,
                "required_state": "COMPLETED",
                "required_exit_code": "0:0",
                "required_tasks_per_cell": 57,
                "required_samples_per_cell": 14042,
                "partial_outcome_viewing": False,
            },
            "automatic_retry": False,
        },
        "analysis": {
            "comparators": list(COMPARATOR_ORDER),
            "bootstrap_method": "joint_paired_sample_bootstrap",
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20260710,
            "joint_index_reuse_all_nine_contrasts": True,
            "holm_family": ["K2_vs_baseline", "K3_vs_baseline", "K4_vs_baseline"],
            "holm_family_size": 3,
            "holm_familywise_alpha": 0.05,
            "k_rank_order": ["accuracy_descending", "k_ascending"],
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
        manifest.get("schema_version") != "loopscope.post_phase3.x4-manifest.v1"
        or manifest.get("gate") != GATE
        or manifest.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or manifest.get("planning_thread_id") != PLANNING_THREAD_ID
        or Path(str(manifest.get("run_root", ""))).resolve() != AUTHORIZED_RUN_ROOT
        or manifest.get("outcome_values_consumed") is not False
        or tuple(manifest.get("required_summary_artifacts", ()))
        != (MANIFEST_NAME, ANALYSIS_NAME, VERIFIER_NAME)
    ):
        raise X4Error("X4 manifest identity/summary contract differs")
    if set(manifest.get("implementation_sha256", {})) != set(
        IMPLEMENTATION_RELATIVE_PATHS
    ):
        raise X4Error("X4 implementation hash set differs")
    if manifest.get("recipe") != {
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
        "window": WINDOW,
        "single_decoder_layer_body": True,
        "k_order": list(K_ORDER),
        "fixed_horizon": True,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "step_size_rule": "alpha_divided_by_k",
        "total_horizon": 1.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "only_k_varies": True,
    }:
        raise X4Error("X4 manifest recipe differs")
    for stage_name in ("smoke", "full"):
        stage = manifest.get(stage_name, {})
        cells = stage.get("cells")
        if (
            stage.get("cell_count") != 3
            or tuple(stage.get("k_order", ())) != K_ORDER
            or not isinstance(cells, list)
            or len(cells) != 3
            or [cell.get("index") for cell in cells] != list(range(3))
            or tuple(cell.get("k") for cell in cells) != K_ORDER
            or any(cell.get("window") != WINDOW for cell in cells)
        ):
            raise X4Error("X4 %s matrix differs from exact K2/K3/K4 cells" % stage_name)
    if manifest["smoke"].get("limit") != 5:
        raise X4Error("X4 smoke must remain limit=5")
    for cell in manifest["smoke"]["cells"]:
        root = AUTHORIZED_RUN_ROOT / (
            "smoke/cell-%02d-window-15-15-k%d" % (cell["index"], cell["k"])
        )
        if (
            Path(cell["cell_root"]).resolve() != root
            or cell["audit_argv"] != _audit_argv(root / "audit", cell["k"])
            or cell["limit_argv"] != _eval_argv(root / "limit", cell["k"], 5)
            or cell.get("accuracy_values_may_not_be_accessed") is not True
        ):
            raise X4Error("X4 smoke cell differs")
    if manifest["full"].get("limit") is not None:
        raise X4Error("X4 full must not use a scientific limit")
    for cell in manifest["full"]["cells"]:
        root = AUTHORIZED_RUN_ROOT / (
            "full/cell-%02d-window-15-15-k%d" % (cell["index"], cell["k"])
        )
        if (
            Path(cell["output_dir"]).resolve() != root
            or cell["argv"] != _eval_argv(root, cell["k"], None)
            or "--limit" in cell["argv"]
            or cell.get("automatic_retry") is not False
        ):
            raise X4Error("X4 full cell differs")
    for stage_name in ("smoke", "full"):
        scheduler = manifest[stage_name].get("scheduler", {})
        if (
            scheduler.get("initial_throttle") not in (1, 2, 3)
            or scheduler.get("array")
            != "0-2%%%d" % scheduler.get("initial_throttle")
            or scheduler.get("cpus_per_task") != 8
            or scheduler.get("memory") != "64G"
            or scheduler.get("gres") != "gpu:a40:1"
            or scheduler.get("requeue") is not False
        ):
            raise X4Error("X4 scheduler contract differs")


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
        + "  echo 'refusing to reuse X4 smoke cell/claim' >&2\n"
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
        + "  echo 'refusing to reuse X4 full cell/claim' >&2\n"
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
    for index in range(3):
        path = AUTHORIZED_RUN_ROOT / ("launch/%s/cell-%02d.sh" % (stage, index))
        lines.append("  %d) exec %s ;;" % (index, shlex.quote(str(path))))
    lines.extend(["  *) echo 'invalid X4 array task id' >&2; exit 64 ;;", "esac", ""])
    return "\n".join(lines)


def _submit_text(stage: str, scheduler: Mapping[str, Any]) -> str:
    argv = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=loopscope-x4-%s" % stage,
        "--partition=%s" % scheduler["partition"],
        "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=%d" % scheduler["cpus_per_task"],
        "--mem=%s" % scheduler["memory"],
        "--time=%s" % scheduler["time_limit"],
        "--array=%s" % scheduler["array"],
        "--no-requeue",
        "--output=%s"
        % (AUTHORIZED_RUN_ROOT / ("slurm/%s/loopscope-x4-%s-%%A_%%a.out" % (stage, stage))),
        "--error=%s"
        % (AUTHORIZED_RUN_ROOT / ("slurm/%s/loopscope-x4-%s-%%A_%%a.err" % (stage, stage))),
    ]
    if scheduler.get("qos"):
        argv.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        argv.append("--account=%s" % scheduler["account"])
    argv.append(str(AUTHORIZED_RUN_ROOT / ("launch/%s/array_runner.sh" % stage)))
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(argv)


def prepare_run(
    *,
    x4_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    x4_root: Path,
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
    assert_authorized_paths(x4_card_path, phase3_card_path, p3c_root, x4_root)
    if Path(x4_root).exists():
        raise FileExistsError("authorized X4 write-once run root already exists")
    git_provenance(expected_commit, require_remote_root=True)
    card = load_x4_card(x4_card_path)
    load_phase3_card(phase3_card_path)
    test_manifest = load_strict_json(Path(p3c_root) / TEST_MANIFEST_NAME)
    verify_manifest_sha256(test_manifest)
    if (
        test_manifest.get("ordered_identity_sha256")
        != card["identity"]["gold_free_canonical_ordered_identity_sha256"]
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
        raise X4Error("X4 preoutcome canonical test metadata differs")
    manifest = build_manifest(
        run_root=x4_root,
        expected_commit=expected_commit,
        partition=partition,
        qos=qos,
        account=account,
        array_throttle=array_throttle,
        smoke_time_limit=smoke_time_limit,
        full_time_limit=full_time_limit,
        created_at_utc=utc_now(),
        card_file_sha256=file_sha256(x4_card_path),
    )
    root = Path(x4_root)
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
        raise X4Error("X4 Slurm job id must be decimal")
    task_pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(str(job_id)))
    tasks = {}
    for raw in rows:
        formatted = str(raw.get("JobID", ""))
        match = task_pattern.fullmatch(formatted)
        if not match:
            continue
        index = int(match.group(1))
        if index in tasks:
            raise X4Error("X4 scheduler rows contain a duplicate array task")
        tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    if set(tasks) != set(range(3)):
        raise X4Error("X4 seal requires the exact three Slurm array tasks")
    for index, row in tasks.items():
        state = row["State"].split()[0].rstrip("+")
        if state != "COMPLETED" or row["ExitCode"] != "0:0":
            raise X4Error(
                "X4 task %d is not COMPLETED/0:0: %s %s"
                % (index, row["State"], row["ExitCode"])
            )
    return [tasks[index] for index in range(3)]


def _validate_eval_command(
    command: Mapping[str, Any],
    *,
    k: int,
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
        "k": k,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "output_dir": str(output_dir),
        "loop": True,
        "window": WINDOW,
    }
    for key, expected in exact.items():
        if command.get(key) != expected:
            raise X4Error("X4 eval command mismatch at %s for K%d" % (key, k))


def _smoke_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Mapping):
        raise X4Error("X4 smoke results lacks inline task sample structure")
    task_names = sorted(
        str(name)
        for name in raw_samples
        if str(name) == "mmlu" or str(name).startswith("mmlu_")
    )
    if len(task_names) != 57:
        raise X4Error("X4 smoke must contain exactly 57 MMLU tasks")
    identities = []
    for task in task_names:
        rows = raw_samples[task]
        if not isinstance(rows, list) or len(rows) != 5:
            raise X4Error("X4 smoke must contain exactly five samples per task")
        for row in rows:
            if not isinstance(row, Mapping):
                raise X4Error("X4 smoke sample structure is malformed")
            doc_id = str(row.get("doc_id", ""))
            doc_hash = str(row.get("doc_hash", ""))
            if not doc_id.isdigit() or str(int(doc_id)) != doc_id:
                raise X4Error("X4 smoke doc_id is not canonical decimal")
            if not re.fullmatch(r"[0-9a-f]{64}", doc_hash):
                raise X4Error("X4 smoke doc_hash is invalid")
            identities.append((task, doc_id, doc_hash))
    return {
        "task_count": 57,
        "sample_count": 285,
        "ordered_structural_identity_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _validate_audit_report(
    payload: Mapping[str, Any], k: int
) -> Dict[str, Any]:
    config = payload.get("loop_config", {})
    patch = payload.get("patch_target", {})
    if (
        config.get("window") != WINDOW
        or config.get("k") != k
        or config.get("iteration_mode") != "block"
        or config.get("strategy") != "damped_euler"
        or config.get("alpha") != 1.0
        or config.get("beta") != 0.0
        or config.get("cache_strategy") != "last"
        or config.get("decode_mode") != "bypass"
        or patch.get("window") != [15, 15]
    ):
        raise X4Error("X4 smoke audit recipe or single-layer target differs")
    classes_after = patch.get("classes_after_patch")
    if not isinstance(classes_after, Mapping) or set(classes_after) != {"15"}:
        raise X4Error("X4 smoke audit did not patch exactly decoder layer 15")
    overall = payload.get("overall_decision", {})
    if overall.get("code") != "loop_effective_logits_changed":
        raise X4Error("X4 smoke audit did not prove loop-effective logits")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise X4Error("X4 smoke audit has no prompt evidence")
    max_logits = []
    for prompt in prompts:
        decision = prompt.get("decision", {})
        if (
            decision.get("operator_body_calls") != k
            or decision.get("restore_allclose") is not True
            or decision.get("code") != "loop_effective_logits_changed"
            or float(decision.get("max_final_logits_diff", 0.0)) <= 1e-12
        ):
            raise X4Error("X4 smoke audit prompt closure differs for K%d" % k)
        max_logits.append(float(decision["max_final_logits_diff"]))
    return {
        "overall_decision": "loop_effective_logits_changed",
        "prompt_count": len(prompts),
        "single_decoder_layer_body": 15,
        "operator_body_calls_per_prompt": [k for _ in prompts],
        "max_final_logits_diff_per_prompt": max_logits,
        "restore_allclose_all_prompts": True,
    }


def verify_smoke(
    *,
    x4_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    x4_root: Path,
    expected_commit: str,
    job_id: str,
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(x4_card_path, phase3_card_path, p3c_root, x4_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    load_x4_card(x4_card_path)
    load_phase3_card(phase3_card_path)
    root = Path(x4_root)
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
                raise X4Error("X4 smoke structural artifact is missing/empty/symlinked")
        audit = _validate_audit_report(load_strict_json(audit_path), cell["k"])
        _validate_eval_command(
            load_strict_json(command_path),
            k=cell["k"],
            output_dir=limit_root,
            limit=5,
        )
        _validate_cell_revision(load_strict_json(revision_path))
        structure = _smoke_identity(load_strict_json(results_path))
        structural_hashes.append(structure["ordered_structural_identity_sha256"])
        entries.append(
            {
                "index": cell["index"],
                "label": cell["label"],
                "window": WINDOW,
                "k": cell["k"],
                "step_size": cell["step_size"],
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
        raise X4Error("X4 smoke identities differ across K cells")
    return {
        "schema_version": "loopscope.post_phase3.x4-smoke-stdout.v1",
        "gate": GATE,
        "git": git,
        "job_id": str(job_id),
        "terminal_completed_0_0": 3,
        "cells": entries,
        "same_57_task_285_sample_structure_all_cells": True,
        "single_layer15_loop_effective_all_cells": True,
        "operator_body_calls_equals_k_all_prompts": True,
        "accuracy_values_accessed": False,
        "accuracy_values_compared": False,
        "status": "STRUCTURAL_SMOKE_PASS",
    }


def _load_test_context(
    phase3_card_path: Path, p3c_root: Path, x4_card: Mapping[str, Any]
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
        != x4_card["identity"]["gold_free_canonical_ordered_identity_sha256"]
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
        raise X4Error("X4 canonical test context differs")
    return phase3_card, records, manifest


def _expected_by_pair_id(
    test_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for record in test_records:
        identity = record["identity"]
        pair_id = "%s:%s" % (identity["task"], identity["doc_id"])
        if pair_id in result:
            raise X4Error("X4 canonical test context has duplicate evaluator pair ids")
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
            raise X4Error("X4 full artifact is missing/empty/symlinked: %s" % path)
    _validate_eval_command(
        load_strict_json(command_path),
        k=int(cell["k"]),
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
        raise X4Error("X4 full legacy identity does not close 57/14,042")
    _verify_result_sample_content(payload, _expected_by_pair_id(test_records))
    sidecars = sorted(output_dir.glob("samples_*.jsonl"), key=lambda path: path.name)
    entry = {
        "cell": str(cell["label"]),
        "window": WINDOW,
        "k": int(cell["k"]),
        "fixed_horizon": True,
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
    x4_root: Path,
    x4_card: Mapping[str, Any],
    job_id: str,
    sacct_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(x4_root)
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
        raise X4Error("X4 full output directory set has missing/extra cells")
    for cell in manifest["full"]["cells"]:
        for name in ("command_args.json", "model_revision.json", "results.json"):
            path = Path(cell["output_dir"]) / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise X4Error("X4 3/3 seal lacks a required non-empty artifact")
    _, test_records, test_manifest = _load_test_context(
        phase3_card_path, p3c_root, x4_card
    )
    entries = []
    for cell, scheduler in zip(manifest["full"]["cells"], terminal):
        entry = _inspect_full_identity(cell, test_records)
        entry["job_id"] = "%s_%d" % (job_id, cell["index"])
        entry["scheduler"] = dict(scheduler)
        entries.append(entry)
    return terminal, entries, test_records, test_manifest


def _load_comparator_entries(
    p3c_root: Path, x4_card: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    allowlist = load_strict_json(
        Path(p3c_root) / "historical13_source_allowlist_receipt.json"
    )
    verify_manifest_sha256(allowlist)
    entries = {str(entry.get("cell")): dict(entry) for entry in allowlist.get("cells", ())}
    if not {"baseline", "12:15"}.issubset(entries):
        raise X4Error("X4 comparator allowlist lacks baseline or 12:15")
    card_comparators = x4_card["comparators"]
    expected = {
        "baseline": card_comparators["baseline"],
        "12:15": card_comparators["context_12_15_k2"],
    }
    result = {}
    for source_name, output_name in (("baseline", "baseline"), ("12:15", "12:15@K2")):
        entry = entries[source_name]
        results = entry.get("results", {})
        if (
            Path(str(results.get("path", ""))).resolve()
            != Path(str(expected[source_name]["results_path"])).resolve()
            or results.get("file_sha256") != expected[source_name]["results_sha256"]
        ):
            raise X4Error("X4 comparator path/hash differs for %s" % source_name)
        result[output_name] = entry
    return result


def _load_correctness_bundle(
    *,
    candidate_entries: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    test_manifest: Mapping[str, Any],
    p3c_root: Path,
    x4_card: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, Any]]:
    comparator_entries = _load_comparator_entries(p3c_root, x4_card)
    maps: Dict[str, Dict[str, bool]] = {}
    summaries: Dict[str, Any] = {"comparators": {}, "candidates": {}}
    for name in COMPARATOR_ORDER:
        correctness, summary = _load_outcome_cell(
            comparator_entries[name], test_records, test_manifest
        )
        maps[name] = correctness
        summaries["comparators"][name] = summary
    baseline_order = list(maps["baseline"])
    if list(maps["12:15@K2"]) != baseline_order or len(baseline_order) != 14042:
        raise X4Error("X4 comparator canonical identity order differs")
    expected_accuracy = {
        "baseline": x4_card["comparators"]["baseline"]["known_accuracy_percent"],
        "12:15@K2": x4_card["comparators"]["context_12_15_k2"][
            "known_accuracy_percent"
        ],
    }
    for name in COMPARATOR_ORDER:
        observed = sum(maps[name].values()) * 100.0 / 14042.0
        if not math.isclose(
            observed, float(expected_accuracy[name]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise X4Error("X4 comparator known accuracy mismatch for %s" % name)
    for entry in candidate_entries:
        label = str(entry["cell"])
        correctness, summary = _load_outcome_cell(entry, test_records, test_manifest)
        if list(correctness) != baseline_order:
            raise X4Error("X4 candidate canonical identity order differs")
        maps[label] = correctness
        summary["window"] = WINDOW
        summary["k"] = int(entry["k"])
        summary["fixed_horizon"] = True
        summaries["candidates"][label] = summary
    if tuple(label for label in K_LABELS if label in maps) != K_LABELS:
        raise X4Error("X4 candidate correctness membership differs")
    return maps, summaries


def _contrast_order() -> Tuple[str, ...]:
    return (
        "K2_vs_baseline",
        "K3_vs_baseline",
        "K4_vs_baseline",
        "K2_vs_12:15@K2",
        "K3_vs_12:15@K2",
        "K4_vs_12:15@K2",
        "K3_vs_K2",
        "K4_vs_K2",
        "K4_vs_K3",
    )


def joint_paired_bootstrap(
    contrasts: Mapping[str, Sequence[int]],
    *,
    replicates: int,
    seed: int,
    force_python: bool = False,
) -> Dict[str, Any]:
    expected_order = _contrast_order()
    if tuple(contrasts) != expected_order:
        raise X4Error("X4 bootstrap contrast order differs from exact nine contrasts")
    lengths = {len(values) for values in contrasts.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
        raise X4Error("X4 bootstrap contrasts must have one positive sample count")
    count = next(iter(lengths))
    rows = []
    for name in expected_order:
        row = []
        for value in contrasts[name]:
            if (
                isinstance(value, bool)
                or int(value) != value
                or int(value) not in (-1, 0, 1)
            ):
                raise X4Error("X4 paired difference must be -1, 0, or 1")
            row.append(int(value))
        rows.append(row)
    if not isinstance(replicates, int) or replicates < 1:
        raise X4Error("X4 bootstrap replicates must be positive")
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
        "joint_index_reuse_all_nine_contrasts": True,
        "contrasts": result,
    }


def _paired_counts(
    reference: Sequence[bool], candidate: Sequence[bool]
) -> Dict[str, int]:
    if len(reference) != len(candidate) or not reference:
        raise X4Error("X4 paired correctness vectors differ in length")
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


def holm_step_down_three(
    raw_p_by_k: Mapping[str, float], *, alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    if set(raw_p_by_k) != set(K_LABELS):
        raise X4Error("X4 Holm family must be exactly K2/K3/K4 versus baseline")
    order_index = {label: index for index, label in enumerate(K_LABELS)}
    ranked = sorted(
        K_LABELS,
        key=lambda label: (float(raw_p_by_k[label]), order_index[label]),
    )
    running = 0.0
    result = {}
    family_size = len(K_LABELS)
    for rank, label in enumerate(ranked, start=1):
        raw = float(raw_p_by_k[label])
        if not math.isfinite(raw) or raw < 0.0 or raw > 1.0:
            raise X4Error("X4 Holm raw p-value is invalid")
        adjusted = max(running, min(1.0, (family_size - rank + 1) * raw))
        running = adjusted
        result[label] = {
            "raw_p": raw,
            "adjusted_p": adjusted,
            "significant": adjusted < alpha,
            "holm_rank": rank,
        }
    return {label: result[label] for label in K_LABELS}


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
            raise X4Error("X4 Slurm ElapsedRaw is not an integer") from exc
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
        "terminal_task_count": 3,
        "completed_0_0": 3,
        "elapsed_raw_seconds_sum": sum(elapsed),
        "gpu_hours_allocated_from_elapsed": gpu_seconds / 3600.0,
        "partitions": sorted({row.get("Partition", "") for row in terminal}),
        "nodes": sorted({row.get("NodeList", "") for row in terminal}),
        "alloc_tres": sorted({row.get("AllocTRES", "") for row in terminal}),
        "tasks": [dict(row) for row in terminal],
        "scientific_argv_config_hash_unchanged": True,
    }


def _comparison_record(
    *,
    bootstrap: Mapping[str, Any],
    contrast_name: str,
    reference: Sequence[bool],
    candidate: Sequence[bool],
) -> Dict[str, Any]:
    contrast = bootstrap["contrasts"][contrast_name]
    interval = contrast["percentile_95_ci_fraction"]
    counts = _paired_counts(reference, candidate)
    return {
        "delta_accuracy_fraction": contrast["point_delta_accuracy_fraction"],
        "delta_accuracy_pp": contrast["point_delta_accuracy_fraction"] * 100.0,
        "paired_ci95_fraction": list(interval),
        "paired_ci95_pp": [float(value) * 100.0 for value in interval],
        "paired_flip_counts": counts,
        "exact_mcnemar_p": exact_mcnemar_p(
            counts["wrong_to_right"], counts["right_to_wrong"]
        ),
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
    all_labels = COMPARATOR_ORDER + K_LABELS
    values = {
        name: [bool(maps[name][key]) for key in identity_order]
        for name in all_labels
    }
    contrasts: Dict[str, List[int]] = {}
    for label in K_LABELS:
        contrasts["%s_vs_baseline" % label] = [
            int(candidate) - int(reference)
            for reference, candidate in zip(values["baseline"], values[label])
        ]
    for label in K_LABELS:
        contrasts["%s_vs_12:15@K2" % label] = [
            int(candidate) - int(reference)
            for reference, candidate in zip(values["12:15@K2"], values[label])
        ]
    for candidate, reference in (("K3", "K2"), ("K4", "K2"), ("K4", "K3")):
        contrasts["%s_vs_%s" % (candidate, reference)] = [
            int(after) - int(before)
            for before, after in zip(values[reference], values[candidate])
        ]
    bootstrap = joint_paired_bootstrap(
        contrasts,
        replicates=2000,
        seed=20260710,
    )
    accuracies = {label: sum(values[label]) / 14042.0 for label in K_LABELS}
    raw_baseline_p = {}
    baseline_records = {}
    context_records = {}
    for label in K_LABELS:
        baseline_record = _comparison_record(
            bootstrap=bootstrap,
            contrast_name="%s_vs_baseline" % label,
            reference=values["baseline"],
            candidate=values[label],
        )
        baseline_records[label] = baseline_record
        raw_baseline_p[label] = baseline_record["exact_mcnemar_p"]
        context_records[label] = _comparison_record(
            bootstrap=bootstrap,
            contrast_name="%s_vs_12:15@K2" % label,
            reference=values["12:15@K2"],
            candidate=values[label],
        )
    holm = holm_step_down_three(raw_baseline_p)
    ranking = sorted(K_LABELS, key=lambda label: (-accuracies[label], int(label[1:])))
    rows = []
    for label, k in zip(K_LABELS, K_ORDER):
        versus_baseline = dict(
            baseline_records[label],
            holm_adjusted_mcnemar_p=holm[label]["adjusted_p"],
            holm_significant=holm[label]["significant"],
            holm_rank=holm[label]["holm_rank"],
        )
        rows.append(
            {
                "label": label,
                "window": WINDOW,
                "k": k,
                "step_size": 1.0 / float(k),
                "total_horizon": 1.0,
                "accuracy_fraction": accuracies[label],
                "accuracy_percent": accuracies[label] * 100.0,
                "observed_rank": ranking.index(label) + 1,
                "versus_baseline": versus_baseline,
                "versus_12_15_k2": context_records[label],
                "source": summaries["candidates"][label],
            }
        )
    pairwise = []
    for candidate, reference in (("K3", "K2"), ("K4", "K2"), ("K4", "K3")):
        pairwise.append(
            {
                "contrast": "%s_vs_%s" % (candidate, reference),
                "candidate": candidate,
                "reference": reference,
                **_comparison_record(
                    bootstrap=bootstrap,
                    contrast_name="%s_vs_%s" % (candidate, reference),
                    reference=values[reference],
                    candidate=values[candidate],
                ),
            }
        )
    candidate_source = {
        str(entry["cell"]): {
            "window": WINDOW,
            "k": int(entry["k"]),
            "output_dir": entry["output_dir"],
            "command_args_file_sha256": entry["command_args"]["file_sha256"],
            "model_revision_file_sha256": entry["model_revision"]["file_sha256"],
            "results_file_sha256": entry["results"]["file_sha256"],
            "sample_sidecar_file_sha256": {
                Path(value["path"]).name: value["file_sha256"]
                for value in entry["sample_sidecars"]
            },
        }
        for entry in candidate_entries
    }
    comparator_accuracy = {
        name: sum(values[name]) / 14042.0 for name in COMPARATOR_ORDER
    }
    best_label = ranking[0]
    return {
        "sample_count": 14042,
        "subject_count": 57,
        "identity_closure": {
            "phase1_legacy_ordered_identity_sha256": EXPECTED_LEGACY_IDENTITY_SHA256,
            "canonical_ordered_identity_sha256": EXPECTED_CANONICAL_IDENTITY_SHA256,
            "all_three_k_cells_and_both_comparators_exact": True,
            "identity_intersection_fallback_used": False,
        },
        "recipe_closure": {
            "window": WINDOW,
            "single_decoder_layer_body": True,
            "ordered_k": list(K_ORDER),
            "fixed_horizon": True,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "model_revision": MODEL_REVISION,
        },
        "comparators": {
            name: {
                "accuracy_fraction": comparator_accuracy[name],
                "accuracy_percent": comparator_accuracy[name] * 100.0,
                "source": summaries["comparators"][name],
            }
            for name in COMPARATOR_ORDER
        },
        "candidate_sources": candidate_source,
        "bootstrap": bootstrap,
        "k_cells": rows,
        "pairwise_k_contrasts": pairwise,
        "observed_k_ranking": ranking,
        "observed_best_k": {
            "label": best_label,
            "k": int(best_label[1:]),
            "accuracy_fraction": accuracies[best_label],
            "accuracy_percent": accuracies[best_label] * 100.0,
        },
        "question_answer": {
            "any_k_exceeds_baseline": any(
                accuracies[label] > comparator_accuracy["baseline"] for label in K_LABELS
            ),
            "k_values_exceeding_baseline": [
                int(label[1:])
                for label in K_LABELS
                if accuracies[label] > comparator_accuracy["baseline"]
            ],
            "any_k_exceeds_12_15_k2": any(
                accuracies[label] > comparator_accuracy["12:15@K2"]
                for label in K_LABELS
            ),
            "k_values_exceeding_12_15_k2": [
                int(label[1:])
                for label in K_LABELS
                if accuracies[label] > comparator_accuracy["12:15@K2"]
            ],
        },
        "holm_family": {
            "method": "Holm step-down",
            "family": ["%s_vs_baseline" % label for label in K_LABELS],
            "family_size": 3,
            "familywise_alpha": 0.05,
        },
        "resource_accounting": _resource_accounting(terminal, manifest, job_id),
        "full_seal": {
            "job_id": str(job_id),
            "terminal_completed_0_0": 3,
            "task_count_per_cell": 57,
            "sample_count_per_cell": 14042,
            "recipe_revision_renderer_closure": True,
            "partial_outcome_viewing": False,
            "all_identity_recipe_checks_completed_before_first_correctness_access": True,
        },
    }


def analyze_full(
    *,
    x4_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    x4_root: Path,
    expected_commit: str,
    job_id: str,
    argv: Sequence[str],
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(x4_card_path, phase3_card_path, p3c_root, x4_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    root = Path(x4_root)
    if (root / ANALYSIS_NAME).exists() or (root / VERIFIER_NAME).exists():
        raise FileExistsError("X4 analysis/verifier is write-once")
    card = load_x4_card(x4_card_path)
    manifest = load_strict_json(root / MANIFEST_NAME)
    validate_manifest(manifest)
    if manifest["git"]["implementation_commit"] != expected_commit:
        raise X4Error("X4 manifest implementation commit differs")
    if manifest["frozen_inputs"]["x4_card_file_sha256"] != file_sha256(x4_card_path):
        raise X4Error("X4 manifest card hash differs from committed card")
    terminal, entries, test_records, test_manifest = _sealed_full_sources(
        manifest=manifest,
        phase3_card_path=phase3_card_path,
        p3c_root=p3c_root,
        x4_root=x4_root,
        x4_card=card,
        job_id=job_id,
        sacct_rows=sacct_rows,
    )
    maps, summaries = _load_correctness_bundle(
        candidate_entries=entries,
        test_records=test_records,
        test_manifest=test_manifest,
        p3c_root=p3c_root,
        x4_card=card,
    )
    core = _compute_core(maps, summaries, entries, terminal, manifest, job_id)
    analysis: Dict[str, Any] = {
        "schema_version": "loopscope.post_phase3.x4-layer15-k234-analysis.v1",
        "artifact_role": "x4_one_shot_layer15_k234_paired_analysis",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
        "x4_card_file_sha256": file_sha256(x4_card_path),
        "phase3_card_file_sha256": PHASE3_CARD_BYTE_SHA256,
        "x4_manifest_file_sha256": file_sha256(root / MANIFEST_NAME),
        "x4_manifest_internal_sha256": manifest["manifest_sha256"],
        "analysis_run_count": 1,
        "analysis_rerun_authorized": False,
        "result": core,
        "status": "X4_ANALYSIS_COMPLETE",
    }
    attach_manifest_sha256(analysis)
    write_new_json(root / ANALYSIS_NAME, analysis)
    del maps
    gc.collect()
    return analysis


def verify_full(
    *,
    x4_card_path: Path,
    phase3_card_path: Path,
    p3c_root: Path,
    x4_root: Path,
    expected_commit: str,
    job_id: str,
    argv: Sequence[str],
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(x4_card_path, phase3_card_path, p3c_root, x4_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    root = Path(x4_root)
    if (root / VERIFIER_NAME).exists():
        raise FileExistsError("X4 verifier is write-once")
    analysis = load_strict_json(root / ANALYSIS_NAME)
    verify_manifest_sha256(analysis)
    if (
        analysis.get("analysis_run_count") != 1
        or analysis.get("analysis_rerun_authorized") is not False
        or analysis.get("status") != "X4_ANALYSIS_COMPLETE"
    ):
        raise X4Error("X4 analysis one-shot contract differs")
    card = load_x4_card(x4_card_path)
    manifest = load_strict_json(root / MANIFEST_NAME)
    validate_manifest(manifest)
    terminal, entries, test_records, test_manifest = _sealed_full_sources(
        manifest=manifest,
        phase3_card_path=phase3_card_path,
        p3c_root=p3c_root,
        x4_root=x4_root,
        x4_card=card,
        job_id=job_id,
        sacct_rows=sacct_rows,
    )
    maps, summaries = _load_correctness_bundle(
        candidate_entries=entries,
        test_records=test_records,
        test_manifest=test_manifest,
        p3c_root=p3c_root,
        x4_card=card,
    )
    recomputed = _compute_core(maps, summaries, entries, terminal, manifest, job_id)
    if canonical_json_bytes(recomputed) != canonical_json_bytes(analysis.get("result")):
        raise X4Error("X4 verifier recomputation differs from analysis")
    verifier: Dict[str, Any] = {
        "schema_version": "loopscope.post_phase3.x4-layer15-k234-verifier.v1",
        "artifact_role": "x4_independent_source_reload_and_statistical_recompute",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "argv": list(argv),
        "analysis_file_sha256": file_sha256(root / ANALYSIS_NAME),
        "analysis_internal_sha256": analysis["manifest_sha256"],
        "x4_manifest_file_sha256": file_sha256(root / MANIFEST_NAME),
        "x4_manifest_internal_sha256": manifest["manifest_sha256"],
        "verification": {
            "full_3_of_3_terminal_rechecked": True,
            "all_five_identity_maps_reloaded_exact": True,
            "three_k_accuracies_recomputed": True,
            "nine_joint_paired_bootstrap_contrasts_recomputed": True,
            "bootstrap_index_stream_sha256": recomputed["bootstrap"][
                "index_stream_sha256"
            ],
            "paired_flip_counts_recomputed": True,
            "exact_mcnemar_recomputed": True,
            "three_test_holm_recomputed": True,
            "pairwise_k_order_and_contrasts_recomputed": True,
            "best_k_and_comparator_checks_recomputed": True,
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
    "K_LABELS",
    "K_ORDER",
    "MANIFEST_NAME",
    "VERIFIER_NAME",
    "WINDOW",
    "X4Error",
    "analyze_full",
    "build_manifest",
    "holm_step_down_three",
    "joint_paired_bootstrap",
    "load_x4_card",
    "prepare_run",
    "validate_manifest",
    "validate_scheduler_rows",
    "verify_full",
    "verify_smoke",
]
