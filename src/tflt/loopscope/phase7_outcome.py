"""Gate E's minimal, outcome-barriered MMLU 5-shot panel runner.

The module intentionally keeps the full lm-eval result only in memory.  It
persists a readable test membership plus the minimal per-identity correctness
bit needed for the one authorized paired analysis; prompts, targets, token
IDs, logits, probabilities, and tensors never cross the write boundary.
"""

from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


GATE = "E"
CARD_RELATIVE = Path("configs/loopscope/phase7_gate_e_outcome_card.json")
CARD_SCHEMA = "loopscope.phase7.gate-e-card.v1"
RUN_SCHEMA = "loopscope.phase7.gate-e-run.v1"
OUTCOME_SCHEMA = "loopscope.phase7.gate-e-outcome-row.v1"
DATASET_REPO = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
EXPECTED_MODELS = ("qwen25_3b", "llama32_3b", "gemma2_2b")
EXPECTED_COUNTS = {"qwen25_3b": 13, "llama32_3b": 13, "gemma2_2b": 9}
FROZEN_BATCH_SIZES = {"qwen25_3b": 16, "llama32_3b": 16, "gemma2_2b": 8}
FROZEN_MODELS = {
    "qwen25_3b": {
        "repo": "Qwen/Qwen2.5-3B",
        "revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        "terminal": "SELECTED_WINDOW",
        "interpretation": "fixed_top_available_exploratory_panel",
        "windows": ((14, 18), (19, 23), (16, 20)),
    },
    "llama32_3b": {
        "repo": "meta-llama/Llama-3.2-3B",
        "revision": "13afe5124825b4f3751f836b40dafda64c1ed062",
        "terminal": "SELECTED_WINDOW",
        "interpretation": "fixed_top_available_exploratory_panel",
        "windows": ((10, 14), (11, 15), (9, 13)),
    },
    "gemma2_2b": {
        "repo": "google/gemma-2-2b",
        "revision": "c5ebcd40d208330abc697524c919956e692655cf",
        "terminal": "ABSTAIN_COMBINED_RANK_UNSTABLE",
        "interpretation": "post_ranking_exploratory_after_abstain",
        "windows": ((10, 14), (12, 16)),
    },
}
EXPECTED_POPULATION = 14042
EXPECTED_SUBJECTS = 57
DEBUG_TASKS = ("mmlu_abstract_algebra", "mmlu_anatomy")
REMOTE_REPO = Path("/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope")
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")


class Phase7OutcomeError(RuntimeError):
    """A fail-fast Gate E implementation, contract, or runtime error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase7OutcomeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _read_json(path: Path) -> Any:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "required JSON artifact is absent: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, ValueError, TypeError) as exc:
        raise Phase7OutcomeError("invalid JSON artifact: %s" % path) from exc


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), "BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise Phase7OutcomeError("could not write JSON artifact: %s" % path) from exc


def _write_new_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), "BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                handle.write("\n")
                count += 1
    except (OSError, TypeError, ValueError) as exc:
        raise Phase7OutcomeError("could not write JSONL artifact: %s" % path) from exc
    _require(count > 0, "JSONL artifact must contain at least one row")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "required JSONL artifact is absent: %s" % path)
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line, parse_constant=_reject_json_constant)
                _require(isinstance(value, dict), "JSONL row %d is not an object" % line_number)
                rows.append(dict(value))
    except (OSError, ValueError, TypeError) as exc:
        raise Phase7OutcomeError("invalid JSONL artifact: %s" % path) from exc
    _require(bool(rows), "JSONL artifact has no rows: %s" % path)
    return rows


def _run_command(argv: Sequence[str], *, cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise Phase7OutcomeError("command failed (%d): %s" % (completed.returncode, " ".join(argv[:3])))
    return completed.stdout.strip()


def _git_binary() -> str:
    """Keep provenance closure independent of a compute node's module PATH."""

    return "/usr/bin/git" if os.path.isfile("/usr/bin/git") else "git"


def validate_git(expected_commit: str, *, require_clean: bool) -> Dict[str, Any]:
    root = repository_root()
    expected = str(expected_commit).strip()
    _require(expected, "expected commit is empty")
    git_binary = _git_binary()
    branch = _run_command((git_binary, "rev-parse", "--abbrev-ref", "HEAD"), cwd=root)
    commit = _run_command((git_binary, "rev-parse", "HEAD"), cwd=root)
    top = _run_command((git_binary, "rev-parse", "--show-toplevel"), cwd=root)
    ancestry = subprocess.run(
        (git_binary, "merge-base", "--is-ancestor", "4f59bd93eca4da3cbf458a93508f91c5b23912bc", "HEAD"),
        cwd=str(root),
        check=False,
        capture_output=True,
        text=True,
    ).returncode
    status = _run_command((git_binary, "status", "--porcelain"), cwd=root)
    _require(top == str(root), "repository root differs from the Gate E clone")
    _require(branch == "loopscope", "Gate E must run on loopscope branch")
    _require(commit == expected, "working commit differs from the frozen launcher commit")
    _require(ancestry == 0, "required LoopScope ancestry is absent")
    if require_clean:
        _require(status == "", "remote Gate E checkout must be clean")
    return {
        "repository": str(root),
        "branch": branch,
        "commit": commit,
        "clean": status == "",
    }


def _read_git_head_metadata(root: Path) -> Tuple[str, str]:
    """Read the checked-out branch/commit without requiring a Git executable."""

    git_dir = root / ".git"
    head_path = git_dir / "HEAD"
    _require(head_path.is_file(), "compute-node Git metadata is unavailable")
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return "HEAD", head
    ref = head[5:].strip()
    _require(bool(ref), "compute-node Git HEAD ref is empty")
    ref_path = git_dir / ref
    if ref_path.is_file():
        commit = ref_path.read_text(encoding="utf-8").strip()
    else:
        commit = ""
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                value, separator, name = line.partition(" ")
                if separator and name == ref:
                    commit = value.strip()
                    break
    _require(bool(commit), "compute-node Git ref cannot be resolved")
    return ref.rsplit("/", 1)[-1], commit


def validate_runtime_git(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain provenance closure on compute images that omit the Git binary."""

    git = manifest.get("git")
    _require(isinstance(git, Mapping), "run manifest Git provenance is invalid")
    expected = str(git.get("expected_commit", "")).strip()
    validated = git.get("validated")
    _require(isinstance(validated, Mapping), "run manifest lacks prepared Git validation")
    try:
        return validate_git(expected, require_clean=True)
    except (FileNotFoundError, PermissionError):
        root = repository_root()
        _require(validated.get("repository") == str(root), "prepared repository differs")
        _require(validated.get("branch") == "loopscope", "prepared branch differs")
        _require(validated.get("commit") == expected, "prepared commit differs")
        _require(validated.get("clean") is True, "prepared checkout was not clean")
        branch, commit = _read_git_head_metadata(root)
        _require(branch == "loopscope", "runtime branch differs from the frozen branch")
        _require(commit == expected, "runtime commit differs from the frozen commit")
        return {
            "repository": str(root),
            "branch": branch,
            "commit": commit,
            "clean": True,
            "validation": "prepared-clean-plus-live-head",
        }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase7OutcomeError("%s must be a finite number" % label)
    number = float(value)
    if not math.isfinite(number):
        raise Phase7OutcomeError("%s must be finite" % label)
    return number


def load_card(path: Optional[Path] = None) -> Dict[str, Any]:
    card_path = Path(path) if path is not None else repository_root() / CARD_RELATIVE
    card = _read_json(card_path)
    _require(isinstance(card, dict), "Gate E card root must be an object")
    validate_card(card)
    return dict(card)


def validate_card(card: Mapping[str, Any]) -> None:
    _require(isinstance(card, Mapping), "Gate E card must be an object")
    _require(card.get("schema_version") == CARD_SCHEMA, "unsupported Gate E card schema")
    _require(card.get("gate") == GATE and card.get("phase") == 7, "Gate E card identity differs")
    dataset = card.get("dataset")
    evaluator = card.get("evaluator")
    runtime = card.get("runtime")
    grid = card.get("loop_grid")
    analysis = card.get("analysis")
    models = card.get("models")
    _require(isinstance(dataset, Mapping), "Gate E card lacks dataset")
    _require(
        dataset.get("repo") == DATASET_REPO
        and dataset.get("revision") == DATASET_REVISION
        and dataset.get("split") == "test"
        and dataset.get("population") == EXPECTED_POPULATION
        and dataset.get("subjects") == EXPECTED_SUBJECTS
        and dataset.get("num_fewshot") == 5
        and dataset.get("fewshot_split") == "dev",
        "Gate E dataset contract differs",
    )
    _require(isinstance(evaluator, Mapping), "Gate E card lacks evaluator")
    _require(
        evaluator.get("lm_eval_version") == "0.4.11"
        and evaluator.get("prompt") == "plain_non_chat"
        and evaluator.get("apply_chat_template") is False
        and evaluator.get("fewshot_as_multiturn") is False
        and evaluator.get("generation") is False
        and evaluator.get("metric") == "acc,none"
        and evaluator.get("log_samples") is True,
        "Gate E evaluator contract differs",
    )
    _require(isinstance(runtime, Mapping), "Gate E card lacks runtime")
    _require(
        runtime.get("dtype") == "bfloat16"
        and runtime.get("quantization") == "none"
        and runtime.get("batch_size") == 16
        and runtime.get("batch_size_by_model") == FROZEN_BATCH_SIZES,
        "Gate E runtime contract differs",
    )
    _require(isinstance(grid, Mapping), "Gate E card lacks loop grid")
    _require(
        grid.get("k") == [2, 3]
        and grid.get("cache_strategy") == ["first", "last"]
        and grid.get("iteration_mode") == "block"
        and grid.get("strategy") == "euler"
        and _finite_number(grid.get("alpha"), "loop alpha") == 1.0
        and _finite_number(grid.get("beta"), "loop beta") == 0.0
        and grid.get("decode_mode") == "full"
        and grid.get("window_width") == 4,
        "Gate E loop grid differs",
    )
    _require(isinstance(analysis, Mapping), "Gate E card lacks analysis")
    _require(
        analysis.get("bootstrap_replicates") == 2000
        and analysis.get("bootstrap_seed") == 20260803
        and analysis.get("paired_bootstrap") == "subject_stratified_percentile_95",
        "Gate E paired analysis contract differs",
    )
    _require(isinstance(models, list) and len(models) == len(EXPECTED_MODELS), "Gate E model membership differs")
    _require(tuple(str(item.get("model_key")) for item in models if isinstance(item, Mapping)) == EXPECTED_MODELS, "Gate E model order differs")
    for model in models:
        _require(isinstance(model, Mapping), "Gate E model entry must be an object")
        key = str(model.get("model_key"))
        frozen = FROZEN_MODELS[key]
        _require(model.get("model_repo") == frozen["repo"], "model repo differs")
        _require(model.get("model_revision") == frozen["revision"], "model revision differs")
        _require(model.get("selector_terminal") == frozen["terminal"], "model selector terminal differs")
        _require(model.get("outcome_interpretation") == frozen["interpretation"], "model outcome interpretation differs")
        _require(model.get("expected_cell_count") == EXPECTED_COUNTS[key], "per-model cell count differs")
        windows = model.get("windows")
        expected_windows = 3 if key != "gemma2_2b" else 2
        _require(isinstance(windows, list) and len(windows) == expected_windows, "model window count differs")
        seen = []
        for rank, window in enumerate(windows, start=1):
            _require(isinstance(window, Mapping), "window entry must be an object")
            start = window.get("start")
            stop = window.get("stop_exclusive")
            _require(isinstance(start, int) and isinstance(stop, int) and stop - start == 4, "window width differs")
            _require(window.get("rank") == rank, "window rank/order differs")
            _require((start, stop) not in seen, "duplicate model window")
            seen.append((start, stop))
        _require(tuple(seen) == frozen["windows"], "model top-available window order differs")


def expand_cells(card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand the frozen 3/3/2 panel in its fixed model/window/k/cache order."""

    validate_card(card)
    grid = card["loop_grid"]
    batch_sizes = card["runtime"]["batch_size_by_model"]
    cells: List[Dict[str, Any]] = []
    for model in card["models"]:
        key = str(model["model_key"])
        batch_size = int(batch_sizes[key])
        cells.append(
            {
                "cell_id": "%s__no_loop" % key,
                "model_key": key,
                "model_repo": str(model["model_repo"]),
                "model_revision": str(model["model_revision"]),
                "selector_terminal": str(model["selector_terminal"]),
                "outcome_interpretation": str(model["outcome_interpretation"]),
                "role": "no_loop_baseline",
                "loop_enabled": False,
                "window_half_open": None,
                "window_inclusive": None,
                "k": None,
                "cache_strategy": None,
                "iteration_mode": None,
                "strategy": None,
                "alpha": None,
                "beta": None,
                "decode_mode": None,
                "dtype": "bfloat16",
                "batch_size": batch_size,
            }
        )
        for window in model["windows"]:
            start = int(window["start"])
            stop = int(window["stop_exclusive"])
            inclusive = "%d:%d" % (start, stop - 1)
            half_open = "%d:%d" % (start, stop)
            for k in grid["k"]:
                for cache in grid["cache_strategy"]:
                    cells.append(
                        {
                            "cell_id": "%s__w%d_%d__k%d__cache_%s__block" % (key, start, stop, int(k), cache),
                            "model_key": key,
                            "model_repo": str(model["model_repo"]),
                            "model_revision": str(model["model_revision"]),
                            "selector_terminal": str(model["selector_terminal"]),
                            "outcome_interpretation": str(model["outcome_interpretation"]),
                            "role": "loop",
                            "loop_enabled": True,
                            "window_half_open": half_open,
                            "window_inclusive": inclusive,
                            "k": int(k),
                            "cache_strategy": str(cache),
                            "iteration_mode": str(grid["iteration_mode"]),
                            "strategy": str(grid["strategy"]),
                            "alpha": float(grid["alpha"]),
                            "beta": float(grid["beta"]),
                            "decode_mode": str(grid["decode_mode"]),
                            "dtype": "bfloat16",
                            "batch_size": batch_size,
                        }
                    )
    for index, cell in enumerate(cells):
        cell["index"] = index
    validate_cells(cells, formal=True)
    return cells


def validate_cells(cells: Sequence[Mapping[str, Any]], *, formal: bool, allow_legacy_batch: bool = False) -> None:
    _require(isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)), "cell list is invalid")
    seen_ids = set()
    per_model: Dict[str, List[Mapping[str, Any]]] = {key: [] for key in EXPECTED_MODELS}
    for index, cell in enumerate(cells):
        _require(isinstance(cell, Mapping), "cell %d is not an object" % index)
        _require(cell.get("index") == index, "cell index/order differs")
        cell_id = str(cell.get("cell_id", ""))
        key = str(cell.get("model_key", ""))
        _require(cell_id and cell_id not in seen_ids, "cell id is absent or duplicate")
        _require(key in per_model, "cell model key differs")
        seen_ids.add(cell_id)
        per_model[key].append(cell)
        observed_batch = cell.get("batch_size")
        if allow_legacy_batch and observed_batch is None:
            observed_batch = 16
        expected_batch = 16 if allow_legacy_batch else FROZEN_BATCH_SIZES[key]
        _require(observed_batch == expected_batch, "cell batch size differs")
        loop = cell.get("loop_enabled")
        _require(isinstance(loop, bool), "loop_enabled must be boolean")
        if not loop:
            _require(cell.get("role") == "no_loop_baseline", "baseline role differs")
            _require(cell.get("window_half_open") is None and cell.get("k") is None, "baseline loop fields differ")
        else:
            _require(cell.get("role") == "loop", "loop role differs")
            _require(cell.get("window_inclusive") and cell.get("window_half_open"), "loop window is absent")
            _require(cell.get("k") in (2, 3), "loop k differs")
            _require(cell.get("cache_strategy") in ("first", "last"), "loop cache differs")
            _require(cell.get("iteration_mode") == "block", "loop mode differs")
            _require(cell.get("strategy") == "euler", "loop strategy differs")
            _require(cell.get("decode_mode") == "full", "loop decode differs")
    if formal:
        _require(len(cells) == 35, "formal panel does not contain 35 cells")
        for key, expected in EXPECTED_COUNTS.items():
            rows = per_model[key]
            _require(len(rows) == expected, "per-model formal cell count differs")
            _require(sum(not bool(item["loop_enabled"]) for item in rows) == 1, "baseline is not unique")
            loop_rows = [item for item in rows if item["loop_enabled"]]
            _require(len(loop_rows) == expected - 1, "loop panel count differs")
            expected_grid = {(item["window_half_open"], item["k"], item["cache_strategy"]) for item in loop_rows}
            window_count = 3 if key != "gemma2_2b" else 2
            _require(len(expected_grid) == window_count * 4, "loop Cartesian grid is incomplete")


def debug_cell_indices(cells: Sequence[Mapping[str, Any]]) -> List[int]:
    """Return seven fixed representative cells covering three models and k/cache paths."""

    validate_cells(cells, formal=True)

    def find(key: str, *, loop: bool, k: Optional[int] = None, cache: Optional[str] = None) -> int:
        for cell in cells:
            if str(cell["model_key"]) != key or bool(cell["loop_enabled"]) != loop:
                continue
            if k is not None and cell.get("k") != k:
                continue
            if cache is not None and cell.get("cache_strategy") != cache:
                continue
            return int(cell["index"])
        raise Phase7OutcomeError("debug representative cell is absent")

    selected = [
        find("qwen25_3b", loop=False),
        find("llama32_3b", loop=False),
        find("gemma2_2b", loop=False),
        find("qwen25_3b", loop=True, k=2, cache="first"),
        find("qwen25_3b", loop=True, k=2, cache="last"),
        find("llama32_3b", loop=True, k=3, cache="first"),
        find("gemma2_2b", loop=True, k=3, cache="last"),
    ]
    _require(len(set(selected)) == 7, "debug cells are unexpectedly duplicate")
    return selected


def _validate_identity_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    _require(isinstance(row, Mapping), "identity manifest row is invalid")
    required = {"canonical_identity", "subject", "task_name", "test_index"}
    _require(set(row) == required, "identity manifest fields differ")
    subject = str(row["subject"]).strip()
    task = str(row["task_name"]).strip()
    index = row["test_index"]
    identity = str(row["canonical_identity"]).strip()
    _require(subject and task == "mmlu_" + subject, "identity subject/task closure differs")
    _require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, "test index differs")
    _require(identity == "mmlu_%s:test:%d" % (subject, index), "canonical test identity differs")
    return {"canonical_identity": identity, "subject": subject, "task_name": task, "test_index": index}


def validate_identity_manifest(
    rows: Iterable[Mapping[str, Any]], *, expected_count: Optional[int] = None, expected_subjects: Optional[int] = None
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    identities = set()
    pairs = set()
    for raw in rows:
        row = _validate_identity_row(raw)
        _require(row["canonical_identity"] not in identities, "identity manifest has duplicate identity")
        pair = (row["task_name"], row["test_index"])
        _require(pair not in pairs, "identity manifest has duplicate task/index")
        identities.add(row["canonical_identity"])
        pairs.add(pair)
        normalized.append(row)
    _require(bool(normalized), "identity manifest is empty")
    if expected_count is not None:
        _require(len(normalized) == expected_count, "identity manifest record count differs")
    if expected_subjects is not None:
        _require(len({row["subject"] for row in normalized}) == expected_subjects, "identity manifest subject count differs")
    return normalized


def _task_map() -> Dict[str, Any]:
    try:
        from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend
    except Exception as exc:  # pragma: no cover - exercised only on HPC2.
        raise Phase7OutcomeError("lm-eval MMLU task backend is unavailable") from exc
    backend = LmEvalMMLURendererBackend()
    manager = backend._task_manager()
    loaded = manager.load_task_or_group("mmlu")
    flat = backend._flatten_tasks(loaded)
    tasks = {str(name): task for name, task in flat.items() if str(name).startswith("mmlu_")}
    _require(len(tasks) == EXPECTED_SUBJECTS, "lm-eval MMLU task membership differs")
    return dict(sorted(tasks.items()))


def build_canonical_test_manifest(*, cache_dir: str | None = None) -> List[Dict[str, Any]]:
    """Build the safe, readable 14,042-row test identity manifest offline."""

    try:
        from datasets import DownloadMode, load_dataset
        from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend
    except Exception as exc:  # pragma: no cover - exercised only on HPC2.
        raise Phase7OutcomeError("Gate E test manifest requires lm-eval and datasets") from exc
    backend = LmEvalMMLURendererBackend()
    root = str(Path(cache_dir or os.environ.get("HF_DATASETS_CACHE", "")).expanduser())
    _require(root and root != ".", "HF datasets cache is required for Gate E")
    rows: List[Dict[str, Any]] = []
    for task_name, task in _task_map().items():
        subject = task_name[len("mmlu_") :]
        config = getattr(task, "config", None)
        dataset_path = str(backend._config_value(config, "dataset_path") or "").strip()
        dataset_name = backend._config_value(config, "dataset_name")
        _require(dataset_path == DATASET_REPO, "MMLU task dataset repo differs")
        kwargs = backend._config_value(config, "dataset_kwargs") or {}
        _require(isinstance(kwargs, Mapping), "MMLU task dataset kwargs differ")
        dataset_kwargs = dict(kwargs)
        for key in ("revision", "cache_dir", "download_mode"):
            dataset_kwargs.pop(key, None)
        try:
            test = load_dataset(
                path=dataset_path,
                name=dataset_name,
                split="test",
                revision=DATASET_REVISION,
                cache_dir=root,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                **dataset_kwargs,
            )
        except Exception as exc:  # pragma: no cover - exercised only on HPC2.
            raise Phase7OutcomeError("offline MMLU test load failed for %s" % task_name) from exc
        _require("subject" in set(getattr(test, "column_names", ())), "MMLU test lacks subject")
        safe = test.select_columns(["subject"])
        for test_index, raw in enumerate(safe):
            _require(set(raw) == {"subject"}, "test identity projection retained an unsafe field")
            _require(str(raw["subject"]).strip() == subject, "test subject differs from task name")
            rows.append(
                {
                    "canonical_identity": "mmlu_%s:test:%d" % (subject, test_index),
                    "subject": subject,
                    "task_name": task_name,
                    "test_index": test_index,
                }
            )
    return validate_identity_manifest(rows, expected_count=EXPECTED_POPULATION, expected_subjects=EXPECTED_SUBJECTS)


def select_debug_records(full_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized = validate_identity_manifest(full_rows, expected_count=EXPECTED_POPULATION, expected_subjects=EXPECTED_SUBJECTS)
    selected = [row for row in normalized if row["task_name"] in DEBUG_TASKS and row["test_index"] < 2]
    _require(len(selected) == 4, "debug subset must contain four test identities")
    _require({row["task_name"] for row in selected} == set(DEBUG_TASKS), "debug tasks differ")
    return selected


def _run_root(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), "Gate E run root is invalid: %s" % root)
    return root


def _load_run(root: Path) -> Dict[str, Any]:
    value = _read_json(_run_root(root) / "manifest" / "run_manifest.json")
    _require(isinstance(value, dict) and value.get("schema_version") == RUN_SCHEMA, "Gate E run manifest schema differs")
    _require(value.get("gate") == GATE, "Gate E run manifest gate differs")
    return dict(value)


def _base_mode(mode: str) -> str:
    value = str(mode)
    if value in {"formal", "formal_retry"}:
        return "formal"
    if value in {"debug", "debug_retry"}:
        return "debug"
    raise Phase7OutcomeError("Gate E run mode is invalid")


def _is_formal_mode(mode: str) -> bool:
    return _base_mode(mode) == "formal"


def _execution_cell_indices(manifest: Mapping[str, Any]) -> List[int]:
    cells = manifest.get("cells")
    _require(isinstance(cells, list), "run cells are invalid")
    raw = manifest.get("execution_cell_indices")
    if raw is None:
        indices = list(range(len(cells)))
    else:
        _require(isinstance(raw, list), "execution cell indices are invalid")
        indices = list(raw)
    _require(
        all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(cells) for index in indices),
        "execution cell index is invalid",
    )
    _require(len(indices) == len(set(indices)), "execution cell indices are duplicate")
    return sorted(indices)


def _retained_cell_roots(manifest: Mapping[str, Any]) -> Dict[str, Path]:
    raw = manifest.get("retained_cell_roots", {})
    _require(isinstance(raw, Mapping), "retained cell roots are invalid")
    roots: Dict[str, Path] = {}
    for cell_id, raw_path in raw.items():
        _require(isinstance(cell_id, str) and cell_id, "retained cell id is invalid")
        _require(isinstance(raw_path, str) and raw_path, "retained cell path is invalid")
        path = Path(raw_path).expanduser().resolve()
        _require(path.is_dir() and not path.is_symlink(), "retained cell root is unavailable: %s" % path)
        roots[cell_id] = path
    return roots


def _cell_artifact_root(root: Path, manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> Path:
    retained = _retained_cell_roots(manifest)
    cell_id = str(cell["cell_id"])
    path = retained.get(cell_id, root / "cells" / cell_id)
    _require(path.is_dir() and not path.is_symlink(), "cell artifact root is unavailable: %s" % path)
    return path


def _validate_retained_cell_metadata(root: Path, manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> Path:
    cell_root = _cell_artifact_root(root, manifest, cell)
    allowed_names = {"command.json", "model_revision.json", "outcomes.jsonl", "producer_receipt.json", "resource.json"}
    _require({path.name for path in cell_root.iterdir()} == allowed_names, "retained cell artifact barrier differs")
    receipt = _read_json(cell_root / "producer_receipt.json")
    _require(
        isinstance(receipt, Mapping)
        and receipt.get("status") == "COMPLETED"
        and receipt.get("cell_id") == cell["cell_id"]
        and receipt.get("outcome_aggregates_computed") is False,
        "retained cell producer receipt differs",
    )
    _validate_cell_batch_metadata(cell_root, cell)
    _validate_model_revision(cell_root / "model_revision.json", cell)
    return cell_root


def _validate_cell_batch_metadata(cell_root: Path, cell: Mapping[str, Any]) -> int:
    """Close the actual evaluator batch while accepting legacy batch-16 receipts."""

    expected = cell.get("batch_size")
    _require(expected == FROZEN_BATCH_SIZES.get(str(cell.get("model_key"))), "cell batch binding differs")
    command = _read_json(cell_root / "command.json")
    _require(isinstance(command, Mapping) and command.get("batch_size") == expected, "cell command batch differs")
    for name in ("producer_receipt.json", "resource.json"):
        payload = _read_json(cell_root / name)
        observed = payload.get("batch_size") if isinstance(payload, Mapping) else None
        legacy_batch_16 = observed is None and expected == 16
        _require(observed == expected or legacy_batch_16, "%s batch differs" % name)
    return int(expected)


def _retry_cards_compatible(source: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    source_copy = json.loads(json.dumps(source))
    current_copy = json.loads(json.dumps(current))
    for payload in (source_copy, current_copy):
        runtime = payload.get("runtime")
        if isinstance(runtime, dict):
            runtime.pop("batch_size_by_model", None)
    return source_copy == current_copy


def _write_static_receipt(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(manifest["mode"])
    cells = list(manifest["cells"])
    identity_payload = _read_json(Path(manifest["identity_manifest"]))
    _require(
        isinstance(identity_payload, Mapping) and isinstance(identity_payload.get("records"), list),
        "identity manifest payload differs",
    )
    records = validate_identity_manifest(
        identity_payload["records"],
        expected_count=EXPECTED_POPULATION if _is_formal_mode(mode) else 4,
        expected_subjects=EXPECTED_SUBJECTS if _is_formal_mode(mode) else 2,
    )
    if _is_formal_mode(mode):
        validate_cells(cells, formal=True)
        per_model = {key: sum(1 for cell in cells if cell["model_key"] == key) for key in EXPECTED_MODELS}
        _require(per_model == EXPECTED_COUNTS, "formal static panel membership differs")
    else:
        _require(len(cells) >= 1, "debug run has no cells")
        _require(all(cell["model_key"] in EXPECTED_MODELS for cell in cells), "debug model membership differs")
    receipt = {
        "schema_version": "loopscope.phase7.gate-e-static-receipt.v1",
        "status": "PASS",
        "mode": mode,
        "run_root": str(root),
        "cell_count": len(cells),
        "record_count": len(records),
        "subject_count": len({row["subject"] for row in records}),
        "formal_panel_counts": ({key: sum(1 for cell in cells if cell["model_key"] == key) for key in EXPECTED_MODELS} if _is_formal_mode(mode) else None),
        "execution_cell_count": len(_execution_cell_indices(manifest)),
        "retained_cell_count": len(_retained_cell_roots(manifest)),
        "outcome_aggregates_computed": False,
        "created_at": utc_now(),
    }
    _write_new_json(root / "manifest" / "static_preoutcome_receipt.json", receipt)
    return receipt


def prepare_run(
    *,
    mode: str,
    run_root: Path,
    expected_commit: str,
    cache_dir: Optional[str] = None,
    formal_identity_manifest: Optional[Path] = None,
    debug_indices: Optional[Sequence[int]] = None,
    retry_source_run_root: Optional[Path] = None,
    retained_cell_indices: Optional[Sequence[int]] = None,
    card_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create one fresh static panel/membership root before model forwards."""

    _require(mode in {"debug", "formal", "debug_retry", "formal_retry"}, "mode is invalid")
    root = Path(run_root).expanduser().resolve()
    _require(not root.exists(), "Gate E run root must be fresh/write-once")
    git = validate_git(expected_commit, require_clean=True)
    card = load_card(card_path)
    formal_cells = expand_cells(card)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "manifest", "cells", "logs", "resource", "slurm"):
        (root / name).mkdir(exist_ok=False)
    retained_roots: Dict[str, str] = {}
    retry_metadata: Optional[Dict[str, Any]] = None
    if mode == "formal":
        records = build_canonical_test_manifest(cache_dir=cache_dir)
        identity_path = root / "inputs" / "canonical_test_manifest.json"
        _write_new_json(identity_path, {"records": records})
        selected = list(formal_cells)
        execution_indices = list(range(len(selected)))
    elif mode == "debug":
        _require(formal_identity_manifest is not None, "debug preparation requires frozen formal identity manifest")
        source = _read_json(Path(formal_identity_manifest))
        _require(isinstance(source, Mapping) and isinstance(source.get("records"), list), "formal identity manifest is invalid")
        records = select_debug_records(source["records"])
        identity_path = root / "inputs" / "debug_test_manifest.json"
        _write_new_json(identity_path, {"records": records})
        allowed = debug_cell_indices(formal_cells)
        requested = list(debug_indices) if debug_indices is not None else allowed
        _require(bool(requested) and len(set(requested)) == len(requested), "debug cell indices are invalid")
        _require(all(isinstance(index, int) and index in allowed for index in requested), "debug cell is outside frozen representative set")
        selected = [dict(formal_cells[index]) for index in requested]
        for index, cell in enumerate(selected):
            cell["source_panel_index"] = cell["index"]
            cell["index"] = index
        execution_indices = list(range(len(selected)))
    else:
        _require(retry_source_run_root is not None, "retry preparation requires a source run root")
        _require(retained_cell_indices is not None, "retry preparation requires retained cell indices")
        _require(formal_identity_manifest is None and debug_indices is None, "retry preparation cannot redefine membership")
        source_root = _run_root(Path(retry_source_run_root))
        source_manifest = _load_run(source_root)
        _require(_base_mode(str(source_manifest["mode"])) == _base_mode(mode), "retry source mode differs")
        source_card = source_manifest.get("panel_card")
        _require(isinstance(source_card, Mapping) and _retry_cards_compatible(source_card, card), "retry source card differs")
        source_cells = list(source_manifest.get("cells", []))
        source_runtime = source_card.get("runtime")
        legacy_source_batch = not isinstance(source_runtime, Mapping) or "batch_size_by_model" not in source_runtime
        if _is_formal_mode(mode):
            validate_cells(source_cells, formal=True, allow_legacy_batch=legacy_source_batch)
        else:
            _require(source_cells and all(cell.get("model_key") in EXPECTED_MODELS for cell in source_cells), "retry debug cells differ")
        identity_path = Path(source_manifest["identity_manifest"]).expanduser().resolve()
        identity_payload = _read_json(identity_path)
        _require(isinstance(identity_payload, Mapping) and isinstance(identity_payload.get("records"), list), "retry source identity manifest is invalid")
        validate_identity_manifest(
            identity_payload["records"],
            expected_count=EXPECTED_POPULATION if _is_formal_mode(mode) else 4,
            expected_subjects=EXPECTED_SUBJECTS if _is_formal_mode(mode) else 2,
        )
        selected = [dict(cell) for cell in source_cells]
        for cell in selected:
            key = str(cell.get("model_key"))
            _require(key in FROZEN_BATCH_SIZES, "retry cell model differs")
            cell["batch_size"] = FROZEN_BATCH_SIZES[key]
        if _is_formal_mode(mode):
            validate_cells(selected, formal=True)
        retained = sorted(set(retained_cell_indices))
        _require(retained and len(retained) < len(selected), "retry retained cell set is invalid")
        _require(all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(selected) for index in retained), "retry retained cell index is invalid")
        execution_indices = [index for index in range(len(selected)) if index not in set(retained)]
        _require(execution_indices, "retry execution set is empty")
        for index in retained:
            source_cell = _cell_for_index(source_manifest, index)
            current_cell = selected[index]
            _require(source_cell.get("cell_id") == current_cell.get("cell_id"), "retry retained cell identity differs")
            source_cell_root = _validate_retained_cell_metadata(source_root, source_manifest, current_cell)
            retained_roots[str(current_cell["cell_id"])] = str(source_cell_root)
        retry_metadata = {
            "source_run_root": str(source_root),
            "source_mode": str(source_manifest["mode"]),
            "retained_cell_indices": retained,
            "source_expected_commit": str(source_manifest["git"]["expected_commit"]),
        }
    if mode in {"formal", "formal_retry"}:
        for cell in selected:
            cell["source_panel_index"] = cell["index"]
    manifest = {
        "schema_version": RUN_SCHEMA,
        "gate": GATE,
        "mode": mode,
        "run_root": str(root),
        "git": {"expected_commit": str(expected_commit), "validated": git},
        "card_path": str((Path(card_path) if card_path is not None else repository_root() / CARD_RELATIVE).resolve()),
        "panel_card": card,
        "identity_manifest": str(identity_path),
        "cells": selected,
        "execution_cell_indices": execution_indices,
        "retained_cell_roots": retained_roots,
        "retry": retry_metadata,
        "outcome_aggregates_computed": False,
        "created_at": utc_now(),
    }
    _write_new_json(root / "manifest" / "run_manifest.json", manifest)
    _write_static_receipt(root, manifest)
    return {
        "status": "PREPARED",
        "mode": mode,
        "run_root": str(root),
        "cell_count": len(selected),
        "execution_cell_count": len(execution_indices),
        "retained_cell_count": len(retained_roots),
        "identity_manifest": str(identity_path),
    }


def verify_static(run_root: Path) -> Dict[str, Any]:
    root = _run_root(run_root)
    manifest = _load_run(root)
    _require(_read_json(root / "manifest" / "static_preoutcome_receipt.json").get("status") == "PASS", "static receipt is not PASS")
    card = load_card(Path(manifest["card_path"]))
    _require(card == manifest["panel_card"], "current card differs from frozen run card")
    records_payload = _read_json(Path(manifest["identity_manifest"]))
    _require(isinstance(records_payload, Mapping) and isinstance(records_payload.get("records"), list), "identity manifest payload differs")
    expected_count = EXPECTED_POPULATION if _is_formal_mode(str(manifest["mode"])) else 4
    expected_subjects = EXPECTED_SUBJECTS if _is_formal_mode(str(manifest["mode"])) else 2
    records = validate_identity_manifest(records_payload["records"], expected_count=expected_count, expected_subjects=expected_subjects)
    execution_indices = _execution_cell_indices(manifest)
    retained_roots = _retained_cell_roots(manifest)
    cells = list(manifest["cells"])
    expected_retained = {str(cells[index]["cell_id"]) for index in range(len(cells)) if index not in set(execution_indices)}
    _require(set(retained_roots) == expected_retained, "retry retained-cell membership differs")
    if _is_formal_mode(str(manifest["mode"])):
        validate_cells(manifest["cells"], formal=True)
    return {
        "status": "PASS",
        "mode": manifest["mode"],
        "cell_count": len(manifest["cells"]),
        "execution_cell_count": len(execution_indices),
        "retained_cell_count": len(retained_roots),
        "record_count": len(records),
        "subject_count": len({row["subject"] for row in records}),
        "outcome_aggregates_computed": False,
    }


def _parse_slurm_seconds(value: str) -> int:
    text = str(value).strip()
    parts = text.split(":")
    _require(len(parts) in (2, 3) and all(part.isdigit() for part in parts), "Slurm time is invalid")
    if len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
    else:
        hours, minutes, seconds = (int(item) for item in parts)
    _require(minutes < 60 and seconds < 60, "Slurm time fields are invalid")
    return hours * 3600 + minutes * 60 + seconds


def _pool_groups(cells: Sequence[Mapping[str, Any]], parent_count: int) -> List[List[int]]:
    return _pool_groups_from_indices([int(cell["index"]) for cell in cells], parent_count)


def _pool_groups_from_indices(cell_indices: Sequence[int], parent_count: int) -> List[List[int]]:
    _require(isinstance(parent_count, int) and not isinstance(parent_count, bool) and parent_count >= 1, "parent count is invalid")
    indices = list(cell_indices)
    _require(indices and all(isinstance(index, int) and not isinstance(index, bool) and index >= 0 for index in indices), "pool cell indices are invalid")
    _require(len(indices) == len(set(indices)), "pool cell indices are duplicate")
    _require(parent_count <= len(indices), "parent count exceeds cell count")
    groups: List[List[int]] = [[] for _ in range(parent_count)]
    for offset, index in enumerate(indices):
        groups[offset % parent_count].append(index)
    _require(all(group for group in groups), "empty worker-pool group")
    _require(sorted(index for group in groups for index in group) == sorted(indices), "worker-pool membership differs")
    return groups


def _sbatch_text(manifest: Mapping[str, Any], launch: Mapping[str, Any]) -> str:
    scheduler = launch["scheduler"]
    mode = str(manifest["mode"])
    if _base_mode(mode) == "debug":
        _require(scheduler["partition"] == "debug", "debug launch must use debug partition")
        _require(_parse_slurm_seconds(str(scheduler["time_limit"])) < 30 * 60, "debug time must be under 30 minutes")
    job_name = "loopscope-p7-e-%s" % mode
    parent_count = len(launch["pools"])
    lines = [
        "#!/usr/bin/env bash",
        "# Gate E worker-pool launcher generated from a frozen read-only card.",
        "#SBATCH --job-name=%s" % job_name,
        "#SBATCH --partition=%s" % scheduler["partition"],
        "#SBATCH --array=0-%d%%%d" % (parent_count - 1, parent_count),
        "#SBATCH --time=%s" % scheduler["time_limit"],
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=%d" % scheduler["cpus_per_task"],
        "#SBATCH --mem=%s" % scheduler["memory"],
        "#SBATCH --gres=gpu:%s:1" % scheduler["gpu_type"],
        "#SBATCH --no-requeue",
        "#SBATCH --output=%s/slurm/parent-%%A_%%a.out" % shlex.quote(str(manifest["run_root"])),
        "#SBATCH --error=%s/slurm/parent-%%A_%%a.err" % shlex.quote(str(manifest["run_root"])),
    ]
    if scheduler.get("qos"):
        lines.append("#SBATCH --qos=%s" % scheduler["qos"])
    lines.extend(
        [
            "set -euo pipefail",
            "cd %s" % shlex.quote(str(REMOTE_REPO)),
            "export PYTHONDONTWRITEBYTECODE=1",
            "export PYTHONPATH=%s/src" % shlex.quote(str(REMOTE_REPO)),
            "export HF_HOME=%s" % shlex.quote(str(HF_HOME)),
            "export TRANSFORMERS_CACHE=%s" % shlex.quote(str(HF_HOME / "hub")),
            "export HF_DATASETS_CACHE=%s" % shlex.quote(str(HF_DATASETS_CACHE)),
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
            "export HF_HUB_DISABLE_XET=1",
            "export HF_HUB_DISABLE_TELEMETRY=1",
            "unset HF_ENDPOINT || true",
            "export PYTHONNOUSERSITE=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTHONHASHSEED=0",
            "export OMP_NUM_THREADS=%d" % launch["child_cpu_threads"],
            "export MKL_NUM_THREADS=%d" % launch["child_cpu_threads"],
            "export RAYON_NUM_THREADS=%d" % launch["child_cpu_threads"],
            "exec %s %s run-pool --run-root %s --pool-index \"$SLURM_ARRAY_TASK_ID\""
            % (
                shlex.quote(str(AUDITED_VENV / "bin/python")),
                shlex.quote(str(REMOTE_REPO / "scripts/loopscope/run_phase7_gate_e_panel.py")),
                shlex.quote(str(manifest["run_root"])),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_launch(
    *,
    run_root: Path,
    parent_count: int,
    concurrency: int,
    partition: str,
    gpu_type: str,
    time_limit: str,
    cpus_per_task: int,
    memory: str,
    child_cpu_threads: int,
    qos: Optional[str] = None,
) -> Dict[str, Any]:
    root = _run_root(run_root)
    manifest = _load_run(root)
    verify_static(root)
    _require(not (root / "manifest" / "launch_plan.json").exists(), "launch plan already exists")
    _require(not (root / "manifest" / "submission.json").exists(), "run was already submitted")
    _require(not any((root / "cells").iterdir()), "launch plan cannot be altered after cell output exists")
    _require(isinstance(concurrency, int) and concurrency >= 1, "concurrency is invalid")
    _require(isinstance(child_cpu_threads, int) and child_cpu_threads >= 1, "child CPU thread cap is invalid")
    _require(isinstance(cpus_per_task, int) and cpus_per_task >= concurrency * child_cpu_threads, "CPU request cannot support child thread caps")
    _require(str(partition).strip() and str(gpu_type).strip() and str(memory).strip(), "scheduler resource is incomplete")
    _parse_slurm_seconds(str(time_limit))
    execution_indices = _execution_cell_indices(manifest)
    pools = _pool_groups_from_indices(execution_indices, parent_count)
    launch = {
        "schema_version": "loopscope.phase7.gate-e-launch.v1",
        "run_root": str(root),
        "mode": manifest["mode"],
        "max_concurrency_per_gpu": concurrency,
        "child_cpu_threads": child_cpu_threads,
        "pools": [{"pool_index": index, "cell_indices": group} for index, group in enumerate(pools)],
        "scheduler": {
            "partition": str(partition),
            "gpu_type": str(gpu_type),
            "time_limit": str(time_limit),
            "cpus_per_task": cpus_per_task,
            "memory": str(memory),
            "qos": str(qos) if qos else None,
        },
        "resource_policy": {
            "allocator_headroom_fraction": 0.10,
            "bounded_worker_pool": True,
            "shared_model_or_tensor_state": False,
            "batch_size_changed": True,
            "batch_size_by_model": dict(FROZEN_BATCH_SIZES),
        },
        "created_at": utc_now(),
    }
    _write_new_json(root / "manifest" / "launch_plan.json", launch)
    sbatch_path = root / "slurm" / "gate_e_pool.sbatch"
    _require(not sbatch_path.exists(), "Slurm launcher already exists")
    with sbatch_path.open("x", encoding="utf-8") as handle:
        handle.write(_sbatch_text(manifest, launch))
    return {
        "status": "LAUNCH_READY",
        "mode": manifest["mode"],
        "run_root": str(root),
        "parent_count": len(pools),
        "max_concurrency_per_gpu": concurrency,
        "sbatch": str(sbatch_path),
    }


def submit_run(run_root: Path, pool_indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    root = _run_root(run_root)
    manifest = _load_run(root)
    launch = _read_json(root / "manifest" / "launch_plan.json")
    _require(isinstance(launch, Mapping), "launch plan is invalid")
    submission_path = root / "manifest" / "submission.json"
    _require(not submission_path.exists(), "run already has a submission receipt")
    sbatch = root / "slurm" / "gate_e_pool.sbatch"
    _require(sbatch.is_file() and not sbatch.is_symlink(), "Slurm launcher is absent")
    sbatch_binary = "/opt/slurm/bin/sbatch" if Path("/opt/slurm/bin/sbatch").is_file() else "sbatch"
    pool_count = len(launch["pools"])
    if pool_indices is None:
        submitted = list(range(pool_count))
    else:
        submitted = list(pool_indices)
        _require(
            submitted
            and all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < pool_count for index in submitted)
            and len(submitted) == len(set(submitted)),
            "submitted pool indices are invalid",
        )
    command = [sbatch_binary, "--parsable"]
    if submitted != list(range(pool_count)):
        command.append("--array=%s" % ",".join(str(index) for index in submitted))
    command.append(str(sbatch))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise Phase7OutcomeError("sbatch submission failed with exit %d" % completed.returncode)
    raw = completed.stdout.strip()
    job_id = raw.split(";", 1)[0].strip()
    _require(job_id and job_id.isdigit(), "scheduler did not return a real job id")
    receipt = {
        "schema_version": "loopscope.phase7.gate-e-submission.v1",
        "status": "SUBMITTED",
        "mode": manifest["mode"],
        "job_id": job_id,
        "run_root": str(root),
        "parent_count": pool_count,
        "submitted_pool_indices": submitted,
        "max_concurrency_per_gpu": launch["max_concurrency_per_gpu"],
        "submitted_at": utc_now(),
    }
    _write_new_json(submission_path, receipt)
    return receipt


def _require_offline_runtime() -> None:
    values = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    _require(all(str(value).lower() in {"1", "true", "yes"} for value in values.values()), "Gate E runtime must be offline")


def _load_records_from_run(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    payload = _read_json(Path(manifest["identity_manifest"]))
    _require(isinstance(payload, Mapping) and isinstance(payload.get("records"), list), "run identity manifest differs")
    return validate_identity_manifest(
        payload["records"],
        expected_count=EXPECTED_POPULATION if _is_formal_mode(str(manifest["mode"])) else 4,
        expected_subjects=EXPECTED_SUBJECTS if _is_formal_mode(str(manifest["mode"])) else 2,
    )


def _cell_for_index(manifest: Mapping[str, Any], cell_index: int) -> Dict[str, Any]:
    cells = manifest.get("cells")
    _require(isinstance(cells, list) and isinstance(cell_index, int) and 0 <= cell_index < len(cells), "cell index is invalid")
    cell = cells[cell_index]
    _require(isinstance(cell, Mapping) and cell.get("index") == cell_index, "cell identity differs")
    return dict(cell)


def _validate_model_revision(path: Path, cell: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _read_json(path)
    _require(isinstance(payload, Mapping), "model revision payload is invalid")
    revision = str(cell["model_revision"])
    _require(
        payload.get("repo_id") == cell["model_repo"]
        and payload.get("model_commit") == revision
        and payload.get("tokenizer_commit") == revision
        and payload.get("manifest_commit") == revision
        and payload.get("match") is True
        and payload.get("lm_eval_version") == "0.4.11",
        "model/tokenizer revision closure differs",
    )
    return dict(payload)


def _correctness_from_sample(sample: Mapping[str, Any]) -> bool:
    containers = (sample, sample.get("metrics"))
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("acc,none", "acc", "correct"):
            if key not in container:
                continue
            value = container[key]
            if isinstance(value, bool):
                return value
            number = _finite_number(value, "logged sample correctness")
            _require(number in (0.0, 1.0), "logged sample correctness must be zero or one")
            return bool(number)
    raise Phase7OutcomeError("logged sample lacks acc,none correctness")


def project_sanitized_outcomes(result: Mapping[str, Any], expected_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Discard raw lm-eval samples after extracting identity-qualified correctness."""

    samples = result.get("samples")
    _require(isinstance(samples, Mapping), "lm-eval did not return logged samples")
    expected = {(str(row["task_name"]), str(row["test_index"])): row for row in expected_rows}
    observed: Dict[tuple[str, str], bool] = {}
    for task, raw_rows in samples.items():
        _require(isinstance(task, str) and isinstance(raw_rows, list), "lm-eval sample mapping is invalid")
        for sample in raw_rows:
            _require(isinstance(sample, Mapping), "lm-eval sample is invalid")
            doc_id = str(sample.get("doc_id", "")).strip()
            _require(doc_id.isdigit(), "lm-eval sample doc_id is invalid")
            pair = (task, doc_id)
            _require(pair in expected, "lm-eval returned an unexpected identity")
            _require(pair not in observed, "lm-eval returned a duplicate identity")
            observed[pair] = _correctness_from_sample(sample)
    _require(set(observed) == set(expected), "lm-eval logged sample membership is incomplete")
    rows: List[Dict[str, Any]] = []
    for raw in expected_rows:
        key = (str(raw["task_name"]), str(raw["test_index"]))
        rows.append(
            {
                "schema_version": OUTCOME_SCHEMA,
                "canonical_identity": str(raw["canonical_identity"]),
                "subject": str(raw["subject"]),
                "task_name": str(raw["task_name"]),
                "test_index": int(raw["test_index"]),
                "correctness": bool(observed[key]),
            }
        )
    return rows


def _resource_snapshot(
    torch_module: Any,
    *,
    started: float,
    record_count: int,
    status: str,
    error: Optional[BaseException],
    batch_size: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "loopscope.phase7.gate-e-process-resource.v1",
        "status": status,
        "batch_size": batch_size,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "record_count": record_count,
        "throughput_records_per_second": None,
        "oom_detected": False,
        "error_type": type(error).__name__ if error is not None else None,
    }
    if payload["elapsed_seconds"] > 0 and record_count:
        payload["throughput_records_per_second"] = record_count / float(payload["elapsed_seconds"])
    if error is not None:
        payload["oom_detected"] = "out of memory" in str(error).lower()
        if isinstance(error, OSError):
            payload["error_errno"] = error.errno
            payload["error_filename"] = str(error.filename) if error.filename is not None else None
        payload["error_frames"] = [
            {
                "file": Path(frame.filename).name,
                "line": int(frame.lineno),
                "function": str(frame.name),
            }
            for frame in traceback.extract_tb(error.__traceback__)[-8:]
        ]
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            payload.update(
                {
                    "gpu_total_bytes": int(total_bytes),
                    "gpu_free_bytes_at_end": int(free_bytes),
                    "peak_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
                    "peak_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
                }
            )
    except Exception as exc:  # pragma: no cover - remote resource evidence fallback.
        payload["resource_snapshot_error"] = type(exc).__name__
    return payload


@contextlib.contextmanager
def _permission_tolerant_lm_eval_git_probe(evaluator_module: Any) -> Iterable[None]:
    """Keep lm-eval's optional provenance probe from requiring Git on compute nodes."""

    original = evaluator_module.get_git_commit_hash

    def guarded() -> Any:
        try:
            return original()
        except PermissionError:
            return "unavailable"

    evaluator_module.get_git_commit_hash = guarded
    try:
        yield
    finally:
        evaluator_module.get_git_commit_hash = original


def run_cell(*, run_root: Path, cell_index: int) -> Dict[str, Any]:
    """Run exactly one frozen cell and persist only safe outcome rows."""

    root = _run_root(run_root)
    manifest = _load_run(root)
    validate_runtime_git(manifest)
    _require((root / "manifest" / "launch_plan.json").is_file(), "run-cell requires a frozen launch plan")
    _require_offline_runtime()
    cell = _cell_for_index(manifest, cell_index)
    _require(cell_index in _execution_cell_indices(manifest), "retained cell cannot be re-run")
    expected_rows = _load_records_from_run(manifest)
    batch_size = int(cell["batch_size"])
    _require(batch_size == FROZEN_BATCH_SIZES[str(cell["model_key"])], "runtime cell batch differs")
    cell_root = root / "cells" / str(cell["cell_id"])
    _require(not cell_root.exists() and not cell_root.is_symlink(), "cell output path already exists")
    cell_root.mkdir(parents=False, exist_ok=False)
    _write_new_json(
        cell_root / "command.json",
        {
            "schema_version": "loopscope.phase7.gate-e-cell-command.v1",
            "gate": GATE,
            "mode": manifest["mode"],
            "cell": cell,
            "batch_size": batch_size,
            "num_fewshot": 5,
            "tasks": "mmlu" if _is_formal_mode(str(manifest["mode"])) else ",".join(DEBUG_TASKS),
            "limit": None if _is_formal_mode(str(manifest["mode"])) else 2,
            "outcome_aggregates_computed": False,
        },
    )
    started = time.monotonic()
    torch_module: Any = None
    error: Optional[BaseException] = None
    sanitized: List[Dict[str, Any]] = []
    try:
        try:
            import torch
            from lm_eval import evaluator as lm_evaluator
            from tflt.config import LoopConfig
            from tflt.eval_runner import run_lm_eval
        except Exception as exc:  # pragma: no cover - remote-only import path.
            raise Phase7OutcomeError("Gate E runtime dependencies are unavailable") from exc
        torch_module = torch
        _require(torch.cuda.is_available(), "Gate E requires CUDA")
        torch.cuda.reset_peak_memory_stats()
        loop_config = None
        if bool(cell["loop_enabled"]):
            loop_config = LoopConfig.from_window_string(
                model_alias=str(cell["model_repo"]),
                window=str(cell["window_inclusive"]),
                k=int(cell["k"]),
                iteration_mode=str(cell["iteration_mode"]),
                strategy=str(cell["strategy"]),
                alpha=float(cell["alpha"]),
                beta=float(cell["beta"]),
                cache_strategy=str(cell["cache_strategy"]),
                decode_mode=str(cell["decode_mode"]),
            )
        # lm-eval can log aggregate metrics to stdout/stderr.  Keep those
        # transient so the pre-outcome barrier cannot be bypassed through a
        # child log; only the deliberately projected correctness bit crosses
        # the write boundary below.
        with _permission_tolerant_lm_eval_git_probe(lm_evaluator), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            result = run_lm_eval(
                model_repo=str(cell["model_repo"]),
                tasks="mmlu" if _is_formal_mode(str(manifest["mode"])) else ",".join(DEBUG_TASKS),
                output_dir=cell_root,
                limit=None if _is_formal_mode(str(manifest["mode"])) else 2,
                num_fewshot=5,
                batch_size=str(batch_size),
                dtype="bfloat16",
                loop_config=loop_config,
                revision=str(cell["model_revision"]),
                exclusive_writes=False,
            )
        _validate_model_revision(cell_root / "model_revision.json", cell)
        sanitized = project_sanitized_outcomes(result, expected_rows)
        del result
        _write_new_jsonl(cell_root / "outcomes.jsonl", sanitized)
        _write_new_json(
            cell_root / "producer_receipt.json",
            {
                "schema_version": "loopscope.phase7.gate-e-cell-producer.v1",
                "status": "COMPLETED",
                "cell_id": cell["cell_id"],
                "mode": manifest["mode"],
                "batch_size": batch_size,
                "record_count": len(sanitized),
                "subject_count": len({row["subject"] for row in sanitized}),
                "outcome_aggregates_computed": False,
            },
        )
    except BaseException as exc:
        error = exc
        # Do not surface evaluator payloads through the per-cell stderr log.
        raise Phase7OutcomeError("Gate E cell failed: %s" % type(exc).__name__) from exc
    finally:
        if torch_module is not None:
            resource = _resource_snapshot(
                torch_module,
                started=started,
                record_count=len(sanitized),
                status="COMPLETED" if error is None else "FAILED",
                error=error,
                batch_size=batch_size,
            )
        else:
            resource = {
                "schema_version": "loopscope.phase7.gate-e-process-resource.v1",
                "status": "FAILED",
                "batch_size": batch_size,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "record_count": len(sanitized),
                "throughput_records_per_second": None,
                "oom_detected": False,
                "error_type": type(error).__name__ if error is not None else None,
            }
        if not (cell_root / "resource.json").exists():
            _write_new_json(cell_root / "resource.json", resource)
    return {
        "status": "CELL_COMPLETED",
        "cell_id": cell["cell_id"],
        "record_count": len(sanitized),
        "subject_count": len({row["subject"] for row in sanitized}),
    }


def _gpu_sample() -> Dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise Phase7OutcomeError("nvidia-smi resource sampling failed")
    gpus = []
    for row in csv.reader(line for line in completed.stdout.splitlines() if line.strip()):
        _require(len(row) == 6, "nvidia-smi resource row differs")
        try:
            gpus.append(
                {
                    "index": int(row[0].strip()),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": int(row[3].strip()),
                    "memory_used_mib": int(row[4].strip()),
                    "utilization_percent": int(row[5].strip()),
                }
            )
        except ValueError as exc:
            raise Phase7OutcomeError("nvidia-smi resource values differ") from exc
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible and all(part.isdigit() for part in visible):
        allowed = {int(part) for part in visible}
        gpus = [gpu for gpu in gpus if gpu["index"] in allowed]
    elif visible and all(part.startswith("GPU-") for part in visible):
        allowed = set(visible)
        gpus = [gpu for gpu in gpus if gpu["uuid"] in allowed]
    _require(bool(gpus), "nvidia-smi returned no GPU resource rows")
    return {"timestamp": utc_now(), "cuda_visible_devices": visible, "gpus": gpus}


def _start_gpu_sampler(path: Path, *, interval_seconds: float) -> tuple[threading.Event, threading.Thread, List[BaseException]]:
    stop = threading.Event()
    errors: List[BaseException] = []

    def sample_loop() -> None:
        while not stop.is_set():
            try:
                sample = _gpu_sample()
                with path.open("a", encoding="utf-8") as handle:
                    json.dump(sample, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    handle.write("\n")
            except BaseException as exc:  # pragma: no cover - remote-only sampling failure.
                errors.append(exc)
                return
            stop.wait(interval_seconds)

    _require(not path.exists(), "GPU sample path already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=False)
    thread = threading.Thread(target=sample_loop, name="gate-e-gpu-sampler", daemon=True)
    thread.start()
    return stop, thread, errors


def run_pool(*, run_root: Path, pool_index: int) -> Dict[str, Any]:
    """Run one fixed worker pool with isolated child processes on one GPU."""

    root = _run_root(run_root)
    manifest = _load_run(root)
    validate_runtime_git(manifest)
    launch = _read_json(root / "manifest" / "launch_plan.json")
    _require(isinstance(launch, Mapping), "launch plan is invalid")
    pools = launch.get("pools")
    _require(isinstance(pools, list) and isinstance(pool_index, int) and 0 <= pool_index < len(pools), "pool index is invalid")
    pool = pools[pool_index]
    _require(isinstance(pool, Mapping) and pool.get("pool_index") == pool_index, "pool identity differs")
    indices = list(pool.get("cell_indices", []))
    _require(indices and all(isinstance(index, int) for index in indices), "pool cell membership is invalid")
    concurrency = int(launch["max_concurrency_per_gpu"])
    thread_cap = int(launch["child_cpu_threads"])
    pool_root = root / "resource" / ("pool-%02d" % pool_index)
    _require(not pool_root.exists(), "pool resource root already exists")
    pool_root.mkdir(parents=False, exist_ok=False)
    sample_path = pool_root / "gpu_samples.jsonl"
    stop, sampler, sample_errors = _start_gpu_sampler(sample_path, interval_seconds=2.0)
    started = time.monotonic()
    pending = list(indices)
    active: Dict[int, tuple[subprocess.Popen[Any], Any, Any]] = {}
    finished: List[Dict[str, Any]] = []
    child_env = dict(os.environ)
    child_env.update(
        {
            "OMP_NUM_THREADS": str(thread_cap),
            "MKL_NUM_THREADS": str(thread_cap),
            "RAYON_NUM_THREADS": str(thread_cap),
        }
    )
    failure: Optional[BaseException] = None
    try:
        while (pending and failure is None) or active:
            while pending and failure is None and len(active) < concurrency:
                cell_index = pending.pop(0)
                cell = _cell_for_index(manifest, cell_index)
                stdout = (root / "logs" / ("%s.stdout.log" % cell["cell_id"])).open("x", encoding="utf-8")
                stderr = (root / "logs" / ("%s.stderr.log" % cell["cell_id"])).open("x", encoding="utf-8")
                command = [
                    sys.executable,
                    str(repository_root() / "scripts" / "loopscope" / "run_phase7_gate_e_panel.py"),
                    "run-cell",
                    "--run-root",
                    str(root),
                    "--cell-index",
                    str(cell_index),
                ]
                process = subprocess.Popen(command, cwd=str(repository_root()), stdout=stdout, stderr=stderr, env=child_env)
                active[cell_index] = (process, stdout, stderr)
            time.sleep(0.25)
            for cell_index, (process, stdout, stderr) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                cell = _cell_for_index(manifest, cell_index)
                resource_path = root / "cells" / str(cell["cell_id"]) / "resource.json"
                resource = _read_json(resource_path) if resource_path.exists() else {"status": "ABSENT"}
                finished.append(
                    {
                        "cell_index": cell_index,
                        "cell_id": cell["cell_id"],
                        "pid": process.pid,
                        "exit_code": code,
                        "resource_status": resource.get("status"),
                        "oom_detected": resource.get("oom_detected"),
                        "peak_memory_reserved_bytes": resource.get("peak_memory_reserved_bytes"),
                        "elapsed_seconds": resource.get("elapsed_seconds"),
                        "throughput_records_per_second": resource.get("throughput_records_per_second"),
                    }
                )
                del active[cell_index]
                if code != 0 and failure is None:
                    failure = Phase7OutcomeError("one or more pool children failed")
        if sample_errors:
            failure = Phase7OutcomeError("GPU sampler failed: %s" % type(sample_errors[0]).__name__)
        elif not all(item["exit_code"] == 0 for item in finished):
            failure = Phase7OutcomeError("one or more pool children failed")
        elif len(finished) != len(indices):
            failure = Phase7OutcomeError("worker-pool child count differs")
    except BaseException as exc:
        failure = exc
    finally:
        stop.set()
        sampler.join(timeout=10.0)
        for process, stdout, stderr in active.values():
            if process.poll() is None:
                process.terminate()
            stdout.close()
            stderr.close()
    receipt = {
        "schema_version": "loopscope.phase7.gate-e-worker-pool.v1",
        "status": "COMPLETED" if failure is None else "FAILED",
        "mode": manifest["mode"],
        "pool_index": pool_index,
        "max_concurrency_per_gpu": concurrency,
        "child_cpu_threads": thread_cap,
        "cell_indices": indices,
        "unstarted_cell_indices": pending,
        "children": finished,
        "gpu_samples": str(sample_path),
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "outcome_aggregates_computed": False,
    }
    _write_new_json(pool_root / "worker_pool_receipt.json", receipt)
    if failure is not None:
        if isinstance(failure, Phase7OutcomeError):
            raise failure
        raise Phase7OutcomeError("worker pool failed: %s" % type(failure).__name__) from failure
    return receipt


def _outcome_rows_for_cell(root: Path, cell: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    manifest = _load_run(root)
    cell_root = _cell_artifact_root(root, manifest, cell)
    allowed_names = {"command.json", "model_revision.json", "outcomes.jsonl", "producer_receipt.json", "resource.json"}
    observed_names = {path.name for path in cell_root.iterdir()}
    _require(observed_names == allowed_names, "cell artifact barrier differs")
    receipt = _read_json(cell_root / "producer_receipt.json")
    _require(isinstance(receipt, Mapping) and receipt.get("status") == "COMPLETED", "cell producer receipt is not complete")
    _validate_cell_batch_metadata(cell_root, cell)
    _validate_model_revision(cell_root / "model_revision.json", cell)
    rows = _read_jsonl(cell_root / "outcomes.jsonl")
    _require(len(rows) == len(expected), "outcome row count differs")
    normalized: List[Dict[str, Any]] = []
    for observed, source in zip(rows, expected):
        _require(isinstance(observed, Mapping) and set(observed) == {"schema_version", "canonical_identity", "subject", "task_name", "test_index", "correctness"}, "outcome row fields differ")
        _require(observed.get("schema_version") == OUTCOME_SCHEMA, "outcome row schema differs")
        for key in ("canonical_identity", "subject", "task_name", "test_index"):
            _require(observed.get(key) == source.get(key), "outcome membership/order differs")
        _require(isinstance(observed.get("correctness"), bool), "outcome correctness is not boolean")
        normalized.append(dict(observed))
    _require(len({row["canonical_identity"] for row in normalized}) == len(expected), "outcome identity is duplicate")
    return normalized


def verify_canary(run_root: Path, cell_indices: Sequence[int]) -> Dict[str, Any]:
    """Verify a completed formal canary subset without computing outcome aggregates."""

    root = _run_root(run_root)
    manifest = _load_run(root)
    verify_static(root)
    _require(_is_formal_mode(str(manifest["mode"])), "canary verification requires a formal run")
    indices = list(cell_indices)
    _require(
        indices
        and all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(manifest["cells"]) for index in indices)
        and len(indices) == len(set(indices)),
        "canary cell indices are invalid",
    )
    expected = _load_records_from_run(manifest)
    completed = []
    for index in indices:
        cell = _cell_for_index(manifest, index)
        cell_root = _cell_artifact_root(root, manifest, cell)
        source_run_root = cell_root.parent.parent
        _require(
            (source_run_root / "manifest" / "submission.json").is_file(),
            "formal canary cell requires a source submission receipt",
        )
        rows = _outcome_rows_for_cell(root, cell, expected)
        resource = _read_json(cell_root / "resource.json")
        _require(
            resource.get("status") == "COMPLETED"
            and resource.get("oom_detected") is False
            and resource.get("batch_size") in (None, cell["batch_size"])
            and not (resource.get("batch_size") is None and cell["batch_size"] != 16),
            "canary resource receipt differs",
        )
        completed.append(
            {
                "cell_index": index,
                "cell_id": cell["cell_id"],
                "model_key": cell["model_key"],
                "batch_size": cell["batch_size"],
                "record_count": len(rows),
                "subject_count": len({row["subject"] for row in rows}),
                "peak_memory_allocated_bytes": resource.get("peak_memory_allocated_bytes"),
                "peak_memory_reserved_bytes": resource.get("peak_memory_reserved_bytes"),
                "oom_detected": False,
            }
        )
    receipt = {
        "schema_version": "loopscope.phase7.gate-e-canary-verifier.v1",
        "status": "PASS",
        "mode": manifest["mode"],
        "run_root": str(root),
        "cell_count": len(completed),
        "cells": completed,
        "outcome_aggregates_computed": False,
        "created_at": utc_now(),
    }
    _write_new_json(root / "manifest" / "canary_verifier_receipt.json", receipt)
    return receipt


def verify_preoutcome(run_root: Path) -> Dict[str, Any]:
    """Close cell/config/identity completeness without computing any accuracy."""

    root = _run_root(run_root)
    manifest = _load_run(root)
    verify_static(root)
    expected = _load_records_from_run(manifest)
    cells = list(manifest["cells"])
    formal = _is_formal_mode(str(manifest["mode"]))
    if formal:
        validate_cells(cells, formal=True)
        _require((root / "manifest" / "submission.json").is_file(), "formal pre-outcome closure requires a submission receipt")
    completed = []
    for cell in cells:
        rows = _outcome_rows_for_cell(root, cell, expected)
        completed.append(
            {
                "cell_id": cell["cell_id"],
                "model_key": cell["model_key"],
                "batch_size": cell["batch_size"],
                "record_count": len(rows),
                "subject_count": len({row["subject"] for row in rows}),
            }
        )
    receipt = {
        "schema_version": "loopscope.phase7.gate-e-preoutcome-verifier.v1",
        "status": "PASS",
        "mode": manifest["mode"],
        "run_root": str(root),
        "cell_count": len(completed),
        "cells": completed,
        "record_count_per_cell": len(expected),
        "subject_count_per_cell": len({row["subject"] for row in expected}),
        "formal_panel_counts": ({key: sum(1 for cell in cells if cell["model_key"] == key) for key in EXPECTED_MODELS} if formal else None),
        "execution_cell_count": len(_execution_cell_indices(manifest)),
        "retained_cell_count": len(_retained_cell_roots(manifest)),
        "outcome_aggregates_computed": False,
        "created_at": utc_now(),
    }
    path = root / "manifest" / "preoutcome_verifier_receipt.json"
    _write_new_json(path, receipt)
    return receipt


def _percentile(values: Sequence[float], quantile: float) -> float:
    _require(values and 0.0 <= quantile <= 1.0, "percentile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def subject_stratified_paired_bootstrap(
    *, baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> Dict[str, Any]:
    _require(len(baseline) == len(candidate) and len(baseline) > 0, "paired bootstrap rows differ")
    _require(isinstance(replicates, int) and replicates >= 2, "bootstrap replicate count differs")
    groups: Dict[str, List[int]] = {}
    deltas: List[float] = []
    for index, (base, loop) in enumerate(zip(baseline, candidate)):
        _require(base["canonical_identity"] == loop["canonical_identity"] and base["subject"] == loop["subject"], "paired outcome identity differs")
        groups.setdefault(str(base["subject"]), []).append(index)
        deltas.append(float(int(bool(loop["correctness"])) - int(bool(base["correctness"]))))
    _require(len(groups) == EXPECTED_SUBJECTS, "paired bootstrap subject membership differs")
    generator = random.Random(seed)
    draws: List[float] = []
    for _ in range(replicates):
        total = 0.0
        count = 0
        for indices in groups.values():
            for _ in indices:
                total += deltas[generator.choice(indices)]
                count += 1
        draws.append(total / float(count) * 100.0)
    return {
        "method": "subject_stratified_paired_percentile_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "ci_percentage_points": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def _cell_analysis(baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], cell: Mapping[str, Any], analysis: Mapping[str, Any]) -> Dict[str, Any]:
    base_correct = sum(1 for row in baseline if row["correctness"])
    correct = sum(1 for row in candidate if row["correctness"])
    n01 = sum(1 for base, loop in zip(baseline, candidate) if not base["correctness"] and loop["correctness"])
    n10 = sum(1 for base, loop in zip(baseline, candidate) if base["correctness"] and not loop["correctness"])
    paired = subject_stratified_paired_bootstrap(
        baseline=baseline,
        candidate=candidate,
        replicates=int(analysis["bootstrap_replicates"]),
        seed=int(analysis["bootstrap_seed"]),
    )
    return {
        "cell_id": cell["cell_id"],
        "role": cell["role"],
        "window_half_open": cell["window_half_open"],
        "window_inclusive": cell["window_inclusive"],
        "k": cell["k"],
        "cache_strategy": cell["cache_strategy"],
        "iteration_mode": cell["iteration_mode"],
        "strategy": cell["strategy"],
        "alpha": cell["alpha"],
        "beta": cell["beta"],
        "decode_mode": cell["decode_mode"],
        "correct_count": correct,
        "accuracy": correct / float(len(candidate)),
        "delta_percentage_points": (correct - base_correct) / float(len(candidate)) * 100.0,
        "n01": n01,
        "n10": n10,
        "paired_ci": paired,
    }


def build_scientific_projection(run_root: Path) -> Dict[str, Any]:
    root = _run_root(run_root)
    manifest = _load_run(root)
    _require(_is_formal_mode(str(manifest["mode"])), "combined analysis only accepts formal outcomes")
    pre = _read_json(root / "manifest" / "preoutcome_verifier_receipt.json")
    _require(isinstance(pre, Mapping) and pre.get("status") == "PASS", "pre-outcome completeness is not closed")
    expected = _load_records_from_run(manifest)
    per_model: Dict[str, List[Mapping[str, Any]]] = {key: [] for key in EXPECTED_MODELS}
    for cell in manifest["cells"]:
        per_model[str(cell["model_key"])].append(cell)
    summaries = []
    for model in manifest["panel_card"]["models"]:
        key = str(model["model_key"])
        cells = per_model[key]
        baseline_cell = next(cell for cell in cells if not cell["loop_enabled"])
        baseline = _outcome_rows_for_cell(root, baseline_cell, expected)
        baseline_count = sum(1 for row in baseline if row["correctness"])
        contrasts = []
        for cell in cells:
            if not cell["loop_enabled"]:
                continue
            candidate = _outcome_rows_for_cell(root, cell, expected)
            contrasts.append(_cell_analysis(baseline, candidate, cell, manifest["panel_card"]["analysis"]))
        summaries.append(
            {
                "model_key": key,
                "model_repo": model["model_repo"],
                "model_revision": model["model_revision"],
                "selector_terminal": model["selector_terminal"],
                "outcome_interpretation": model["outcome_interpretation"],
                "baseline": {
                    "cell_id": baseline_cell["cell_id"],
                    "correct_count": baseline_count,
                    "accuracy": baseline_count / float(len(baseline)),
                },
                "contrasts": contrasts,
            }
        )
    return {
        "schema_version": "loopscope.phase7.gate-e-combined-analysis.v1",
        "panel_cell_count": len(manifest["cells"]),
        "population": {"records": len(expected), "subjects": len({row["subject"] for row in expected})},
        "primary_metric": "standard_accuracy",
        "multiple_comparison_status": "exploratory_nominal_single_cell_results_non_confirmatory",
        "models": summaries,
    }


def analyze_once(run_root: Path) -> Dict[str, Any]:
    root = _run_root(run_root)
    analysis_path = root / "analysis" / "combined_analysis.json"
    _require(not analysis_path.exists(), "combined analysis already exists")
    scientific = build_scientific_projection(root)
    payload = {
        "schema_version": "loopscope.phase7.gate-e-analysis-artifact.v1",
        "status": "COMPLETED",
        "run_root": str(root),
        "scientific": scientific,
        "created_at": utc_now(),
    }
    _write_new_json(analysis_path, payload)
    return {"status": "ANALYSIS_COMPLETED", "analysis": str(analysis_path), "model_count": len(scientific["models"])}


def verify_analysis(run_root: Path) -> Dict[str, Any]:
    root = _run_root(run_root)
    analysis_path = root / "analysis" / "combined_analysis.json"
    observed = _read_json(analysis_path)
    _require(isinstance(observed, Mapping) and isinstance(observed.get("scientific"), Mapping), "analysis artifact is invalid")
    expected = build_scientific_projection(root)
    _require(observed["scientific"] == expected, "fresh combined analysis recomputation differs")
    receipt = {
        "schema_version": "loopscope.phase7.gate-e-analysis-verifier.v1",
        "status": "PASS",
        "run_root": str(root),
        "analysis": str(analysis_path),
        "independent_recomputation": True,
        "panel_cell_count": expected["panel_cell_count"],
        "created_at": utc_now(),
    }
    _write_new_json(root / "analysis" / "fresh_verifier_receipt.json", receipt)
    return receipt


def resource_summary(run_root: Path) -> Dict[str, Any]:
    """Summarize resource-only evidence without opening outcome rows."""

    root = _run_root(run_root)
    manifest = _load_run(root)
    resources = []
    for cell in manifest["cells"]:
        path = root / "cells" / str(cell["cell_id"]) / "resource.json"
        if path.is_file():
            payload = _read_json(path)
            _require(isinstance(payload, Mapping), "cell resource payload is invalid")
            resources.append({"cell_id": cell["cell_id"], **dict(payload)})
    sample_rows = []
    for path in sorted((root / "resource").glob("pool-*/gpu_samples.jsonl")):
        sample_rows.extend(_read_jsonl(path))
    _require(bool(resources) and bool(sample_rows), "resource evidence is incomplete")
    max_reserved = max(int(row.get("peak_memory_reserved_bytes") or 0) for row in resources)
    totals = [int(gpu["memory_total_mib"]) for sample in sample_rows for gpu in sample.get("gpus", [])]
    used = [int(gpu["memory_used_mib"]) for sample in sample_rows for gpu in sample.get("gpus", [])]
    utilization = [int(gpu["utilization_percent"]) for sample in sample_rows for gpu in sample.get("gpus", [])]
    _require(bool(totals) and bool(used), "GPU samples are malformed")
    _require(max_reserved > 0, "per-process peak reservation was not recorded")
    total_mib = max(totals)
    predicted = max(1, int(math.floor((total_mib * 1024 * 1024 * 0.90) / float(max(1, max_reserved)))))
    return {
        "status": "RESOURCE_SUMMARY",
        "mode": manifest["mode"],
        "process_count": len(resources),
        "max_process_peak_reserved_bytes": max_reserved,
        "gpu_total_mib": total_mib,
        "whole_gpu_peak_used_mib": max(used),
        "whole_gpu_peak_utilization_percent": max(utilization) if utilization else None,
        "allocator_headroom_fraction": 0.10,
        "predicted_max_concurrency_from_single_process_peak": predicted,
    }


__all__ = [
    "Phase7OutcomeError",
    "analyze_once",
    "build_canonical_test_manifest",
    "build_launch",
    "build_scientific_projection",
    "debug_cell_indices",
    "expand_cells",
    "load_card",
    "prepare_run",
    "project_sanitized_outcomes",
    "resource_summary",
    "run_cell",
    "run_pool",
    "select_debug_records",
    "subject_stratified_paired_bootstrap",
    "submit_run",
    "validate_cells",
    "validate_identity_manifest",
    "verify_analysis",
    "verify_preoutcome",
    "verify_static",
]
