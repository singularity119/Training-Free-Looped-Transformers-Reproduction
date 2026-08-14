"""Outcome-blind standard MMLU five-shot renderer for Phase 7 smoke jobs.

The manifest builder and renderer deliberately keep validation rows to the safe
``question/choices/subject`` projection.  The only target-like value created
in this module is an in-memory empty sentinel required by lm-eval's standard
final-turn renderer; validation gold is never read.  Prompts and token IDs are
ephemeral and are returned only to the CUDA producer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase7_schema import Phase7ContractError, scan_forbidden_fields


DATASET_REPO = "cais/mmlu"
TASK_GROUP = "mmlu"
VALIDATION_SPLIT = "validation"
FEWSHOT_SPLIT = "dev"
DEMONSTRATION_COUNT = 5
EXPECTED_SUBJECT_COUNT = 57
FEWSHOT_SEED = 20260710
MANIFEST_KEYS = frozenset(
    {"canonical_identity", "subject", "task_name", "validation_index"}
)
SAFE_VALIDATION_COLUMNS = ("question", "choices", "subject")
SAFE_DEV_COLUMNS = ("question", "choices", "subject", "answer")


def canonical_identity(subject: str, validation_index: int) -> str:
    """Return the readable, model-independent MMLU validation identity."""

    if not isinstance(subject, str) or not subject.strip():
        raise Phase7ContractError("subject must be a non-empty string")
    if isinstance(validation_index, bool) or not isinstance(validation_index, int):
        raise Phase7ContractError("validation_index must be an integer")
    if validation_index < 0:
        raise Phase7ContractError("validation_index must be non-negative")
    return "mmlu_%s:validation:%d" % (subject.strip(), validation_index)


def select_smoke_manifest_rows(task_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Select two rows from each of the first two subjects in natural order.

    ``task_rows`` must already be in canonical task/validation order.  This
    function never sorts by batch order or by a model-specific property.
    """

    ordered_rows: List[Dict[str, Any]] = []
    subject_order: List[str] = []
    counts: Dict[str, int] = {}
    for raw in task_rows:
        if not isinstance(raw, Mapping):
            raise Phase7ContractError("canonical task row must be an object")
        scan_forbidden_fields(raw)
        if set(raw) != set(MANIFEST_KEYS):
            raise Phase7ContractError("canonical task row keys differ from manifest contract")
        subject = str(raw["subject"]).strip()
        task_name = str(raw["task_name"]).strip()
        validation_index = raw["validation_index"]
        identity = str(raw["canonical_identity"]).strip()
        if not subject or not task_name or not task_name.startswith("mmlu_"):
            raise Phase7ContractError("canonical task row has invalid subject/task")
        if isinstance(validation_index, bool) or not isinstance(validation_index, int) or validation_index < 0:
            raise Phase7ContractError("canonical task row has invalid validation_index")
        if identity != canonical_identity(subject, validation_index):
            raise Phase7ContractError("canonical identity does not close to subject/index")
        if subject not in counts:
            subject_order.append(subject)
            counts[subject] = 0
        counts[subject] += 1
        ordered_rows.append(
            {
                "canonical_identity": identity,
                "subject": subject,
                "task_name": task_name,
                "validation_index": validation_index,
            }
        )
    eligible_subjects = [subject for subject in subject_order if counts[subject] >= 2][:2]
    if len(eligible_subjects) != 2:
        raise Phase7ContractError("canonical MMLU order did not provide two rows for two subjects")
    # The identity strings are unique, so a simple per-subject counter keeps
    # the selection in canonical natural order.
    selected = []
    selected_counts: Dict[str, int] = {subject: 0 for subject in eligible_subjects}
    for row in ordered_rows:
        subject = str(row["subject"])
        if subject not in selected_counts or selected_counts[subject] >= 2:
            continue
        selected.append(dict(row))
        selected_counts[subject] += 1
    if len(selected) != 4:
        raise Phase7ContractError("canonical MMLU order did not provide two rows per selected subject")
    return selected


def _backend() -> Any:
    from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend

    return LmEvalMMLURendererBackend()


def _task_map() -> Dict[str, Any]:
    backend = _backend()
    manager = backend._task_manager()
    loaded = manager.load_task_or_group(TASK_GROUP)
    tasks = backend._flatten_tasks(loaded)
    tasks = {str(name): task for name, task in tasks.items() if str(name).startswith("mmlu_")}
    if len(tasks) != EXPECTED_SUBJECT_COUNT:
        raise Phase7ContractError("lm-eval MMLU task count differs")
    return dict(sorted(tasks.items()))


def _dataset_cache_root(cache_dir: str | None = None) -> str:
    value = cache_dir or os.environ.get("HF_DATASETS_CACHE")
    if not value:
        raise Phase7ContractError("HF_DATASETS_CACHE or --cache-dir is required")
    return str(Path(value).expanduser())


def _safe_dataset_pair(task: Any, *, cache_dir: str | None) -> Tuple[Any, Any]:
    try:
        from datasets import DownloadMode, load_dataset
        from datasets import DatasetDict
    except Exception as exc:  # pragma: no cover - remote-only dependency path.
        raise Phase7ContractError("Phase 7 renderer requires datasets") from exc
    backend = _backend()
    config = getattr(task, "config", None)
    dataset_path = str(backend._config_value(config, "dataset_path") or "").strip()
    dataset_name = backend._config_value(config, "dataset_name")
    if dataset_path != DATASET_REPO:
        raise Phase7ContractError("MMLU task dataset path differs from frozen dataset")
    kwargs = backend._config_value(config, "dataset_kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise Phase7ContractError("MMLU task dataset_kwargs must be a mapping")
    kwargs = dict(kwargs)
    kwargs.pop("revision", None)
    kwargs.pop("cache_dir", None)
    kwargs.pop("download_mode", None)
    cache_root = _dataset_cache_root(cache_dir)
    validation = load_dataset(
        path=dataset_path,
        name=dataset_name,
        split=VALIDATION_SPLIT,
        cache_dir=cache_root,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        **kwargs,
    )
    dev = load_dataset(
        path=dataset_path,
        name=dataset_name,
        split=FEWSHOT_SPLIT,
        cache_dir=cache_root,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        **kwargs,
    )
    validation_columns = set(getattr(validation, "column_names", ()))
    dev_columns = set(getattr(dev, "column_names", ()))
    if not set(SAFE_VALIDATION_COLUMNS).issubset(validation_columns):
        raise Phase7ContractError("MMLU validation dataset lacks safe columns")
    if not set(SAFE_DEV_COLUMNS).issubset(dev_columns):
        raise Phase7ContractError("MMLU dev dataset lacks safe demonstration columns")
    validation = validation.select_columns(list(SAFE_VALIDATION_COLUMNS))
    dev = dev.select_columns(list(SAFE_DEV_COLUMNS))
    if set(getattr(validation, "column_names", ())) != set(SAFE_VALIDATION_COLUMNS):
        raise Phase7ContractError("validation safe projection retained unexpected fields")
    if set(getattr(dev, "column_names", ())) != set(SAFE_DEV_COLUMNS):
        raise Phase7ContractError("dev safe projection retained unexpected fields")
    task.dataset = DatasetDict({VALIDATION_SPLIT: validation, FEWSHOT_SPLIT: dev})
    return validation, dev


def build_canonical_smoke_rows(*, cache_dir: str | None = None) -> List[Dict[str, Any]]:
    """Read safe validation columns and freeze the common four-row manifest."""

    rows: List[Dict[str, Any]] = []
    for task_name, task in _task_map().items():
        subject = task_name[len("mmlu_") :]
        validation, _ = _safe_dataset_pair(task, cache_dir=cache_dir)
        for validation_index, raw in enumerate(validation):
            if set(raw) != set(SAFE_VALIDATION_COLUMNS):
                raise Phase7ContractError("validation row is outside the safe projection")
            if str(raw["subject"]).strip() != subject:
                raise Phase7ContractError("validation subject differs from task name")
            choices = raw["choices"]
            if not isinstance(raw["question"], str) or not isinstance(choices, Sequence) or len(choices) != 4:
                raise Phase7ContractError("validation safe row shape differs")
            rows.append(
                {
                    "canonical_identity": canonical_identity(subject, validation_index),
                    "subject": subject,
                    "task_name": task_name,
                    "validation_index": validation_index,
                }
            )
    return select_smoke_manifest_rows(rows)


_RENDER_STATE: Dict[str, Tuple[Any, Any, Any]] = {}


def _state_for_subject(task_name: str, *, cache_dir: str | None) -> Tuple[Any, Any, Any]:
    key = "%s|%s" % (task_name, _dataset_cache_root(cache_dir))
    if key not in _RENDER_STATE:
        task = _task_map().get(task_name)
        if task is None:
            raise Phase7ContractError("manifest task is not a frozen MMLU task")
        validation, dev = _safe_dataset_pair(task, cache_dir=cache_dir)
        # ``dev`` is assigned to task.dataset by _safe_dataset_pair; retain the
        # task object so the standard fewshot_context sees both safe splits.
        _RENDER_STATE[key] = (_backend(), task, validation)
    return _RENDER_STATE[key]


def render_mmlu_smoke(row: Mapping[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Render one safe manifest row and return ephemeral tokenizer tensors."""

    if not isinstance(row, Mapping) or set(row) != set(MANIFEST_KEYS):
        raise Phase7ContractError("renderer manifest row keys differ")
    scan_forbidden_fields(row)
    subject = str(row["subject"]).strip()
    task_name = str(row["task_name"]).strip()
    validation_index = row["validation_index"]
    if task_name != "mmlu_" + subject:
        raise Phase7ContractError("renderer task name does not close to subject")
    if str(row["canonical_identity"]) != canonical_identity(subject, validation_index):
        raise Phase7ContractError("renderer identity does not close to subject/index")
    backend, task, validation = _state_for_subject(task_name, cache_dir=None)
    if validation_index >= len(validation):
        raise Phase7ContractError("manifest validation index is outside the safe split")
    raw = dict(validation[validation_index])
    if set(raw) != set(SAFE_VALIDATION_COLUMNS) or str(raw["subject"]).strip() != subject:
        raise Phase7ContractError("renderer validation row is outside the safe projection")
    target_doc = {
        "question": raw["question"],
        "choices": list(raw["choices"]),
        "subject": subject,
        "answer": "",
    }
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - remote-only dependency path.
        raise Phase7ContractError("Phase 7 renderer requires numpy") from exc
    seed = int.from_bytes(hashlib.sha256((str(FEWSHOT_SEED) + "\0" + task_name).encode("utf-8")).digest()[:8], "big")
    prompt, _ = backend._render_with_captured_demos(
        task,
        target_doc,
        task.dataset[FEWSHOT_SPLIT],
        np,
        seed,
    )
    if not isinstance(prompt, str) or not prompt.rstrip().endswith("Answer:"):
        raise Phase7ContractError("standard MMLU renderer did not end at Answer:")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
        raise Phase7ContractError("tokenizer did not return input_ids")
    input_ids = inputs["input_ids"]
    shape = tuple(getattr(input_ids, "shape", ()))
    if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise Phase7ContractError("tokenized prompt must have batch size one")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        probe_index = int(shape[1] - 1)
    else:
        mask = attention_mask[0]
        non_padding = (mask != 0).nonzero(as_tuple=False).reshape(-1)
        if int(non_padding.numel()) < 1:
            raise Phase7ContractError("tokenized prompt has no non-padding token")
        probe_index = int(non_padding[-1].item())
    if probe_index != shape[1] - 1:
        raise Phase7ContractError("MMLU renderer produced padding after Answer:")
    return {
        "inputs": dict(inputs),
        "sequence_length": int(shape[1]),
        "probe_index": probe_index,
    }


__all__ = [
    "FEWSHOT_SEED",
    "MANIFEST_KEYS",
    "build_canonical_smoke_rows",
    "canonical_identity",
    "render_mmlu_smoke",
    "select_smoke_manifest_rows",
]
