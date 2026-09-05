"""Phase 8 identity selection and offline, gold-free MMLU debug projection.

No model forward is performed. Heavy dependencies are imported only by the
remote builder. Validation labels are projected away before any row is read.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

DATASET_REPO = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
SEED = 20260905
SAFE_COLUMNS = ("question", "subject", "choices")


def standard_task_names(task_index):
    """Select the installed standard MMLU YAMLs, excluding named variants."""
    result = []
    for name, entry in task_index.items():
        if not name.startswith("mmlu_") or entry.get("type") != "task":
            continue
        path = Path(entry.get("yaml_path", ""))
        if path.parent.name == "default" and path.parent.parent.name == "mmlu":
            result.append(name)
    return sorted(result)


def identity(subject: str, doc_index: int) -> str:
    return f"{DATASET_REPO}@{DATASET_REVISION}:validation:{subject}:{doc_index}"


def select_calibration(subject_counts: Mapping[str, int], count: int = 512,
                       seed: int = SEED) -> list[dict[str, Any]]:
    """Largest-remainder proportional allocation, then one shared PRNG."""
    total = sum(subject_counts.values())
    if not subject_counts or any(n <= 0 for n in subject_counts.values()) or not 0 < count <= total:
        raise ValueError("calibration count must fit positive subject populations")
    allocation = {s: count * n // total for s, n in subject_counts.items()}
    remaining = count - sum(allocation.values())
    order = sorted(subject_counts, key=lambda s: (-(count * subject_counts[s] % total), s))
    for subject in order[:remaining]:
        allocation[subject] += 1
    rng = random.Random(seed)
    result = []
    for subject in sorted(subject_counts):
        indices = sorted(rng.sample(range(subject_counts[subject]), allocation[subject]))
        result.extend({"identity": identity(subject, i), "subject": subject,
                       "doc_index": i, "split": "validation"} for i in indices)
    return result


def select_debug(rows: Sequence[Mapping[str, Any]],
                 calibration: Sequence[Mapping[str, Any]],
                 models: Sequence[str]) -> list[dict[str, Any]]:
    """First four calibration identities, then four longest validation prompts."""
    if len(calibration) < 4 or len(rows) < 8 or len(models) != 2:
        raise ValueError("debug requires four fit rows, eight validation rows, two tokenizers")
    by_id = {row["identity"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate validation identity")
    fit_ids = [row["identity"] for row in calibration[:4]]
    fit = [dict(by_id[key], role="fit") for key in fit_ids]
    candidates = [row for row in rows if row["identity"] not in fit_ids]
    longest = sorted(candidates, key=lambda row: (
        -max(row["prompt_token_lengths"][model] for model in models),
        row["subject"], row["doc_index"]))[:4]
    return fit + [dict(row, role="verify") for row in longest]


def _load_safe_dataset(task: Any, backend: Any, cache_dir: str) -> tuple[Any, dict[str, Any]]:
    from datasets import DatasetDict, DownloadConfig, DownloadMode, load_dataset

    config = task.config
    if backend._config_value(config, "dataset_path") != DATASET_REPO:
        raise ValueError("task dataset differs from frozen cais/mmlu")
    subject = backend._config_value(config, "dataset_name")
    splits = {}
    evidence = {"subject": subject, "revision": DATASET_REVISION, "splits": {}}
    for split, columns in (("validation", SAFE_COLUMNS), ("dev", SAFE_COLUMNS + ("answer",))):
        raw = load_dataset(DATASET_REPO, subject, revision=DATASET_REVISION,
                           split=split, cache_dir=cache_dir,
                           download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                           download_config=DownloadConfig(local_files_only=True))
        # Offline datasets may fall back to a latest cached builder. Verify the
        # readable source URLs, not cache fingerprints or content digests.
        source_urls = sorted(str(url) for url in
                             (getattr(raw.info, "download_checksums", None) or {}))
        cache_files = [str(item["filename"]) for item in raw.cache_files]
        if source_urls:
            if any(DATASET_REVISION not in url for url in source_urls):
                raise ValueError("cached MMLU source URLs differ from frozen revision")
            revision_evidence = "source_urls"
        else:
            # Historical offline cache records the source version as its builder
            # directory, with no download URL metadata. Check the actual loaded
            # files rather than accepting the generic 'latest cached' message.
            if not cache_files or any(Path(path).parent.name != DATASET_REVISION for path in cache_files):
                raise ValueError("loaded MMLU cache directory differs from frozen revision")
            revision_evidence = "loaded_cache_version_directory"
        # Arrow column projection precedes iteration, indexing, or conversion.
        safe = raw.select_columns(list(columns))
        splits[split] = safe
        evidence["splits"][split] = {
            "count": len(safe), "columns": list(safe.column_names), "source_urls": source_urls,
            "revision_evidence": revision_evidence,
            "cache_files": [str(item["filename"]) for item in safe.cache_files],
        }
    return DatasetDict(splits), evidence


def build_debug_pool(model_config: Mapping[str, Any], cache_dir: str) -> dict[str, Any]:
    """Remote-only builder; exact revisions and existing offline cache required.

    Task construction is scoped to validation/dev: the normal download hook is
    replaced and has_test_docs disabled only inside that construction context.
    An empty answer sentinel supports lm-eval initialization; source labels are
    never materialized, and the sentinel is removed before target iteration.
    """
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    import importlib
    import inspect
    from unittest.mock import patch
    from lm_eval.api.task import ConfigurableTask
    from transformers import AutoTokenizer
    from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend

    backend = LmEvalMMLURendererBackend()
    if backend.distribution_version() != "0.4.11":
        raise ValueError("Phase 8 requires lm_eval 0.4.11")
    models = model_config["models"]
    tokenizers = {entry["model"]: AutoTokenizer.from_pretrained(
        entry["model"], revision=entry["observed_revision"], local_files_only=True)
        for entry in models}
    if len(tokenizers) != 2:
        raise ValueError("Phase 8 requires the two frozen model tokenizers")
    manager = backend._task_manager()
    names = standard_task_names(manager.task_index)
    if len(names) != 57:
        raise ValueError(f"expected 57 MMLU subjects, found {len(names)}")
    sources = []
    safe_datasets = {}

    def safe_download(task: Any, *args: Any, **kwargs: Any) -> None:
        dataset, evidence = _load_safe_dataset(task, backend, cache_dir)
        subject = backend._config_value(task.config, "dataset_name")
        safe_datasets[subject] = dataset
        sources.append(evidence)
        # Some lm-eval initialization paths inspect doc_to_target. This is an
        # explicit empty placeholder, never the validation answer column.
        from datasets import DatasetDict
        task.dataset = DatasetDict({"validation": dataset["validation"].add_column(
            "answer", [""] * len(dataset["validation"])), "dev": dataset["dev"]})

    rows = []
    subject_counts = {}
    task_evidence = []
    with patch.object(ConfigurableTask, "download", safe_download), \
            patch.object(ConfigurableTask, "has_test_docs", lambda self: False):
        for name in names:
            tasks = backend._flatten_tasks(manager.load_task_or_group(name))
            task = tasks[name]
            subject = name[len("mmlu_"):]
            task.dataset = safe_datasets[subject]
            config = task.config
            if backend._config_value(config, "fewshot_split") != "dev":
                raise ValueError("MMLU fewshot split differs from dev")
            if backend._config_value(config, "process_docs") not in (None, ""):
                raise ValueError("unexpected MMLU process_docs")
            fewshot_config = backend._config_value(config, "fewshot_config")
            if not isinstance(fewshot_config, Mapping) or fewshot_config.get("sampler") != "first_n":
                raise ValueError("MMLU must use the standard first_n demonstration sampler")
            dev = task.dataset["dev"]
            if len(dev) != 5:
                raise ValueError("MMLU requires exactly five dev demonstrations")
            task.set_fewshot_seed(SEED)
            validation = task.dataset["validation"]
            subject_counts[subject] = len(validation)
            task_evidence.append({"task_name": name, "yaml_path": str(manager.task_index[name]["yaml_path"]),
                                  "sampler": "first_n", "demo_doc_indices": list(range(5))})
            for index, safe in enumerate(validation):
                if set(safe) != set(SAFE_COLUMNS) or safe["subject"] != subject or len(safe["choices"]) != 4:
                    raise ValueError("validation safe row differs from frozen MMLU schema")
                target = dict(safe, answer="")
                prompt = backend._call_fewshot_context(task, target, random.Random(SEED))
                lengths = {model: len(tokenizer.encode(prompt, add_special_tokens=False))
                           for model, tokenizer in tokenizers.items()}
                rows.append({"identity": identity(subject, index), "subject": subject,
                             "doc_index": index, "split": "validation", "question": safe["question"],
                             "choices": list(safe["choices"]), "prompt": prompt,
                             "prompt_token_lengths": lengths})
    if sum(subject_counts.values()) != 1531:
        raise ValueError("validation population differs from 1531")
    calibration = select_calibration(subject_counts)
    package = importlib.import_module("lm_eval")
    return {
        "schema_version": "loopscope.phase8.debug_pool.v1", "status": "SMOKE_ONLY",
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "split": "validation"},
        "seed": SEED, "calibration_identities": [row["identity"] for row in calibration],
        "rows": select_debug(rows, calibration, list(tokenizers)),
        "source_evidence": {"subject_counts": subject_counts, "datasets": sources,
            "lm_eval_version": "0.4.11", "lm_eval_package": str(package.__file__),
            "renderer_source": inspect.getsourcefile(ConfigurableTask.fewshot_context),
            "renderer_entrypoint": "lm_eval.api.task.ConfigurableTask.fewshot_context",
            "tasks": task_evidence, "tokenizers": [
                {"model": entry["model"], "revision": entry["observed_revision"],
                 "add_special_tokens": False} for entry in models],
            "offline": True, "loaded_splits": ["validation", "dev"],
            "target_gold_loaded": False, "initialization_answer_sentinel": ""},
    }


def write_debug_pool(bundle: Mapping[str, Any], output_dir: str) -> None:
    """Write once; only eight target prompts are persisted."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    identities = {"dataset": bundle["dataset"], "seed": bundle["seed"],
                  "count": len(bundle["calibration_identities"]),
                  "identities": bundle["calibration_identities"]}
    for name, value in (("debug_pool.json", bundle), ("calibration_identities.json", identities)):
        with (root / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
