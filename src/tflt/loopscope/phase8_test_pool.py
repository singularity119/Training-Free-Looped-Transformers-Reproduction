"""Gate D full canonical MMLU test pool; target gold is projected away.

Reuses Gate B/C's standard lm-eval renderer, tokenizer and source evidence
semantics. Heavy dependencies are only imported by the remote-only builder.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from tflt.loopscope.phase8_pool import (
    DATASET_REPO, DATASET_REVISION, SEED, SAFE_COLUMNS, standard_task_names,
)


def test_identity(subject: str, doc_index: int) -> str:
    return f"{DATASET_REPO}@{DATASET_REVISION}:test:{subject}:{doc_index}"


def _load_safe_dataset(task: Any, backend: Any, cache_dir: str) -> tuple[Any, dict[str, Any]]:
    from datasets import DatasetDict, DownloadConfig, DownloadMode, load_dataset

    config = task.config
    if backend._config_value(config, "dataset_path") != DATASET_REPO:
        raise ValueError("task dataset differs from frozen cais/mmlu")
    subject = backend._config_value(config, "dataset_name")
    splits = {}
    evidence = {"subject": subject, "revision": DATASET_REVISION, "splits": {}}
    for split, columns in (("test", SAFE_COLUMNS), ("dev", SAFE_COLUMNS + ("answer",))):
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


def build_test_pool(model_config: Mapping[str, Any], cache_dir: str) -> dict[str, Any]:
    """Remote-only builder; exact revisions and existing offline cache required.

    Task construction is scoped to test/dev: the normal download hook is
    replaced and has_validation_docs disabled only inside that construction context.
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
        # Standard MMLU YAML names test only. Route this builder's initialization
        # to the authorized safe test projection; rendering stays standard.
        task.config.test_split = "test"
        subject = backend._config_value(task.config, "dataset_name")
        safe_datasets[subject] = dataset
        sources.append(evidence)
        # Some lm-eval initialization paths inspect doc_to_target. This is an
        # explicit empty placeholder, never the test answer column.
        from datasets import DatasetDict
        task.dataset = DatasetDict({"test": dataset["test"].add_column(
            "answer", [""] * len(dataset["test"])), "dev": dataset["dev"]})

    rows = []
    subject_counts = {}
    task_evidence = []
    with patch.object(ConfigurableTask, "download", safe_download), \
            patch.object(ConfigurableTask, "has_validation_docs", lambda self: False):
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
            if backend._config_value(fewshot_config, "sampler") != "first_n":
                raise ValueError("MMLU must use the standard first_n demonstration sampler")
            dev = task.dataset["dev"]
            if len(dev) != 5:
                raise ValueError("MMLU requires exactly five dev demonstrations")
            task.set_fewshot_seed(SEED)
            test = task.dataset["test"]
            subject_counts[subject] = len(test)
            task_evidence.append({"task_name": name, "yaml_path": str(manager.task_index[name]["yaml_path"]),
                                  "sampler": "first_n", "demo_doc_indices": list(range(5))})
            for index, safe in enumerate(test):
                if set(safe) != set(SAFE_COLUMNS) or safe["subject"] != subject or len(safe["choices"]) != 4:
                    raise ValueError("test safe row differs from frozen MMLU schema")
                target = dict(safe, answer="")
                prompt = backend._call_fewshot_context(task, target, random.Random(SEED))
                lengths = {model: len(tokenizer.encode(prompt, add_special_tokens=False))
                           for model, tokenizer in tokenizers.items()}
                rows.append({"identity": test_identity(subject, index), "subject": subject,
                             "doc_index": index, "split": "test", "question": safe["question"],
                             "choices": list(safe["choices"]), "prompt": prompt,
                             "prompt_token_lengths": lengths})
    if sum(subject_counts.values()) != 14042:
        raise ValueError("test population differs from 14042")
    package = importlib.import_module("lm_eval")
    return {
        "schema_version": "loopscope.phase8.test_pool.v1",
        "status": "FORMAL_TEST_GOLD_FREE",
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "split": "test"},
        "seed": SEED, "test_identities": [row["identity"] for row in rows],
        "rows": rows,
        "source_evidence": {"subject_counts": subject_counts, "datasets": sources,
            "lm_eval_version": "0.4.11", "lm_eval_package": str(package.__file__),
            "renderer_source": inspect.getsourcefile(ConfigurableTask.fewshot_context),
            "renderer_entrypoint": "lm_eval.api.task.ConfigurableTask.fewshot_context",
            "tasks": task_evidence, "tokenizers": [
                {"model": entry["model"], "revision": entry["observed_revision"],
                 "add_special_tokens": False} for entry in models],
            "offline": True, "loaded_splits": ["test", "dev"],
            "target_gold_loaded": False, "initialization_answer_sentinel": ""},
    }


def validate_test_pool(bundle: Mapping[str, Any]) -> None:
    """Require complete canonical rows with only gold-free fields."""
    if (bundle.get("schema_version") != "loopscope.phase8.test_pool.v1"
            or bundle.get("status") != "FORMAL_TEST_GOLD_FREE"
            or bundle.get("dataset") != {"repo": DATASET_REPO,
                "revision": DATASET_REVISION, "split": "test"}
            or bundle.get("seed") != SEED):
        raise ValueError("frozen formal test pool scope required")
    rows = bundle["rows"]
    counts = bundle["source_evidence"]["subject_counts"]
    if len(counts) != 57 or sum(counts.values()) != 14042 or any(n <= 0 for n in counts.values()):
        raise ValueError("test requires 14042 rows across 57 subjects")
    expected = [(subject, i) for subject in sorted(counts) for i in range(counts[subject])]
    if len(rows) != 14042 or [(r["subject"], r["doc_index"]) for r in rows] != expected:
        raise ValueError("test rows must have complete canonical subject/doc_index order")
    model_names = {item["model"] for item in bundle["source_evidence"]["tokenizers"]}
    if len(model_names) != 2:
        raise ValueError("two frozen tokenizer lengths required")
    fields = {"identity", "subject", "doc_index", "split", "question", "choices",
              "prompt", "prompt_token_lengths"}
    for row in rows:
        if (set(row) != fields or row["split"] != "test"
                or row["identity"] != test_identity(row["subject"], row["doc_index"])
                or len(row["choices"]) != 4
                or set(row["prompt_token_lengths"]) != model_names
                or any(type(n) is not int or n <= 0 for n in row["prompt_token_lengths"].values())):
            raise ValueError("test row identity, safe fields or token lengths invalid")
    if bundle["test_identities"] != [r["identity"] for r in rows]:
        raise ValueError("test identity manifest differs from canonical rows")
    evidence = bundle["source_evidence"]
    if evidence["loaded_splits"] != ["test", "dev"] or evidence["target_gold_loaded"] is not False:
        raise ValueError("gold-free source evidence required")


def write_test_pool(bundle: Mapping[str, Any], output_dir: str) -> None:
    """Write full safe pool and readable identity manifest once in a fresh root."""
    validate_test_pool(bundle)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    identities = {"dataset": bundle["dataset"], "seed": bundle["seed"],
                  "count": len(bundle["rows"]), "identities": bundle["test_identities"]}
    for name, value in (("test_pool.json", bundle), ("test_identities.json", identities)):
        with (root / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
