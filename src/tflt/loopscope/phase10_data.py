"""Outcome-blind ARC preparation using the pinned native harness renderer.

Heavy libraries are imported only by build_arc_inputs, which is CPU-only and
reads existing local files/cache. Target labels are Arrow-projected away before
any target indexing/iteration. Train demonstration labels alone are retained.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .phase10_adapter import request_arguments, token_metadata

DATASET_REPO = "allenai/ai2_arc"
DATASET_NAME = "ARC-Challenge"
SEED = 20261002
NUM_FEWSHOT = 25
SAFE_COLUMNS = ("id", "question", "choices")
COUNTS = {"train": 1119, "validation": 299, "test": 1172}
SCHEMA = "loopscope.phase10.arc_inputs.v1"


def sample_identity(revision, split, index, source_id):
    return f"{DATASET_REPO}@{revision}:{DATASET_NAME}:{split}:{index}:{source_id}"


def sanitize_target(row: Mapping[str, Any]):
    """Copy only authorized target fields; never inspect label fields."""
    choices = row["choices"]
    labels, texts = list(choices["label"]), list(choices["text"])
    if len(labels) != len(texts) or len(texts) < 2 or len(set(labels)) != len(labels):
        raise ValueError("ARC label/text option counts must agree and labels must be unique")
    if any(not isinstance(s, str) or not s for s in labels + texts):
        raise ValueError("ARC labels and full choice texts must be nonempty strings")
    return {"id": str(row["id"]), "question": str(row["question"]),
            "choices": {"label": labels, "text": texts}}


def train_answer_index(row):
    """Native label lookup, for train demonstrations or later authorized gold only."""
    return list(row["choices"]["label"]).index(row["answerKey"])


def describe_choices(prompt, choices, encoder, max_length, special_ids=(), pad_token_id=None):
    """Tokenize through native TemplateLM._encode_pair, without any model."""
    candidates = []
    for _, continuation in request_arguments(prompt, choices):
        context, answer = encoder._encode_pair(prompt, continuation)
        candidates.append(token_metadata(context, answer, max_length, special_ids, pad_token_id))
    consistent = len({item["prefix_tokens"] for item in candidates}) == 1
    # Tokens are useful internally for the comparison, but length diagnostics
    # need not duplicate whole prompts once per candidate in saved artifacts.
    fields = ("position", "input_length", "context_length", "continuation_length", "left_truncated_tokens")
    return {"context_length": candidates[0]["context_length"],
            "continuation_lengths": [c["continuation_length"] for c in candidates],
            "candidates": [{k: c[k] for k in fields} | {"valid_prefix_count": len(c["valid_positions"])}
                           for c in candidates],
            "candidate_prefix_consistent": consistent}


def prepare_split(task, targets, revision, split, encoders):
    """One native sampler RNG sequence per canonical split, reset to SEED."""
    task.set_fewshot_seed(SEED)
    native_sample = task.sampler.sample
    sampled = []

    def record_sample(*args, **kwargs):
        docs = native_sample(*args, **kwargs)
        sampled[:] = docs
        return docs

    task.sampler.sample = record_sample
    train_index = {str(doc["id"]): i for i, doc in enumerate(task.sampler.df)}
    if len(train_index) != len(task.sampler.df):
        raise ValueError("train source IDs must be unique")
    rows = []
    try:
        for i, source in enumerate(targets):
            safe = sanitize_target(source)
            # Explicit initialization/rendering sentinel, not source answerKey.
            # Native fewshot_context queries doc_to_target but never appends the
            # current target answer for this ordinary multiple-choice task.
            target = dict(safe, answerKey=safe["choices"]["label"][0])
            prompt = task.fewshot_context(target, num_fewshot=NUM_FEWSHOT)
            demo_ids = [sample_identity(revision, "train", train_index[str(d["id"])], d["id"])
                        for d in sampled]
            if len(demo_ids) != NUM_FEWSHOT:
                raise ValueError("native sampler did not return 25 demonstrations")
            texts = safe["choices"]["text"]
            tokenization = {key: describe_choices(prompt, texts, spec["encoder"], spec["max_length"],
                            spec.get("special_ids", ()), spec.get("pad_token_id"))
                            for key, spec in encoders.items()}
            identity = sample_identity(revision, split, i, safe["id"])
            rows.append({"identity": identity, "sample_id": identity, "task": "arc_challenge",
                         "subject": DATASET_NAME, "doc_index": i, "source_index": i,
                         "source_id": safe["id"], "split": split, "question": safe["question"],
                         "choices": texts, "choice_labels": safe["choices"]["label"],
                         "candidate_text_lengths": [len(c) for c in texts], "prompt": prompt,
                         "fewshot_sample_ids": demo_ids, "tokenization": tokenization,
                         "prompt_token_lengths": {k: v["context_length"] for k, v in tokenization.items()}})
    finally:
        task.sampler.sample = native_sample
    return rows


def summarize_lengths(rows, models):
    result = {}
    for model in models:
        entries = [r["tokenization"][model] for r in rows]
        inconsistent = [r["identity"] for r in rows
                        if not r["tokenization"][model]["candidate_prefix_consistent"]]
        candidates = [c for e in entries for c in e["candidates"]]
        result[model] = {"sample_count": len(rows),
            "max_context_tokens": max(e["context_length"] for e in entries),
            "max_continuation_tokens": max(c["continuation_length"] for c in candidates),
            "max_input_tokens": max(c["input_length"] for c in candidates),
            "truncated_candidate_count": sum(c["left_truncated_tokens"] > 0 for c in candidates),
            "candidate_prefix_inconsistent_count": len(inconsistent),
            "candidate_prefix_inconsistent_identities": inconsistent,
            "minimal_inconsistent_sample": next((
                {"identity": r["identity"], "choices": r["choices"], "tokenization": r["tokenization"][model]}
                for r in rows if not r["tokenization"][model]["candidate_prefix_consistent"]), None)}
    return result


def write_arc_inputs(bundle, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    for split, rows in bundle["rows_by_split"].items():
        with (root / f"ARC-{split}.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        demos = [{"identity": r["identity"], "fewshot_sample_ids": r["fewshot_sample_ids"]} for r in rows]
        (root / f"ARC-{split}-demo-identities.json").write_text(json.dumps(demos, indent=2) + "\n", encoding="utf-8")
        # Bundle supports the existing accuracy-runner's rows convention.
        pool = {k: v for k, v in bundle.items() if k != "rows_by_split"}
        pool.update(rows=rows, split=split, test_identities=[r["identity"] for r in rows])
        (root / f"ARC-{split}-pool.json").write_text(json.dumps(pool, ensure_ascii=False) + "\n", encoding="utf-8")
    facts = {k: v for k, v in bundle.items() if k != "rows_by_split"}
    (root / "arc_input_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_arc_inputs(model_config, dataset_revision, cache_dir, max_length, data_dir=None, source_manifest=None):
    """Remote CPU-only builder: exact source revision and cached tokenizers."""
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    from importlib.metadata import version
    if version("lm_eval") != "0.4.11":
        raise ValueError("ARC native adapter requires lm_eval 0.4.11")
    from datasets import DatasetDict, DownloadConfig, DownloadMode, load_dataset
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config
    from transformers import AutoTokenizer
    if len(dataset_revision) != 40 or any(c not in "0123456789abcdef" for c in dataset_revision):
        raise ValueError("ARC dataset revision must be exact commit")
    if max_length <= 0:
        raise ValueError("explicit proposed max_length must be positive")
    sources, safe_splits = {}, {}
    for split in ("train", "test", "validation"):
        columns = list(SAFE_COLUMNS) + (["answerKey"] if split == "train" else [])
        if data_dir:
            files = sorted(str(p) for p in Path(data_dir).glob(f"{split}*.parquet"))
            if not files:
                raise FileNotFoundError(f"missing local ARC {split} parquet")
            # Read only authorized Arrow columns at the file boundary.
            # Target answerKey is never materialized into a Dataset.
            from datasets import Dataset
            import pyarrow.parquet as parquet
            raw = Dataset(parquet.read_table(files, columns=columns))
            urls = (source_manifest or {}).get(split, [])
            if not urls or any(f"/{dataset_revision}/" not in u or f"{DATASET_NAME}/{split}" not in u for u in urls):
                raise ValueError("local parquet needs revision-bound source URLs for each split")
        else:
            raw = load_dataset(DATASET_REPO, DATASET_NAME, revision=dataset_revision, split=split,
                               cache_dir=cache_dir, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                               download_config=DownloadConfig(local_files_only=True))
            urls = sorted(str(u) for u in (getattr(raw.info, "download_checksums", None) or {}))
            if not urls or any(dataset_revision not in u for u in urls):
                raise ValueError("cached ARC source URLs do not establish exact revision")
        # This is the first access to target payloads: project label away first.
        safe = raw.select_columns(columns)
        if len(safe) != COUNTS[split]:
            raise ValueError(f"ARC {split} count differs from frozen full population")
        safe_splits[split] = safe
        sources[split] = {"count": len(safe), "columns": columns, "source_urls": urls,
                          "cache_files": [f["filename"] for f in safe.cache_files],
                          "parquet_columns_projected_at_read": bool(data_dir)}
    dataset = DatasetDict(safe_splits)
    manager = TaskManager()
    yaml_path = manager.task_index["arc_challenge"]["yaml_path"]
    config = load_yaml_config(yaml_path, mode="full")

    class SafeARC(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = dataset

        def doc_to_target(self, doc, *args, **kwargs):
            if "answerKey" not in doc:
                return 0  # initialization sentinel; projected target label absent
            return super().doc_to_target(doc, *args, **kwargs)

    task = SafeARC(config=config)
    if (task.config.training_split != "train" or task.fewshot_cfg.split not in (None, "train")
            or task.config.target_delimiter != " " or task.config.fewshot_delimiter != "\n\n"
            or task.config.doc_to_text != "Question: {{question}}\nAnswer:"
            or task.config.doc_to_choice != "{{choices.text}}"
            or task.config.doc_to_target != "{{choices.label.index(answerKey)}}"
            or type(task.sampler).__name__ != "ContextSampler"):
        raise ValueError("installed native ARC task differs from verified v0.4.11 semantics")
    if [d["id"] for d in task.sampler.df] != list(dataset["train"]["id"]):
        raise ValueError("native ARC demonstration pool differs from canonical train order")

    class Encoder:
        backend = "causal"
        _encode_pair = TemplateLM._encode_pair
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def tok_encode(self, text, **kwargs):
            return self.tokenizer.encode(text, add_special_tokens=False)

    encoders, tokenizer_facts = {}, []
    for spec in model_config["models"]:
        model, revision = spec["model"], spec.get("observed_revision", spec.get("revision"))
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
        encoders[model] = {"encoder": Encoder(tokenizer), "max_length": max_length,
                           "special_ids": tokenizer.all_special_ids, "pad_token_id": tokenizer.pad_token_id}
        tokenizer_facts.append({"model": model, "revision": revision, "max_length": max_length,
                                "add_special_tokens": False, "path": tokenizer.name_or_path})
    rows = {split: prepare_split(task, dataset[split], dataset_revision, split, encoders)
            for split in ("test", "validation")}
    return {"schema_version": SCHEMA, "status": "ARC_TARGET_GOLD_FREE",
            "dataset": {"repo": DATASET_REPO, "name": DATASET_NAME, "revision": dataset_revision},
            "seed": SEED, "num_fewshot": NUM_FEWSHOT, "rows_by_split": rows,
            "source_evidence": {"offline": True, "lm_eval_version": "0.4.11", "yaml_path": str(yaml_path),
                "sampler": "ContextSampler", "sampler_seed_reset_per_split": True,
                "sampler_scope": "one native RNG sequence across canonical target order within split",
                "target_gold_loaded": False, "target_gold_projected_before_iteration": True,
                "target_label_initialization_sentinel": "index 0 only; not source gold",
                "train_demo_gold_allowed": True, "sources": sources, "tokenizers": tokenizer_facts,
                "tokenization": "TemplateLM._encode_pair; causal, add_special_tokens=False",
                "acc_norm": "loglikelihood / len(raw choice text), Python characters; first-index tie"},
            "length_statistics": {split: summarize_lengths(items, encoders) for split, items in rows.items()}}
