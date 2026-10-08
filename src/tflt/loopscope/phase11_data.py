"""CPU-only Phase11 input preparation; target gold stays outside model inputs.

Heavy dependencies are lazy imports. Production sources are existing local
parquet or author-archive CSV files with explicit provenance; no model is loaded and
no tokenizer or dataset is silently downloaded.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
GPQA_SEED = 20261008
GPQA_GITHUB_REPO = "idavidrein/gpqa"
GPQA_GITHUB_COMMIT = "56686c06f5e19865c153de0fdb11be3890014df7"
GPQA_ARCHIVE_MEMBER = "dataset/gpqa_main.csv"
GPQA_ARCHIVE_URL = f"https://raw.githubusercontent.com/{GPQA_GITHUB_REPO}/{GPQA_GITHUB_COMMIT}/dataset.zip"
MAX_NEW_TOKENS = 2048
COUNTS = {"arc_challenge": 1172, "mmlu_pro": 12032, "gpqa_main": 448}
DATASETS = {"mmlu_pro": "TIGER-Lab/MMLU-Pro", "gpqa_main": "Idavidrein/gpqa"}
MMLU_SAFE_COLUMNS = ("question_id", "question", "options", "category", "src")
GPQA_COLUMNS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
INSTRUCTION = ("Reason step by step, then end your response with exactly one line in the form "
               "Final answer: (X), where X is the letter of the correct option.")
SCHEMA = "loopscope.phase11.inputs.v1"


def require_revision(value):
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("source revision must be an exact commit")
    return value


def option_labels(options):
    if not 2 <= len(options) <= 26 or any(not isinstance(o, str) or not o for o in options):
        raise ValueError("options must contain 2..26 nonempty strings")
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)])


def mmlu_target(row: Mapping[str, Any]):
    """Only allowlisted lookups, including when a mapping guards gold access."""
    result = {key: row[key] for key in MMLU_SAFE_COLUMNS}
    result["options"] = list(result["options"])
    option_labels(result["options"])
    return result


def normalize_demo_cot(cot: str, answer: str, options: Sequence[str]):
    if answer not in option_labels(options) or not isinstance(cot, str) or not cot.strip():
        raise ValueError("validation demonstration needs CoT and a valid answer")
    # The installed native renderer's terminal answer is replaced, while its
    # reasoning remains intact. A pre-existing Phase11 terminal line is idempotent.
    body = cot.strip().replace("A: Let's think step by step.", "Answer: Let's think step by step.")
    body = re.sub(r"(?:\n|^)?\s*Final answer: \([A-Z]\)\s*$", "", body).rstrip()
    body = re.sub(r"(?:\n|^)?\s*[Tt]he answer is \([A-Z]\)\.?\s*$", "", body).rstrip()
    return body + "\nFinal answer: (" + answer + ")"


def render_question(question: str, options: Sequence[str]):
    labels = option_labels(options)
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a nonempty string")
    return "Question:\n" + question + "\nOptions:\n" + "".join(
        f"{label}. {text.strip()}\n" for label, text in zip(labels, options))


def render_generation_prompt(target, demonstrations=()):
    """Explicit ordered demonstrations; the caller owns native selection."""
    parts = []
    for demo in demonstrations:
        if demo["category"] != target["category"]:
            raise ValueError("MMLU-Pro demonstrations must match target category")
        cot = normalize_demo_cot(demo["cot_content"], demo["answer"], demo["options"])
        if not cot.startswith("Answer:"):
            cot = "Answer: " + cot
        parts.append(render_question(demo["question"], demo["options"]) + cot)
    parts.append(render_question(target["question"], target["options"]) + "Answer: Let's think step by step.")
    return "\n\n".join(parts) + "\n\n" + INSTRUCTION


def chat_input(content, tokenizer, max_length):
    messages = [{"role": "user", "content": content}]
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not ids or len(ids) + MAX_NEW_TOKENS > max_length:
        raise ValueError("complete prompt plus 2048 generation tokens exceeds context; truncation forbidden")
    return {"prompt": prompt, "prompt_token_length": len(ids), "input_ids": ids}


def prepare_mmlu_inputs(targets, demonstrations_by_category, tokenizer, revision, split, max_length):
    rows = []
    for index, raw in enumerate(targets):
        target = mmlu_target(raw)
        demos = demonstrations_by_category[target["category"]]
        if len(demos) != 5:
            raise ValueError("MMLU-Pro requires exactly five explicit validation demonstrations")
        identity = f"{DATASETS['mmlu_pro']}@{revision}:{split}:{index}:{target['question_id']}"
        row = {"identity": identity, "sample_id": identity, "task": "mmlu_pro", "split": split,
               "source_index": index, "source_id": target["question_id"], "subject": target["category"],
               "question": target["question"], "choices": target["options"],
               "choice_labels": option_labels(target["options"]),
               "fewshot_sample_ids": [d["identity"] for d in demos]}
        row.update(chat_input(render_generation_prompt(target, demos), tokenizer, max_length))
        rows.append(row)
    return rows


def gpqa_source_provenance(source):
    """The planning-accepted author archive binding, not an HF revision."""
    if (source.get("source_kind") != "author_github_archive"
            or source.get("github_repo") != GPQA_GITHUB_REPO
            or source.get("github_commit") != GPQA_GITHUB_COMMIT
            or source.get("archive_member") != GPQA_ARCHIVE_MEMBER
            or source.get("revision") != GPQA_GITHUB_COMMIT
            or source.get("split") != "main_csv"
            or source.get("format") != "csv"
            or source.get("name") != "gpqa_main"):
        raise ValueError("GPQA author archive provenance differs from accepted source binding")
    return {"source_kind": "author_github_archive", "source_repo": GPQA_GITHUB_REPO,
            "source_member": GPQA_ARCHIVE_MEMBER, "archive_member": GPQA_ARCHIVE_MEMBER,
            "github_repo": GPQA_GITHUB_REPO,
            "github_commit": GPQA_GITHUB_COMMIT, "archive": "dataset.zip",
            "repo": GPQA_GITHUB_REPO,
            "canonical_order": "fixed GitHub commit archive member CSV row order"}


def prepare_gpqa_inputs(raw_targets, tokenizer, revision, split, max_length, source=None):
    """The sole isolated builder allowed to read GPQA answer-role fields.

    One RNG sequence follows canonical source order. The permutation and gold
    letter go only to the sealed sidecar, never to the model-facing rows.
    """
    rng = random.Random(GPQA_SEED)
    provenance = gpqa_source_provenance(source) if source is not None else None
    if provenance is not None and (revision != source["revision"] or split != source["split"]):
        raise ValueError("GPQA identity revision/split differs from its source binding")
    rows, sealed = [], []
    for index, raw in enumerate(raw_targets):
        original = [raw[key] for key in GPQA_COLUMNS[1:]]
        permutation = list(range(4))
        rng.shuffle(permutation)
        options = [original[i] for i in permutation]
        identity = (f"github:{provenance['source_repo']}@{revision}:{provenance['source_member']}:{index}"
                    if provenance is not None else f"{DATASETS['gpqa_main']}@{revision}:gpqa_main:{split}:{index}")
        target = {"question": raw["Question"], "options": options, "category": "gpqa_main"}
        row = {"identity": identity, "sample_id": identity, "task": "gpqa_main", "split": split,
               "source_index": index, "source_id": index, "subject": "gpqa_main",
               "question": target["question"], "choices": options, "choice_labels": option_labels(options),
               "fewshot_sample_ids": []}
        row.update(chat_input(render_generation_prompt(target), tokenizer, max_length))
        rows.append(row)
        sealed.append({"identity": identity, "gold_letter": "ABCD"[permutation.index(0)],
                       "option_source_indices": permutation})
    return rows, sealed


def length_facts(rows, max_length, generation=True):
    if not rows:
        raise ValueError("input population cannot be empty")
    lengths = [r["prompt_token_length"] for r in rows]
    return {"sample_count": len(rows), "max_prompt_tokens": max(lengths), "min_prompt_tokens": min(lengths),
            "max_input_tokens": max(lengths), "context_length": max_length,
            "max_new_tokens": MAX_NEW_TOKENS if generation else 0,
            "context_enough_without_truncation": max(lengths) + (MAX_NEW_TOKENS if generation else 0) <= max_length,
            "truncated_count": 0}


def read_parquet_source(source, columns):
    """Project at Arrow read before target row indexing or iteration."""
    revision = require_revision(source["revision"])
    files = source["files"]
    urls = source["source_urls"]
    if not files or len(files) != len(urls) or any(f"/{revision}/" not in u for u in urls):
        raise ValueError("each existing parquet needs an exact revision-bound source URL")
    from datasets import Dataset
    import pyarrow.parquet as parquet
    dataset = Dataset(parquet.read_table(files, columns=list(columns)))
    if len(dataset) != source["count"]:
        raise ValueError("source population differs from binding")
    return dataset


def read_gpqa_source(source):
    """Read an already accessible source only inside the isolated GPQA builder.

    CSV parsing necessarily decodes each physical row; only the five authorized
    construction columns leave this reader. The caller's bound split is retained
    separately and is not inferred from a CSV filename.
    """
    revision = require_revision(source["revision"])
    if source["count"] != COUNTS["gpqa_main"]:
        raise ValueError("GPQA source differs from complete bound Main population")
    source_format = source.get("format", "parquet")
    source_kind = source.get("source_kind")
    if source_kind == "author_github_archive":
        gpqa_source_provenance(source)
        files, urls = source["files"], source["source_urls"]
        if len(files) != 1 or urls != [GPQA_ARCHIVE_URL]:
            raise ValueError("GPQA author CSV requires the single accepted archive member and source URL")
    elif source_kind != "huggingface":
        raise ValueError("GPQA source_kind must be explicitly declared")
    if source_format == "parquet":
        return read_parquet_source(source, GPQA_COLUMNS)
    if source_format != "csv":
        raise ValueError("GPQA source format must be explicit csv or parquet")
    files, urls = source["files"], source["source_urls"]
    if source_kind == "huggingface" and (not files or len(files) != len(urls)
            or any(not url.split("?", 1)[0].endswith(f"/{revision}/gpqa_main.csv") for url in urls)):
        raise ValueError("each existing GPQA CSV needs an exact revision-bound Main source URL")
    rows = []
    for filename in files:
        with Path(filename).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not set(GPQA_COLUMNS).issubset(reader.fieldnames):
                raise ValueError("GPQA CSV lacks authorized construction columns")
            for raw in reader:
                rows.append({column: raw[column] for column in GPQA_COLUMNS})
    if len(rows) != source["count"]:
        raise ValueError("GPQA CSV population differs from complete binding")
    return rows


def native_mmlu_indices(validation):
    """lm_eval 0.4.11: category filter preserves order, then first_n=5."""
    selection = {}
    for index, source in enumerate(validation):
        category = source["category"]
        selected = selection.setdefault(category, [])
        if len(selected) < 5:
            selected.append(index)
    if any(len(indices) != 5 for indices in selection.values()):
        raise ValueError("validation category lacks five native demonstrations")
    return selection


def selected_mmlu_demonstrations(validation, selection, revision):
    """Use explicit verified native indices in canonical validation order."""
    result = {}
    for category, indices in selection.items():
        if len(indices) != 5 or len(set(indices)) != 5:
            raise ValueError("native demonstration selection needs five unique indices per category")
        demos = []
        for index in indices:
            source = validation[index]
            if source["category"] != category:
                raise ValueError("selected validation category differs from binding")
            demos.append({key: source[key] for key in ("question", "options", "cot_content", "answer", "category")} |
                         {"identity": f"{DATASETS['mmlu_pro']}@{revision}:validation:{index}:{source['question_id']}"})
        result[category] = demos
    return result


def build_phase11_inputs(binding, tasks):
    """Build existing remote inputs. The binding contains metadata, never questions."""
    if binding["model"] != MODEL or binding["model_revision"] != MODEL_REVISION:
        raise ValueError("Phase11 model/tokenizer binding differs from contract")
    max_length = binding["max_length"]
    if not isinstance(max_length, int) or max_length <= MAX_NEW_TOKENS:
        raise ValueError("explicit sufficient context length is required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    output = {"schema_version": SCHEMA, "model": MODEL, "model_revision": MODEL_REVISION,
              "tokenizer_revision": MODEL_REVISION, "rows_by_task": {}, "sealed_gold_by_task": {}, "task_facts": {}}
    for task in tasks:
        source = binding["tasks"][task]
        source_format = "parquet"
        if source["count"] != COUNTS[task]:
            raise ValueError("binding differs from complete frozen target population")
        if task == "arc_challenge":
            if source["revision"] != ARC_REVISION:
                raise ValueError("ARC revision differs from contract")
            from .phase10_data import build_arc_inputs
            arc = build_arc_inputs({"models": [{"model": MODEL, "revision": MODEL_REVISION}]},
                                   ARC_REVISION, binding["cache_dir"], max_length,
                                   data_dir=source["data_dir"], source_manifest=source["source_manifest"])
            rows = arc["rows_by_split"]["test"]
            facts = arc["length_statistics"]["test"][MODEL]
            if facts["truncated_candidate_count"] or facts["candidate_prefix_inconsistent_count"]:
                raise ValueError("ARC full candidates require identical untruncated prompt prefixes")
            for row in rows:
                row["prompt_token_length"] = row["prompt_token_lengths"][MODEL]
            facts = {**facts, "source_evidence": arc["source_evidence"], "context_length": max_length,
                     "context_enough_without_truncation": True}
        elif task == "mmlu_pro":
            if source["revision"] != MMLU_REVISION or source["split"] != "test":
                raise ValueError("MMLU-Pro revision/split differs from contract")
            targets = read_parquet_source(source, MMLU_SAFE_COLUMNS)
            demo_source = source["validation"]
            if demo_source["revision"] != MMLU_REVISION:
                raise ValueError("validation revision differs from target")
            validation = read_parquet_source(demo_source, MMLU_SAFE_COLUMNS + ("cot_content", "answer"))
            if source["demonstration_indices"] != native_mmlu_indices(validation):
                raise ValueError("bound demonstrations differ from native first_n category order")
            demos = selected_mmlu_demonstrations(validation, source["demonstration_indices"], MMLU_REVISION)
            rows = prepare_mmlu_inputs(targets, demos, tokenizer, MMLU_REVISION, "test", max_length)
            facts = length_facts(rows, max_length) | {"demonstration_indices": source["demonstration_indices"],
                    "demonstration_identities": {category: [d["identity"] for d in items]
                                                 for category, items in demos.items()},
                    "native_demo_selection_evidence": source["native_demo_selection_evidence"],
                    "target_columns_projected_at_read": list(MMLU_SAFE_COLUMNS)}
        elif task == "gpqa_main":
            provenance = gpqa_source_provenance(source)
            source_format = source.get("format", "parquet")
            targets = read_gpqa_source(source)
            rows, sealed = prepare_gpqa_inputs(targets, tokenizer, source["revision"], source["split"], max_length, source=source)
            output["sealed_gold_by_task"][task] = sealed
            facts = length_facts(rows, max_length) | provenance | {"option_shuffle_seed": GPQA_SEED,
                    "option_shuffle_scope": "one RNG sequence in canonical source order",
                    "physical_split_is_not_development_data": True,
                    "gold_and_option_source_indices": "sealed/gpqa_main-gold.jsonl"}
        else:
            raise ValueError("unknown Phase11 task")
        if len(rows) != COUNTS[task] or len({r["identity"] for r in rows}) != COUNTS[task]:
            raise ValueError("input identity population is incomplete or nonunique")
        output["rows_by_task"][task] = rows
        output["task_facts"][task] = {"repo": source.get("repo", DATASETS.get(task, "allenai/ai2_arc")),
            "revision": source["revision"], "split": source["split"], "source_format": source_format,
            "canonical_order": f"{source_format} source file order, then row order",
            "source_files": source.get("files", source.get("data_dir")), "source_urls": source.get("source_urls", source.get("source_manifest")),
            "count": len(rows), **facts}
    return output


def write_phase11_inputs(bundle, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    for task, rows in bundle["rows_by_task"].items():
        with (root / f"{task}.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        pool = {"schema_version": SCHEMA, "task": task, "model": bundle["model"],
                "model_revision": bundle["model_revision"], "tokenizer_revision": bundle["tokenizer_revision"],
                "dataset_revision": bundle["task_facts"][task]["revision"],
                "split": bundle["task_facts"][task]["split"], "scope": "FORMAL_TEST",
                "engineering_synthetic": False, "target_gold_loaded": False, "rows": rows,
                "test_identities": [r["identity"] for r in rows]}
        source_facts = bundle["task_facts"][task]
        pool.update({key: source_facts[key] for key in (
            "source_kind", "source_repo", "source_member", "archive_member", "github_repo", "github_commit", "archive", "source_format")
            if key in source_facts})
        (root / f"{task}-pool.json").write_text(json.dumps(pool, ensure_ascii=False) + "\n", encoding="utf-8")
    if bundle.get("sealed_gold_by_task"):
        sealed_root = root / "sealed"
        sealed_root.mkdir(mode=0o700)
        for task, rows in bundle["sealed_gold_by_task"].items():
            with (sealed_root / f"{task}-gold.jsonl").open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
    facts = {key: value for key, value in bundle.items() if key not in ("rows_by_task", "sealed_gold_by_task")}
    (root / "input_facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
