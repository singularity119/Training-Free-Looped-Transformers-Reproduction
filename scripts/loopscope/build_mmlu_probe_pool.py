#!/usr/bin/env python3
"""Build a deterministic MMLU calibration pool without target gold labels.

The input must be the source projection emitted by ``export_mmlu_renderer.py``.
Before sampling, this builder independently reloads lm-eval and the exact
dataset revision, re-renders the entire projection, and recomputes source,
template, dataset and prompt hashes.  Local question/choice rendering and
self-asserted manifests are deliberately unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.mmlu_renderer import (
    PROJECTION_RECORD_VERSION,
    RendererVerificationError,
    validate_renderer_manifest_payload,
    verify_export_bundle,
)


DEFAULT_SEED = 20260710
SCHEMA_VERSION = "loopscope.probe-pool-manifest.v1"
PROMPT_TEMPLATE_NAME = "lm_eval_mmlu_5shot_verified_projection_v2"
FORBIDDEN_GOLD_KEYS = {
    "answer",
    "answers",
    "answer_idx",
    "answer_index",
    "answer_key",
    "answerkey",
    "correct",
    "correct_answer",
    "correct_choice",
    "correct_index",
    "gold",
    "gold_answer",
    "gold_label",
    "label",
    "labels",
    "target",
}


class PoolBuildError(ValueError):
    """Raised before writing when the calibration input is unsafe or malformed."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed-seed, subject-stratified MMLU probe pool without target "
            "gold labels. Test splits and target gold-label fields are rejected."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Pre-rendered five-shot source projection without target gold labels.",
    )
    parser.add_argument("--output-jsonl", required=True, help="New probe-pool JSONL path.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="New manifest path (default: <output-jsonl>.manifest.json).",
    )
    parser.add_argument("--source", required=True, help="Dataset name/revision description.")
    parser.add_argument("--split", required=True, help="Provably non-test source split.")
    parser.add_argument(
        "--renderer-manifest",
        required=True,
        help="Verified v2 manifest emitted by export_mmlu_renderer.py.",
    )
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE_NAME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the deterministic selection without writing files.",
    )
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    renderer_verifier: Optional[Any] = None,
) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_path.with_name(output_path.name + ".manifest.json")
    )
    renderer_path = Path(args.renderer_manifest).expanduser().resolve()

    _validate_destinations(input_path, output_path, manifest_path, args.dry_run)
    renderer_manifest = _load_renderer_manifest(renderer_path)
    verifier = renderer_verifier or verify_export_bundle
    try:
        verified_manifest = verifier(input_path, renderer_path)
    except RendererVerificationError as exc:
        raise PoolBuildError("renderer authenticity verification failed: %s" % exc) from exc
    if verified_manifest != renderer_manifest:
        raise PoolBuildError("renderer verifier returned evidence for a different manifest")
    if args.prompt_template != PROMPT_TEMPLATE_NAME:
        raise PoolBuildError(
            "phase one only accepts the frozen %s contract" % PROMPT_TEMPLATE_NAME
        )
    if args.count < 1:
        raise PoolBuildError("--count must be positive")
    if not str(args.source).strip():
        raise PoolBuildError("--source must be non-empty and include dataset provenance")
    _reject_test_split(args.split, "--split")
    if args.source != renderer_manifest["dataset"]["source"]:
        raise PoolBuildError("--source must equal the verified renderer dataset source")
    if args.split != renderer_manifest["target_split"]:
        raise PoolBuildError("--split must equal the verified renderer target split")

    raw_bytes = input_path.read_bytes()
    source_records = _load_source_records(
        raw_bytes,
        source=args.source,
        split=args.split,
        renderer_manifest=renderer_manifest,
    )
    if args.count > len(source_records):
        raise PoolBuildError(
            "requested %d records, but the validated source contains only %d"
            % (args.count, len(source_records))
        )

    selected = stratified_sample(source_records, args.count, args.seed)
    pool_bytes = _jsonl_bytes(selected)
    source_counts = Counter(str(item["subject"]) for item in source_records)
    selected_counts = Counter(str(item["subject"]) for item in selected)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.source),
        "split": str(args.split),
        "task_group": "mmlu",
        "num_fewshot": 5,
        "count": len(selected),
        "seed": int(args.seed),
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "test_split_forbidden": True,
        "stratification": {
            "field": "subject",
            "method": "minimum-one-then-proportional-largest-remainder",
            "source_subject_counts": dict(sorted(source_counts.items())),
            "selected_subject_counts": dict(sorted(selected_counts.items())),
        },
        "prompt_template": {
            "name": args.prompt_template,
            "mode": "pre-rendered-only",
            "template_sha256": renderer_manifest["renderer"]["template_sha256"],
            "task_group": "mmlu",
            "num_fewshot": 5,
        },
        "renderer": dict(renderer_manifest["renderer"]),
        "renderer_manifest": {
            "path": str(renderer_path),
            "file_sha256": hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
            "manifest_sha256": renderer_manifest["manifest_sha256"],
        },
        "input": {
            "path": str(input_path),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "validated_count": len(source_records),
        },
        "output": {
            "path": str(output_path),
            "pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        },
        "sample_ids": [item["id"] for item in selected],
        "records": [
            {
                "id": item["id"],
                "prompt_sha256": item["prompt_sha256"],
            }
            for item in selected
        ],
        "rendering_records": [_rendering_manifest_entry(item) for item in selected],
    }
    manifest["render_contract_subset_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest["rendering_records"])
    ).hexdigest()
    manifest["manifest_sha256"] = manifest_sha256(manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new_bytes(output_path, pool_bytes)
    try:
        _write_new_bytes(manifest_path, _pretty_json_bytes(manifest))
    except Exception:
        print(
            "pool was written but manifest creation failed; preserving the new pool for audit: %s"
            % output_path,
            file=sys.stderr,
        )
        raise
    print(str(output_path))
    print(str(manifest_path))
    print(manifest["manifest_sha256"])
    return 0


def _validate_destinations(
    input_path: Path, output_path: Path, manifest_path: Path, dry_run: bool
) -> None:
    if not input_path.is_file():
        raise PoolBuildError("input JSONL does not exist: %s" % input_path)
    if output_path == manifest_path:
        raise PoolBuildError("pool and manifest paths must be different")
    if input_path in (output_path, manifest_path):
        raise PoolBuildError("input and output paths must be different")
    if not dry_run:
        for path in (output_path, manifest_path):
            if path.exists():
                raise FileExistsError("refusing to overwrite existing artifact: %s" % path)


def _load_source_records(
    raw_bytes: bytes,
    source: str,
    split: str,
    renderer_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_prompts = set()
    fixed_demonstrations: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
    fixed_task_names: Dict[str, str] = {}
    renderer = renderer_manifest["renderer"]
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PoolBuildError("input JSONL must be UTF-8") from exc

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PoolBuildError("invalid JSON at input line %d" % line_number) from exc
        if not isinstance(item, Mapping):
            raise PoolBuildError("input line %d must be a JSON object" % line_number)

        if item.get("schema_version") != PROJECTION_RECORD_VERSION:
            raise PoolBuildError(
                "input line %d is not an authenticated renderer projection record"
                % line_number
            )
        expected_record_hash = item.get("projection_record_sha256")
        record_body = {
            key: value for key, value in item.items() if key != "projection_record_sha256"
        }
        if expected_record_hash != hashlib.sha256(canonical_json_bytes(record_body)).hexdigest():
            raise PoolBuildError("projection record SHA256 mismatch at input line %d" % line_number)
        leaked = _forbidden_gold_paths(item)
        if leaked:
            raise PoolBuildError(
                "input line %d contains forbidden gold-label fields: %s"
                % (line_number, ", ".join(leaked))
            )
        record_split = str(item.get("split", split))
        _reject_test_split(record_split, "input line %d split" % line_number)
        if record_split != split:
            raise PoolBuildError(
                "input line %d split %r does not match frozen --split %r"
                % (line_number, record_split, split)
            )
        record_source = str(item.get("source", source))
        if record_source != source:
            raise PoolBuildError(
                "input line %d source %r does not match frozen --source %r"
                % (line_number, record_source, source)
            )
        subject = str(item.get("subject", "")).strip()
        if not subject:
            raise PoolBuildError("input line %d is missing subject" % line_number)

        if item.get("task_group") != "mmlu" or item.get("num_fewshot") != 5:
            raise PoolBuildError(
                "input line %d must declare task_group=mmlu and num_fewshot=5"
                % line_number
            )
        task_name = str(item.get("task_name", "")).strip()
        target_doc_id = str(item.get("target_doc_id", "")).strip()
        if not task_name or not target_doc_id:
            raise PoolBuildError(
                "input line %d needs task_name and target_doc_id" % line_number
            )
        if item.get("uses_target_gold_labels") is not False:
            raise PoolBuildError(
                "input line %d must certify uses_target_gold_labels=false" % line_number
            )
        if item.get("fewshot_answers_present") is not True:
            raise PoolBuildError(
                "input line %d must certify fewshot_answers_present=true" % line_number
            )
        if item.get("template_sha256") != renderer["template_sha256"]:
            raise PoolBuildError(
                "input line %d template hash differs from the renderer manifest" % line_number
            )
        if item.get("render_contract_sha256") != renderer["render_contract_sha256"]:
            raise PoolBuildError(
                "input line %d render contract hash differs from the renderer manifest"
                % line_number
            )
        if item.get("dataset_revision") != renderer["dataset_revision"]:
            raise PoolBuildError(
                "input line %d dataset revision differs from renderer evidence" % line_number
            )
        if item.get("dataset_fingerprint_sha256") != renderer[
            "dataset_fingerprint_sha256"
        ]:
            raise PoolBuildError(
                "input line %d dataset fingerprint differs from renderer evidence"
                % line_number
            )
        if item.get("renderer_source_sha256") != renderer["renderer_source_sha256"]:
            raise PoolBuildError(
                "input line %d renderer source hash differs from verified evidence"
                % line_number
            )
        demonstrations = _validate_demonstrations(item, subject, line_number)
        fewshot_ids = tuple(str(demo["id"]) for demo in demonstrations)
        demo_contract = tuple(
            tuple(str(demo[key]) for key in sorted(demo)) for demo in demonstrations
        )
        previous = fixed_demonstrations.setdefault(subject, demo_contract)
        if previous != demo_contract:
            raise PoolBuildError(
                "input line %d changes the fixed five-shot provenance for subject %s"
                % (line_number, subject)
            )
        previous_task = fixed_task_names.setdefault(subject, task_name)
        if previous_task != task_name:
            raise PoolBuildError(
                "input line %d changes task_name within subject %s" % (line_number, subject)
            )

        rendered = _render_prompt(item, line_number)
        prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if item.get("render_sha256") != prompt_hash:
            raise PoolBuildError("render_sha256 mismatch at input line %d" % line_number)
        supplied_hash = item.get("prompt_sha256")
        if supplied_hash not in (None, prompt_hash):
            raise PoolBuildError("prompt_sha256 mismatch at input line %d" % line_number)
        record_id = str(item.get("id") or item.get("sample_id") or "").strip()
        if not record_id:
            record_id = "%s:%s:%s:%s" % (source, split, subject, prompt_hash[:20])
        if target_doc_id in fewshot_ids:
            raise PoolBuildError(
                "input line %d target_doc_id appears in the five-shot demonstrations"
                % line_number
            )
        if record_id in seen_ids:
            raise PoolBuildError(
                "duplicate sample id at input line %d: %s" % (line_number, record_id)
            )
        if prompt_hash in seen_prompts:
            raise PoolBuildError("duplicate rendered prompt at input line %d" % line_number)
        seen_ids.add(record_id)
        seen_prompts.add(prompt_hash)
        records.append(
            {
                "id": record_id,
                "text": rendered,
                "source": source,
                "split": split,
                "subject": subject,
                "prompt_sha256": prompt_hash,
                "task_group": "mmlu",
                "task_name": task_name,
                "target_doc_id": target_doc_id,
                "target_doc_index": int(item["target_doc_index"]),
                "target_doc_sha256": str(item["target_doc_sha256"]),
                "dataset_fingerprint": str(item["dataset_fingerprint"]),
                "num_fewshot": 5,
                "uses_target_gold_labels": False,
                "fewshot_answers_present": True,
                "renderer": _renderer_record(renderer_manifest, prompt_hash),
                "fewshot_sample_ids": list(fewshot_ids),
                "demonstrations": demonstrations,
            }
        )
    if not records:
        raise PoolBuildError("input JSONL contains no records")
    return records


def _render_prompt(item: Mapping[str, Any], line_number: int) -> str:
    if "question" in item or "choices" in item:
        raise PoolBuildError(
            "input line %d uses the forbidden zero-shot structured rendering path"
            % line_number
        )
    if item.get("text") is None:
        raise PoolBuildError(
            "input line %d must contain a pre-rendered lm-eval five-shot text" % line_number
        )
    rendered = str(item["text"])
    if not rendered.strip():
        raise PoolBuildError("input line %d has empty text" % line_number)
    if not rendered.rstrip().endswith("Answer:"):
        raise PoolBuildError(
            "input line %d pre-rendered text must end at the target Answer: boundary"
            % line_number
        )
    return rendered


def _load_renderer_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise PoolBuildError("renderer manifest does not exist: %s" % path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoolBuildError("renderer manifest is not valid JSON: %s" % path) from exc
    try:
        validate_renderer_manifest_payload(payload)
    except RendererVerificationError as exc:
        raise PoolBuildError(str(exc)) from exc
    return payload


def _validate_demonstrations(
    item: Mapping[str, Any], subject: str, line_number: int
) -> List[Dict[str, str]]:
    values = item.get("demonstrations")
    if not isinstance(values, list) or len(values) != 5:
        raise PoolBuildError("input line %d must list exactly five demonstrations" % line_number)
    normalized = []
    seen = set()
    for index, demo in enumerate(values):
        if not isinstance(demo, Mapping):
            raise PoolBuildError(
                "input line %d demonstration %d must be an object" % (line_number, index)
            )
        leaked = _forbidden_gold_paths(demo)
        if leaked:
            raise PoolBuildError(
                "input line %d demonstration %d contains gold fields: %s"
                % (line_number, index, ", ".join(leaked))
            )
        required = (
            "id",
            "doc_index",
            "source",
            "split",
            "subject",
            "doc_sha256",
            "rendered_sha256",
            "gold_sha256",
        )
        if any(not str(demo.get(key, "")).strip() for key in required):
            raise PoolBuildError(
                "input line %d demonstration %d lacks provenance" % (line_number, index)
            )
        if str(demo["split"]) != "dev":
            raise PoolBuildError(
                "input line %d demonstration %d must come from the frozen dev split"
                % (line_number, index)
            )
        if str(demo["subject"]) != subject:
            raise PoolBuildError(
                "input line %d demonstration %d subject differs from target"
                % (line_number, index)
            )
        demo_id = str(demo["id"])
        if demo_id in seen:
            raise PoolBuildError("input line %d repeats a demonstration ID" % line_number)
        seen.add(demo_id)
        _require_sha256(demo["doc_sha256"], "demonstration doc_sha256")
        _require_sha256(demo["rendered_sha256"], "demonstration rendered_sha256")
        _require_sha256(demo["gold_sha256"], "demonstration gold_sha256")
        normalized.append(
            {
                key: int(demo[key]) if key == "doc_index" else str(demo[key])
                for key in required
            }
        )
    return normalized


def _renderer_record(renderer_manifest: Mapping[str, Any], render_hash: str) -> Dict[str, str]:
    renderer = renderer_manifest["renderer"]
    result = {
        "renderer_entrypoint": str(renderer["renderer_entrypoint"]),
        "lm_eval_version": str(renderer["lm_eval_version"]),
        "renderer_source_sha256": str(renderer["renderer_source_sha256"]),
        "template_sha256": str(renderer["template_sha256"]),
        "render_contract_sha256": str(renderer["render_contract_sha256"]),
        "source_files_sha256": str(renderer["source_files_sha256"]),
        "task_configs_sha256": str(renderer["task_configs_sha256"]),
        "dataset_revision": str(renderer["dataset_revision"]),
        "dataset_fingerprint_sha256": str(renderer["dataset_fingerprint_sha256"]),
        "source_projection_sha256": str(renderer["source_projection_sha256"]),
        "render_sha256": render_hash,
        "renderer_manifest_sha256": str(renderer_manifest["manifest_sha256"]),
    }
    return result


def _rendering_manifest_entry(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "target": {
            "source": item["source"],
            "split": item["split"],
            "subject": item["subject"],
        },
        "task_group": item["task_group"],
        "task_name": item["task_name"],
        "target_doc_id": item["target_doc_id"],
        "target_doc_index": item["target_doc_index"],
        "target_doc_sha256": item["target_doc_sha256"],
        "dataset_fingerprint": item["dataset_fingerprint"],
        "num_fewshot": item["num_fewshot"],
        "uses_target_gold_labels": item["uses_target_gold_labels"],
        "fewshot_answers_present": item["fewshot_answers_present"],
        "renderer": dict(item["renderer"]),
        "fewshot_sample_ids": list(item["fewshot_sample_ids"]),
        "demonstrations": [dict(value) for value in item["demonstrations"]],
        "prompt_sha256": item["prompt_sha256"],
    }


def _require_sha256(value: Any, context: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PoolBuildError("%s must be a lowercase SHA256" % context)
    return digest


def _forbidden_gold_paths(value: Any, path: Tuple[str, ...] = ()) -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            current = path + (str(key),)
            if normalized in FORBIDDEN_GOLD_KEYS:
                found.append(".".join(current))
            found.extend(_forbidden_gold_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_gold_paths(item, path + (str(index),)))
    return sorted(set(found))


def _reject_test_split(split: str, context: str) -> None:
    if "test" in str(split).strip().lower():
        raise PoolBuildError("%s is forbidden for LoopScope calibration: %r" % (context, split))
    if not str(split).strip():
        raise PoolBuildError("%s must be non-empty" % context)


def stratified_sample(
    records: Sequence[Mapping[str, str]], count: int, seed: int
) -> List[Dict[str, str]]:
    """Select deterministic per-subject quotas, then deterministically mix order."""

    groups: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[str(record["subject"])].append(dict(record))
    subjects = sorted(groups)
    if not subjects:
        raise PoolBuildError("cannot stratify an empty record set")

    for subject in subjects:
        groups[subject].sort(key=lambda item: item["id"])
        random.Random(_subject_seed(seed, subject)).shuffle(groups[subject])

    quotas = {subject: 0 for subject in subjects}
    if count >= len(subjects):
        for subject in subjects:
            quotas[subject] = 1
        remaining = count - len(subjects)
    else:
        shuffled_subjects = list(subjects)
        random.Random(seed).shuffle(shuffled_subjects)
        for subject in shuffled_subjects[:count]:
            quotas[subject] = 1
        remaining = 0

    if remaining:
        capacities = {subject: len(groups[subject]) - quotas[subject] for subject in subjects}
        total_capacity = sum(capacities.values())
        if remaining > total_capacity:
            raise PoolBuildError("requested count exceeds stratification capacity")
        ideals = {
            subject: (remaining * capacities[subject] / float(total_capacity))
            for subject in subjects
        }
        for subject in subjects:
            addition = min(capacities[subject], int(ideals[subject]))
            quotas[subject] += addition
        leftover = count - sum(quotas.values())
        order = sorted(
            subjects,
            key=lambda subject: (-(ideals[subject] - int(ideals[subject])), subject),
        )
        while leftover:
            progressed = False
            for subject in order:
                if quotas[subject] < len(groups[subject]):
                    quotas[subject] += 1
                    leftover -= 1
                    progressed = True
                    if not leftover:
                        break
            if not progressed:
                raise PoolBuildError("could not allocate all stratified samples")

    selected = [item for subject in subjects for item in groups[subject][: quotas[subject]]]
    random.Random(seed).shuffle(selected)
    if len(selected) != count:
        raise AssertionError("stratified sampler selected an unexpected record count")
    return selected


def _subject_seed(seed: int, subject: str) -> int:
    digest = hashlib.sha256((str(seed) + "\0" + subject).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for item in records
    ]
    return (("\n".join(lines)) + "\n").encode("utf-8")


def _pretty_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_new_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PoolBuildError, FileExistsError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
