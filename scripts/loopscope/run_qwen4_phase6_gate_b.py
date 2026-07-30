#!/usr/bin/env python3
"""Minimal offline provenance and CPU admission for LoopScope Phase 6 Gate B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tflt.loopscope.phase4_runtime import (  # noqa: E402
    CATEGORIES,
    project_target_row,
    project_test_dataset,
    render_exact_prefix,
    tokenization_metadata,
    validation_demos_by_category,
)
from tflt.loopscope.phase4_schema import (  # noqa: E402
    canonical_identity,
    canonical_json_bytes,
    file_sha256,
    sanitized_content_sha256,
    semantic_sha256,
)
from tflt.loopscope.phase6_anchor import (  # noqa: E402
    ANSWER_REGEX_PATTERN,
    ANSWER_SPAN_EXTRACTOR_SHA256,
    DEFAULT_TEMPLATE_YAML_SHA256,
    GENERATED_ID_TEXT_ALIGNER_SHA256,
    LM_EVAL_UTILS_SHA256,
    MMLU_PRO_YAML_SHA256,
    answer_span_extractor,
)
from tflt.loopscope.phase6_schema import load_json, validate_card  # noqa: E402


EXECUTOR_THREAD_ID = "019fb439-6a3a-71d1-a917-0bca66701cf9"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
CARD_PATH = REPO_ROOT / "configs/loopscope/phase6_pre_answer_v2_card.json"
CARD_SHA256 = "31bd12dc3b58cb7df92c7617047cd86c290c678381ec744b9af73d01945e6f38"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
MODEL_SNAPSHOT = (
    HF_HOME
    / "hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    / MODEL_REVISION
)
DATASET_SNAPSHOT = (
    HF_DATASETS_CACHE
    / "TIGER-Lab___mmlu-pro/default/0.0.0"
    / DATASET_REVISION
)
PHASE4_FREEZE_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-p4b-20260716T204154Z/freeze"
)
PHASE4_PANEL_PATH = PHASE4_FREEZE_ROOT / "phase4_outcome_panel_manifest.json"
PHASE4_SELECTOR_PATH = PHASE4_FREEZE_ROOT / "phase4_selector_freeze.json"
PHASE4_PANEL_SHA256 = "02148127ef7b634106b1ee1044f591012d3543563e94344583480a7c3061eb79"
PHASE4_SELECTOR_SHA256 = "a4cc48a9b82b3d89b962cdefb55b6e4133c08a0aff3431f0de72c3750fc062c8"

DATASET_HASHES = {
    "dataset_info.json": "ce81f7ec00c6701dc737889a3171a9dd32d2521914bc1e5029a3f8bb28df9aab",
    "mmlu-pro-test.arrow": "4c741944e3b53719b433be6e7916e4ab9e8ff33f70e9c365614b4e317c0476bb",
    "mmlu-pro-validation.arrow": "5730c2a9cd07a1ee5f70f4cd1940e43da1df7602c21977b7dc7ef51d496bdf00",
}
PACKAGE_VERSIONS = {
    "lm_eval": "0.4.11",
    "transformers": "4.51.3",
    "tokenizers": "0.21.4",
    "datasets": "5.0.0",
    "torch": "2.3.1+cu121",
    "accelerate": "1.14.0",
}
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
)
KNOWN_REGISTRY = (
    "no-loop",
    "15:18",
    "6:9",
    "10:13",
    "25:28",
    "4:7",
    "5:8",
    "22:25",
)
IMPLEMENTATION_FILES = (
    "src/tflt/loopscope/phase6_anchor.py",
    "src/tflt/loopscope/phase6_acquisition.py",
    "src/tflt/loopscope/phase6_schema.py",
    "scripts/loopscope/run_qwen4_phase6_gate_b.py",
)
FORBIDDEN_PERSISTED_KEYS = {
    "answer",
    "cot_content",
    "gold",
    "label",
    "generated_text",
    "generated_tokens",
    "input_ids",
    "prediction",
    "correctness",
    "accuracy",
    "gain",
    "flip",
    "outcome_values",
    "logits",
    "probabilities",
    "hidden_tensors",
}


class GateBError(RuntimeError):
    """Gate B failed a frozen provenance or information-barrier check."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(command: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(command), cwd=str(cwd), check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateBError(message)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _scan_forbidden_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in FORBIDDEN_PERSISTED_KEYS:
                raise GateBError("forbidden persisted key at %s.%s" % (path, raw_key))
            _scan_forbidden_keys(child, "%s.%s" % (path, raw_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, "%s[%d]" % (path, index))


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"), cwd=REPO_ROOT))
    _require(root == REPO_ROOT, "remote command is not running in the dedicated clone")
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"), cwd=REPO_ROOT)
    commit = _run(("git", "rev-parse", "HEAD"), cwd=REPO_ROOT)
    origin = _run(("git", "rev-parse", "origin/loopscope"), cwd=REPO_ROOT)
    dirty = _run(("git", "status", "--porcelain"), cwd=REPO_ROOT)
    _require(branch == "loopscope", "remote branch differs from loopscope")
    _require(commit == expected_commit, "remote commit differs from expected commit")
    _require(origin == expected_commit, "remote origin/loopscope differs from expected commit")
    _require(dirty == "", "remote dedicated clone is dirty")
    return {
        "repo": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": False,
    }


def _assert_offline_environment() -> None:
    required = {
        "HF_HOME": str(HF_HOME),
        "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
        "HF_DATASETS_CACHE": str(HF_DATASETS_CACHE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    for key, expected in required.items():
        _require(os.environ.get(key) == expected, "offline environment differs: %s" % key)
    _require(not os.environ.get("HF_ENDPOINT"), "HF_ENDPOINT substitution is forbidden")


def _environment_closure() -> Dict[str, Any]:
    observed = {
        name: importlib.metadata.version(name.replace("_", "-"))
        for name in PACKAGE_VERSIONS
    }
    _require(platform.python_version() == "3.11.9", "Python version differs")
    _require(observed == PACKAGE_VERSIONS, "audited package versions differ")
    return {
        "python": platform.python_version(),
        "packages": observed,
        "venv": str(AUDITED_VENV),
        "offline": True,
    }


def _model_tokenizer_closure() -> tuple[Dict[str, Any], Any]:
    from transformers import AutoTokenizer

    _require(MODEL_SNAPSHOT.is_dir(), "pinned model snapshot is missing")
    config_path = MODEL_SNAPSHOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _require(config.get("model_type") == "qwen3", "model type differs")
    _require(config.get("num_hidden_layers") == 36, "decoder layer count differs")
    hidden_size = config.get("hidden_size")
    _require(isinstance(hidden_size, int) and hidden_size > 0, "hidden size is invalid")
    _require(config.get("torch_dtype") == "bfloat16", "model dtype differs")
    tokenizer_hashes = {}
    for name in TOKENIZER_FILES:
        path = MODEL_SNAPSHOT / name
        _require(path.is_file(), "tokenizer file is missing: %s" % name)
        tokenizer_hashes[name] = file_sha256(path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT),
        local_files_only=True,
        trust_remote_code=True,
    )
    return (
        {
            "repo": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "snapshot": str(MODEL_SNAPSHOT),
            "config_sha256": file_sha256(config_path),
            "generation_config_sha256": file_sha256(
                MODEL_SNAPSHOT / "generation_config.json"
            ),
            "tokenizer_hashes": tokenizer_hashes,
            "model_type": config["model_type"],
            "decoder_layers": config["num_hidden_layers"],
            "hidden_size": hidden_size,
            "dtype": config["torch_dtype"],
            "model_weights_loaded": False,
        },
        tokenizer,
    )


def _renderer_closure() -> tuple[Dict[str, Any], Any]:
    from lm_eval.tasks.mmlu_pro import utils

    directory = Path(utils.__file__).resolve().parent
    paths = {
        "utils.py": Path(utils.__file__).resolve(),
        "_mmlu_pro.yaml": directory / "_mmlu_pro.yaml",
        "_default_template_yaml": directory / "_default_template_yaml",
    }
    observed = {name: file_sha256(path) for name, path in paths.items()}
    expected = {
        "utils.py": LM_EVAL_UTILS_SHA256,
        "_mmlu_pro.yaml": MMLU_PRO_YAML_SHA256,
        "_default_template_yaml": DEFAULT_TEMPLATE_YAML_SHA256,
    }
    _require(observed == expected, "lm-eval MMLU-Pro source bundle differs")
    template_text = paths["_default_template_yaml"].read_text(encoding="utf-8")
    _require(ANSWER_REGEX_PATTERN in template_text, "literal custom-extract regex is absent")
    fixture = "Reasoning complete; the answer is (C)."
    span = answer_span_extractor(fixture)
    _require(fixture[span.char_start : span.char_end] == "C", "extractor semantics differ")
    return (
        {
            "lm_eval_version": "0.4.11",
            "source_paths": {name: str(path) for name, path in paths.items()},
            "source_sha256": observed,
            "filter_name": "custom-extract",
            "regex_pattern": ANSWER_REGEX_PATTERN,
            "capture_group": 1,
            "case_sensitive": True,
            "single_match_fixture_option": "C",
            "answer_span_extractor_sha256": ANSWER_SPAN_EXTRACTOR_SHA256,
            "generated_id_text_aligner_sha256": GENERATED_ID_TEXT_ALIGNER_SHA256,
        },
        utils,
    )


def _dataset_closure() -> tuple[Dict[str, Any], Any, Any]:
    from datasets import DownloadMode, load_dataset

    observed_hashes = {
        name: file_sha256(DATASET_SNAPSHOT / name) for name in DATASET_HASHES
    }
    _require(observed_hashes == DATASET_HASHES, "cached dataset component hashes differ")
    kwargs = {
        "path": "TIGER-Lab/MMLU-Pro",
        "revision": DATASET_REVISION,
        "cache_dir": str(HF_DATASETS_CACHE),
        "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
    }
    test = load_dataset(split="test", **kwargs)
    validation = load_dataset(split="validation", **kwargs)
    _require(len(test) == 12032, "cached test split count differs")
    _require(len(validation) == 70, "cached validation split count differs")
    return (
        {
            "repo": "TIGER-Lab/MMLU-Pro",
            "revision": DATASET_REVISION,
            "snapshot": str(DATASET_SNAPSHOT),
            "component_sha256": observed_hashes,
            "split_counts": {"test": len(test), "validation": len(validation)},
        },
        test,
        validation,
    )


def _safe_rows(projected: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for raw in projected:
        target = project_target_row(raw)
        content_hash = sanitized_content_sha256(
            target["question"], target["ordered_options"]
        )
        rows.append(
            {
                **target,
                "safe_content_sha256": content_hash,
                "canonical_identity": canonical_identity(target),
            }
        )
    rows.sort(
        key=lambda row: (
            int(str(row["question_id"]).strip()),
            row["safe_content_sha256"],
        )
    )
    return rows


def _identity_closure(test: Any, validation: Any, tokenizer: Any, renderer: Any) -> Dict[str, Any]:
    rows = _safe_rows(project_test_dataset(test))
    identities = [row["canonical_identity"] for row in rows]
    _require(len(rows) == 12032, "canonical population count differs")
    _require(len(set(identities)) == 12032, "canonical identities are not unique")
    expected_order = sorted(
        rows,
        key=lambda row: (
            int(str(row["question_id"]).strip()),
            row["safe_content_sha256"],
        ),
    )
    _require(rows == expected_order, "canonical identity order differs")

    demos = validation_demos_by_category(validation)
    token_closures = []
    smoke = []
    seen_categories = set()
    for ordinal, row in enumerate(rows):
        target = {
            key: row[key]
            for key in ("question_id", "category", "src", "question", "ordered_options")
        }
        prefix = render_exact_prefix(target, demos[row["category"]], renderer)
        metadata = tokenization_metadata(tokenizer, prefix)
        closure = {
            "ordinal": ordinal,
            "canonical_identity": row["canonical_identity"],
            "category": row["category"],
            "safe_content_sha256": row["safe_content_sha256"],
            "rendered_prefix_sha256": metadata["rendered_prefix_sha256"],
            "rendered_token_ids_sha256": metadata["rendered_token_ids_sha256"],
            "sequence_length": metadata["sequence_length"],
        }
        token_closures.append(closure)
        if row["category"] not in seen_categories and len(smoke) < 4:
            smoke.append(
                {
                    **closure,
                    "question_id": int(str(row["question_id"]).strip()),
                    "src": row["src"],
                }
            )
            seen_categories.add(row["category"])

    _require(len(smoke) == 4, "four distinct-category smoke identities were not found")
    return {
        "canonical_order_rule": "numeric_question_id_then_safe_content_sha256",
        "record_count": len(rows),
        "unique_count": len(set(identities)),
        "missing_count": 0,
        "extra_count": 0,
        "duplicate_count": 0,
        "canonical_order_violation_count": 0,
        "ordered_identity_sha256": semantic_sha256(identities),
        "ordered_safe_token_closure_sha256": semantic_sha256(token_closures),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "smoke_selection_rule": "first_four_distinct_categories_in_canonical_order",
        "smoke_identities": smoke,
        "target_labels_accessed": False,
    }


def _registry_closure(card: Mapping[str, Any]) -> Dict[str, Any]:
    _require(tuple(card["known_outcome_registry"]) == KNOWN_REGISTRY, "registry names differ")
    observed = {
        str(PHASE4_PANEL_PATH): file_sha256(PHASE4_PANEL_PATH),
        str(PHASE4_SELECTOR_PATH): file_sha256(PHASE4_SELECTOR_PATH),
    }
    expected = {
        str(PHASE4_PANEL_PATH): PHASE4_PANEL_SHA256,
        str(PHASE4_SELECTOR_PATH): PHASE4_SELECTOR_SHA256,
    }
    _require(observed == expected, "Phase 4 registry artifact hashes differ")
    return {
        "names": list(KNOWN_REGISTRY),
        "outcome_known_before_phase6": True,
        "artifact_sha256": observed,
        "artifact_sizes": {
            str(path): path.stat().st_size
            for path in (PHASE4_PANEL_PATH, PHASE4_SELECTOR_PATH)
        },
        "artifact_content_opened": False,
    }


def gate_c_runtime_plan(card: Mapping[str, Any]) -> Dict[str, Any]:
    from tflt.loopscope.phase6_acquisition import (
        full_vocabulary_entropy_and_kl,
        validate_generation_replay_id_closure,
    )
    from tflt.loopscope.probe import decoder_boundary_states, find_final_norm

    _require(callable(full_vocabulary_entropy_and_kl), "Phase 6 scalar reducer import failed")
    _require(
        callable(validate_generation_replay_id_closure),
        "Phase 6 replay closure import failed",
    )
    _require(callable(decoder_boundary_states), "decoder boundary helper import failed")
    _require(callable(find_final_norm), "final norm helper import failed")
    return {
        "status": "GATE_C_RUNTIME_PLAN_IMPORTABLE",
        "gate_c_executed": False,
        "model_or_data_loaded": False,
        "model_forward_executed": False,
        "generation_or_replay_executed": False,
        "pass_1": "native_no_loop_deterministic_full_cot_generation",
        "anchor": "unique_span_then_token_strictly_preceding_answer_first_token",
        "pass_2": {
            "input": "prompt_plus_generated_ids_strictly_before_answer_first_token",
            "use_cache": False,
            "output_hidden_states": True,
            "loop_insertions": 0,
        },
        "boundary_capture": "B0_embedding_through_B36_final_norm_pre_hook_raw_input",
        "raw_b36_hook_path": "find_final_norm(model).register_forward_pre_hook",
        "decoder_layers": card["cell"]["decoder_layers"],
        "generation": dict(card["generation"]),
    }


def collect_provenance(expected_commit: str) -> Dict[str, Any]:
    _assert_offline_environment()
    _require(file_sha256(CARD_PATH) == CARD_SHA256, "Phase 6 card byte hash differs")
    card = load_json(CARD_PATH)
    validate_card(card)
    _require(card["cell"]["model_revision"] == MODEL_REVISION, "model revision differs")
    _require(card["cell"]["dataset_revision"] == DATASET_REVISION, "dataset revision differs")
    _require(card["generation"] == {
        "mode": "greedy",
        "do_sample": False,
        "temperature": 0.0,
        "max_gen_toks": 2048,
        "until": "Question:",
        "generation_count": 1,
    }, "generation contract differs")

    git = _git_provenance(expected_commit)
    environment = _environment_closure()
    model, tokenizer = _model_tokenizer_closure()
    renderer_evidence, renderer = _renderer_closure()
    dataset, test, validation = _dataset_closure()
    identity = _identity_closure(test, validation, tokenizer, renderer)
    payload = {
        "schema_version": "loopscope.phase6.gate-b-provenance-manifest.v1",
        "gate": "B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "card_sha256": CARD_SHA256,
        "environment": environment,
        "model_tokenizer": model,
        "dataset": dataset,
        "renderer_filter": renderer_evidence,
        "generation_contract": dict(card["generation"]),
        "known_registry": _registry_closure(card),
        "identity": identity,
        "gate_c_runtime_plan": gate_c_runtime_plan(card),
        "implementation_sha256": {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in IMPLEMENTATION_FILES
        },
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "generation_or_replay_executed": False,
        "outcomes_read": False,
        "target_labels_accessed": False,
        "cuda_or_slurm_used": False,
        "network_downloads": False,
        "gate_c_entered": False,
    }
    _scan_forbidden_keys(payload)
    return payload


def acquire(run_root: Path, expected_commit: str, argv: Sequence[str]) -> Dict[str, Any]:
    _require(not run_root.exists(), "Gate B run root already exists")
    manifest = collect_provenance(expected_commit)
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / "gate_b_manifest.json"
    manifest_file_sha256 = _write_new_json(manifest_path, manifest)
    receipt = {
        "schema_version": "loopscope.phase6.gate-b-receipt.v1",
        "status": "READY_FOR_FRESH_PROCESS_VERIFICATION",
        "created_at_utc": _utc_now(),
        "gate": "B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "git_commit": expected_commit,
        "run_root": str(run_root),
        "manifest_file": manifest_path.name,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_semantic_sha256": semantic_sha256(manifest),
        "record_count": manifest["identity"]["record_count"],
        "ordered_identity_sha256": manifest["identity"]["ordered_identity_sha256"],
        "smoke_identities": manifest["identity"]["smoke_identities"],
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "outcomes_read": False,
        "target_labels_accessed": False,
        "argv": list(argv),
    }
    _scan_forbidden_keys(receipt)
    receipt_file_sha256 = _write_new_json(run_root / "gate_b_receipt.json", receipt)
    return {
        "status": receipt["status"],
        "run_root": str(run_root),
        "manifest_file_sha256": manifest_file_sha256,
        "receipt_file_sha256": receipt_file_sha256,
        "record_count": receipt["record_count"],
        "ordered_identity_sha256": receipt["ordered_identity_sha256"],
        "smoke_identities": receipt["smoke_identities"],
    }


def verify(run_root: Path, expected_commit: str, argv: Sequence[str]) -> Dict[str, Any]:
    manifest_path = run_root / "gate_b_manifest.json"
    receipt_path = run_root / "gate_b_receipt.json"
    _require(manifest_path.is_file() and receipt_path.is_file(), "Gate B artifacts are missing")
    observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _scan_forbidden_keys(observed_manifest)
    _scan_forbidden_keys(observed_receipt)
    _require(
        file_sha256(manifest_path) == observed_receipt["manifest_file_sha256"],
        "manifest file hash differs",
    )
    _require(
        semantic_sha256(observed_manifest) == observed_receipt["manifest_semantic_sha256"],
        "manifest semantic hash differs",
    )
    recomputed = collect_provenance(expected_commit)
    _require(
        canonical_json_bytes(recomputed) == canonical_json_bytes(observed_manifest),
        "fresh-process provenance recomputation differs",
    )
    verifier = {
        "schema_version": "loopscope.phase6.gate-b-verifier-receipt.v1",
        "status": "PASS",
        "verified_at_utc": _utc_now(),
        "gate": "B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "git_commit": expected_commit,
        "manifest_file_sha256": file_sha256(manifest_path),
        "receipt_file_sha256": file_sha256(receipt_path),
        "manifest_semantic_sha256": semantic_sha256(recomputed),
        "record_count": recomputed["identity"]["record_count"],
        "unique_count": recomputed["identity"]["unique_count"],
        "ordered_identity_sha256": recomputed["identity"]["ordered_identity_sha256"],
        "ordered_safe_token_closure_sha256": recomputed["identity"][
            "ordered_safe_token_closure_sha256"
        ],
        "smoke_identities": recomputed["identity"]["smoke_identities"],
        "gate_c_runtime_plan_status": recomputed["gate_c_runtime_plan"]["status"],
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "outcomes_read": False,
        "target_labels_accessed": False,
        "gate_c_entered": False,
        "argv": list(argv),
    }
    _scan_forbidden_keys(verifier)
    verifier_file_sha256 = _write_new_json(
        run_root / "gate_b_verifier_receipt.json", verifier
    )
    return {**verifier, "verifier_file_sha256": verifier_file_sha256}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "verify"):
        child = sub.add_parser(name)
        child.add_argument("--run-root", required=True, type=Path)
        child.add_argument("--expected-commit", required=True)
    dry = sub.add_parser("dry-plan")
    dry.add_argument("--card", type=Path, default=CARD_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "dry-plan":
        _require(file_sha256(args.card) == CARD_SHA256, "Phase 6 card byte hash differs")
        card = load_json(args.card)
        validate_card(card)
        result = gate_c_runtime_plan(card)
    elif args.command == "acquire":
        result = acquire(args.run_root, args.expected_commit, sys.argv)
    else:
        result = verify(args.run_root, args.expected_commit, sys.argv)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
