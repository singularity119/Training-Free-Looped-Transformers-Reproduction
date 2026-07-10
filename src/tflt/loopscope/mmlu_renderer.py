"""Authentic lm-eval MMLU five-shot projection export and verification.

The public helpers in this module are standard-library only until the real
backend is instantiated.  The production backend imports ``lm_eval``,
``datasets`` and ``numpy`` lazily, reloads the exact dataset revision, calls the
task's real ``fewshot_context`` renderer, and records every source/config hash
needed to repeat that rendering.  Tests inject a deterministic fake backend;
there is deliberately no CLI trust/skip switch.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


LM_EVAL_VERSION = "0.4.11"
RENDERER_SCHEMA_VERSION = "loopscope.mmlu-renderer-manifest.v2"
PROJECTION_RECORD_VERSION = "loopscope.mmlu-renderer-projection-record.v2"
RENDER_CONTRACT_VERSION = "loopscope.mmlu-5shot-render.v2"
FORBIDDEN_TARGET_KEYS = {
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


class RendererVerificationError(ValueError):
    """Raised when renderer provenance cannot be independently reproduced."""


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


def projection_jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [canonical_json_bytes(record).decode("utf-8") for record in records]
    if not lines:
        raise RendererVerificationError("renderer projection must contain records")
    return ("\n".join(lines) + "\n").encode("utf-8")


def create_renderer_bundle(
    dataset_revision: str,
    target_split: str = "auxiliary_train",
    fewshot_split: str = "dev",
    seed: int = 20260710,
    task_group: str = "mmlu",
    task_names: Optional[Sequence[str]] = None,
    max_targets_per_task: Optional[int] = None,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Render and self-hash one exact source projection plus its manifest."""

    revision = _validate_exact_revision(dataset_revision)
    if target_split.lower().find("test") >= 0:
        raise RendererVerificationError("target_split must be non-test")
    if fewshot_split != "dev":
        raise RendererVerificationError("phase one freezes fewshot_split=dev")
    if max_targets_per_task is not None and int(max_targets_per_task) < 1:
        raise RendererVerificationError("max_targets_per_task must be positive")
    implementation = backend or LmEvalMMLURendererBackend()
    version = str(implementation.distribution_version())
    if version != LM_EVAL_VERSION:
        raise RendererVerificationError(
            "phase one requires lm-eval %s, resolved %s" % (LM_EVAL_VERSION, version)
        )
    exact_tasks = list(task_names or implementation.resolve_task_names(task_group))
    exact_tasks = sorted(str(value) for value in exact_tasks)
    if not exact_tasks or len(set(exact_tasks)) != len(exact_tasks):
        raise RendererVerificationError("resolved MMLU task names must be non-empty and unique")
    if any(not name.startswith("mmlu_") for name in exact_tasks):
        raise RendererVerificationError("every phase-one task must be an mmlu_* task")

    collected = implementation.collect(
        task_names=exact_tasks,
        dataset_revision=revision,
        target_split=target_split,
        fewshot_split=fewshot_split,
        seed=int(seed),
        max_targets_per_task=max_targets_per_task,
    )
    if not isinstance(collected, Mapping):
        raise RendererVerificationError("renderer backend returned a non-object collection")
    source_files = _normalize_hash_records(collected.get("source_files"), "source_files")
    task_configs = _normalize_task_configs(collected.get("task_configs"), exact_tasks)
    dataset_tasks = _normalize_dataset_tasks(collected.get("dataset_tasks"), exact_tasks)
    renderer_source_hash = _sha256_object(source_files)
    template_hash = _sha256_object(task_configs)
    dataset_fingerprint_hash = _sha256_object(dataset_tasks)
    render_contract = {
        "version": RENDER_CONTRACT_VERSION,
        "task_group": task_group,
        "task_names": exact_tasks,
        "num_fewshot": 5,
        "fewshot_split": fewshot_split,
        "target_split": target_split,
        "chat_template": False,
        "multiturn": False,
        "seed": int(seed),
        "dataset_revision": revision,
        "dataset_fingerprint_sha256": dataset_fingerprint_hash,
        "renderer_source_sha256": renderer_source_hash,
        "template_sha256": template_hash,
    }
    render_contract_hash = _sha256_object(render_contract)
    renderer = {
        "renderer_entrypoint": str(collected.get("renderer_entrypoint") or "").strip(),
        "lm_eval_distribution": "lm_eval",
        "lm_eval_version": version,
        "renderer_source_sha256": renderer_source_hash,
        "source_files_sha256": renderer_source_hash,
        "template_sha256": template_hash,
        "task_configs_sha256": template_hash,
        "render_contract_sha256": render_contract_hash,
        "render_contract_version": RENDER_CONTRACT_VERSION,
        "dataset_revision": revision,
        "dataset_fingerprint_sha256": dataset_fingerprint_hash,
    }
    if not renderer["renderer_entrypoint"].startswith("lm_eval."):
        raise RendererVerificationError("renderer_entrypoint must resolve inside lm_eval")

    records = _normalize_projection_records(
        collected.get("records"),
        renderer=renderer,
        source=str(collected.get("source") or ""),
        target_split=target_split,
        fewshot_split=fewshot_split,
        task_names=exact_tasks,
    )
    projection_bytes = projection_jsonl_bytes(records)
    projection_hash = hashlib.sha256(projection_bytes).hexdigest()
    renderer["source_projection_sha256"] = projection_hash
    manifest: Dict[str, Any] = {
        "schema_version": RENDERER_SCHEMA_VERSION,
        "task_group": task_group,
        "task_names": exact_tasks,
        "num_fewshot": 5,
        "fewshot_split": fewshot_split,
        "target_split": target_split,
        "chat_template": False,
        "multiturn": False,
        "seed": int(seed),
        "max_targets_per_task": max_targets_per_task,
        "dataset": {
            "source": str(collected.get("source") or ""),
            "revision": revision,
            "fingerprint_sha256": dataset_fingerprint_hash,
            "tasks": dataset_tasks,
        },
        "renderer": renderer,
        "source_files": source_files,
        "task_configs": task_configs,
        "render_contract": render_contract,
        "projection": {
            "schema_version": PROJECTION_RECORD_VERSION,
            "sha256": projection_hash,
            "count": len(records),
            "record_ids": [record["id"] for record in records],
            "record_hashes": [record["projection_record_sha256"] for record in records],
        },
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return {"records": records, "projection_bytes": projection_bytes, "manifest": manifest}


def validate_renderer_manifest_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != RENDERER_SCHEMA_VERSION:
        raise RendererVerificationError("unsupported renderer manifest schema")
    if payload.get("manifest_sha256") != manifest_sha256(payload):
        raise RendererVerificationError("renderer manifest SHA256 mismatch")
    frozen = {
        "task_group": "mmlu",
        "num_fewshot": 5,
        "fewshot_split": "dev",
        "chat_template": False,
        "multiturn": False,
    }
    for key, value in frozen.items():
        if payload.get(key) != value:
            raise RendererVerificationError("renderer manifest must freeze %s=%r" % (key, value))
    _validate_exact_revision(_nested(payload, "dataset", "revision"))
    renderer = payload.get("renderer")
    if not isinstance(renderer, Mapping):
        raise RendererVerificationError("renderer manifest needs renderer evidence")
    required_renderer = (
        "renderer_entrypoint",
        "lm_eval_distribution",
        "lm_eval_version",
        "renderer_source_sha256",
        "source_files_sha256",
        "template_sha256",
        "task_configs_sha256",
        "render_contract_sha256",
        "render_contract_version",
        "dataset_revision",
        "dataset_fingerprint_sha256",
        "source_projection_sha256",
    )
    for key in required_renderer:
        if key not in renderer:
            raise RendererVerificationError("renderer evidence missing %s" % key)
    if renderer["lm_eval_distribution"] != "lm_eval" or renderer["lm_eval_version"] != LM_EVAL_VERSION:
        raise RendererVerificationError("renderer distribution/version mismatch")
    if not str(renderer["renderer_entrypoint"]).startswith("lm_eval."):
        raise RendererVerificationError("renderer entrypoint is not inside lm_eval")
    for key in required_renderer:
        if key.endswith("sha256"):
            _require_sha256(renderer[key], "renderer.%s" % key)
    if renderer["render_contract_version"] != RENDER_CONTRACT_VERSION:
        raise RendererVerificationError("unsupported render contract version")
    if renderer["dataset_revision"] != _nested(payload, "dataset", "revision"):
        raise RendererVerificationError("renderer and dataset revisions disagree")
    if renderer["dataset_fingerprint_sha256"] != _nested(
        payload, "dataset", "fingerprint_sha256"
    ):
        raise RendererVerificationError("renderer and dataset fingerprints disagree")
    projection = payload.get("projection")
    if not isinstance(projection, Mapping) or projection.get("schema_version") != PROJECTION_RECORD_VERSION:
        raise RendererVerificationError("renderer projection metadata is incomplete")
    _require_sha256(projection.get("sha256"), "projection.sha256")
    if projection["sha256"] != renderer["source_projection_sha256"]:
        raise RendererVerificationError("renderer source projection hashes disagree")
    count = int(projection.get("count", 0))
    if count < 1:
        raise RendererVerificationError("renderer projection count must be positive")
    for key in ("record_ids", "record_hashes"):
        values = projection.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise RendererVerificationError("projection.%s length mismatch" % key)
    if len(set(str(value) for value in projection["record_ids"])) != count:
        raise RendererVerificationError("projection record IDs must be unique")
    for value in projection["record_hashes"]:
        _require_sha256(value, "projection.record_hashes")
    source_files = _normalize_hash_records(payload.get("source_files"), "source_files")
    task_names = [str(value) for value in payload.get("task_names", [])]
    task_configs = _normalize_task_configs(payload.get("task_configs"), task_names)
    dataset_tasks = _normalize_dataset_tasks(_nested(payload, "dataset", "tasks"), task_names)
    if _sha256_object(source_files) != renderer["source_files_sha256"]:
        raise RendererVerificationError("source file aggregate hash mismatch")
    if renderer["renderer_source_sha256"] != renderer["source_files_sha256"]:
        raise RendererVerificationError("renderer source hash aliases disagree")
    if _sha256_object(task_configs) != renderer["task_configs_sha256"]:
        raise RendererVerificationError("task config aggregate hash mismatch")
    if renderer["template_sha256"] != renderer["task_configs_sha256"]:
        raise RendererVerificationError("template/task-config hash aliases disagree")
    if _sha256_object(dataset_tasks) != renderer["dataset_fingerprint_sha256"]:
        raise RendererVerificationError("dataset task fingerprint aggregate mismatch")
    if _sha256_object(payload.get("render_contract")) != renderer["render_contract_sha256"]:
        raise RendererVerificationError("render contract SHA256 mismatch")


def verify_export_bundle(
    projection_path: Path,
    manifest_path: Path,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Independently reload lm-eval/data and reproduce the complete projection."""

    projection_path = Path(projection_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RendererVerificationError("renderer manifest is not valid JSON") from exc
    validate_renderer_manifest_payload(payload)
    actual_bytes = projection_path.read_bytes()
    if hashlib.sha256(actual_bytes).hexdigest() != _nested(payload, "projection", "sha256"):
        raise RendererVerificationError("source projection file SHA256 mismatch")
    expected = create_renderer_bundle(
        dataset_revision=_nested(payload, "dataset", "revision"),
        target_split=str(payload["target_split"]),
        fewshot_split=str(payload["fewshot_split"]),
        seed=int(payload["seed"]),
        task_group=str(payload["task_group"]),
        task_names=[str(value) for value in payload["task_names"]],
        max_targets_per_task=payload.get("max_targets_per_task"),
        backend=backend,
    )
    if expected["projection_bytes"] != actual_bytes:
        raise RendererVerificationError(
            "source projection does not reproduce under the installed lm-eval renderer"
        )
    if expected["manifest"] != payload:
        raise RendererVerificationError(
            "renderer manifest evidence differs from independently recomputed evidence"
        )
    return dict(payload)


class LmEvalMMLURendererBackend:
    """Lazy production adapter for lm-eval 0.4.11 and Hugging Face datasets."""

    def distribution_version(self) -> str:
        errors = []
        for name in ("lm_eval", "lm-eval"):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                errors.append(exc)
        raise RendererVerificationError("lm_eval distribution is not installed") from errors[-1]

    def resolve_task_names(self, task_group: str) -> List[str]:
        manager = self._task_manager()
        loaded = manager.load_task_or_group(task_group)
        tasks = self._flatten_tasks(loaded)
        return sorted(name for name in tasks if name.startswith("mmlu_"))

    def collect(
        self,
        task_names: Sequence[str],
        dataset_revision: str,
        target_split: str,
        fewshot_split: str,
        seed: int,
        max_targets_per_task: Optional[int],
    ) -> Dict[str, Any]:
        try:
            import numpy as np
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - exercised on HPC2 only.
            raise RendererVerificationError(
                "real renderer verification requires numpy and datasets"
            ) from exc

        manager = self._task_manager()
        package = importlib.import_module("lm_eval")
        package_root = Path(package.__file__).resolve().parent
        records: List[Dict[str, Any]] = []
        source_files: Dict[str, str] = {}
        task_configs: List[Dict[str, Any]] = []
        dataset_tasks: List[Dict[str, Any]] = []
        source_names = set()
        for task_name in task_names:
            loaded = manager.load_task_or_group(task_name)
            tasks = self._flatten_tasks(loaded)
            if task_name not in tasks:
                if len(tasks) != 1:
                    raise RendererVerificationError("could not resolve exact task %s" % task_name)
                task = next(iter(tasks.values()))
            else:
                task = tasks[task_name]
            config = getattr(task, "config", None)
            dataset_path = str(self._config_value(config, "dataset_path") or "").strip()
            dataset_name = self._config_value(config, "dataset_name")
            if not dataset_path:
                raise RendererVerificationError("task %s has no dataset_path" % task_name)
            kwargs = dict(self._config_value(config, "dataset_kwargs") or {})
            kwargs.pop("revision", None)
            dataset = load_dataset(
                path=dataset_path,
                name=dataset_name,
                revision=dataset_revision,
                **kwargs,
            )
            task.dataset = dataset
            source_names.add(dataset_path)
            target_docs = self._processed_docs(task, target_split)
            fewshot_docs = self._processed_docs(task, fewshot_split)
            raw_fingerprints = {
                split: str(getattr(value, "_fingerprint", ""))
                for split, value in dataset.items()
            }
            processed_fingerprints = {
                "target": str(getattr(target_docs, "_fingerprint", "")),
                "fewshot": str(getattr(fewshot_docs, "_fingerprint", "")),
            }
            if any(not value for value in raw_fingerprints.values()) or any(
                not value for value in processed_fingerprints.values()
            ):
                raise RendererVerificationError("dataset fingerprints are unavailable")
            dataset_tasks.append(
                {
                    "task_name": task_name,
                    "dataset_path": dataset_path,
                    "dataset_name": str(dataset_name or ""),
                    "revision": dataset_revision,
                    "raw_split_fingerprints": dict(sorted(raw_fingerprints.items())),
                    "processed_target_fingerprint": processed_fingerprints["target"],
                    "processed_fewshot_fingerprint": processed_fingerprints["fewshot"],
                }
            )
            task_config, task_paths = self._task_config_evidence(task, task_name, package_root)
            task_configs.append(task_config)
            for path in task_paths:
                relative = str(path.resolve().relative_to(package_root))
                source_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

            count = len(target_docs)
            if max_targets_per_task is not None:
                count = min(count, int(max_targets_per_task))
            subject = task_name[len("mmlu_") :]
            fixed_demo_contract = None
            for target_index in range(count):
                target_doc = dict(target_docs[target_index])
                prompt, demos = self._render_with_captured_demos(
                    task,
                    target_doc,
                    fewshot_docs,
                    np,
                    _subject_seed(seed, task_name),
                )
                normalized_demos = self._demo_evidence(
                    task, demos, fewshot_docs, dataset_path, subject
                )
                demo_contract = canonical_json_bytes(normalized_demos)
                if fixed_demo_contract is None:
                    fixed_demo_contract = demo_contract
                elif fixed_demo_contract != demo_contract:
                    raise RendererVerificationError(
                        "five-shot demonstrations changed within task %s" % task_name
                    )
                sanitized_target = _strip_target_gold(target_doc)
                target_hash = _sha256_object(sanitized_target)
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                record_id = "%s:%s:%d" % (task_name, target_split, target_index)
                records.append(
                    {
                        "id": record_id,
                        "target_doc_id": record_id,
                        "target_doc_index": target_index,
                        "target_doc_sha256": target_hash,
                        "dataset_fingerprint": hashlib.sha256(
                            processed_fingerprints["target"].encode("utf-8")
                        ).hexdigest(),
                        "task_group": "mmlu",
                        "task_name": task_name,
                        "num_fewshot": 5,
                        "source": "%s@%s" % (dataset_path, dataset_revision),
                        "split": target_split,
                        "subject": subject,
                        "text": prompt,
                        "render_sha256": prompt_hash,
                        "uses_target_gold_labels": False,
                        "fewshot_answers_present": True,
                        "fewshot_sample_ids": [demo["id"] for demo in normalized_demos],
                        "demonstrations": normalized_demos,
                    }
                )
        return {
            "renderer_entrypoint": "lm_eval.api.task.ConfigurableTask.fewshot_context",
            "source": ",".join(sorted(source_names)) + "@" + dataset_revision,
            "records": records,
            "source_files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(source_files.items())
            ],
            "task_configs": sorted(task_configs, key=lambda value: value["task_name"]),
            "dataset_tasks": sorted(dataset_tasks, key=lambda value: value["task_name"]),
        }

    @staticmethod
    def _task_manager() -> Any:
        try:
            from lm_eval.tasks import TaskManager
        except Exception as exc:  # pragma: no cover - remote-only dependency path.
            raise RendererVerificationError("lm_eval 0.4.11 TaskManager is unavailable") from exc
        return TaskManager()

    @classmethod
    def _flatten_tasks(cls, value: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, Mapping):
                    result.update(cls._flatten_tasks(item))
                elif hasattr(item, "fewshot_context"):
                    result[str(key)] = item
        return result

    @staticmethod
    def _config_value(config: Any, name: str) -> Any:
        if isinstance(config, Mapping):
            return config.get(name)
        return getattr(config, name, None)

    def _processed_docs(self, task: Any, split: str) -> Any:
        if split not in task.dataset:
            raise RendererVerificationError("task dataset lacks split %s" % split)
        docs = task.dataset[split]
        process_docs = self._config_value(getattr(task, "config", None), "process_docs")
        if callable(process_docs):
            docs = process_docs(docs)
        return docs

    def _task_config_evidence(
        self, task: Any, task_name: str, package_root: Path
    ) -> Any:
        config = getattr(task, "config", None)
        projection = {
            "task_name": task_name,
            "dataset_path": str(self._config_value(config, "dataset_path") or ""),
            "dataset_name": str(self._config_value(config, "dataset_name") or ""),
            "fewshot_split": str(self._config_value(config, "fewshot_split") or ""),
            "doc_to_text": str(self._config_value(config, "doc_to_text") or ""),
            "doc_to_target": str(self._config_value(config, "doc_to_target") or ""),
            "fewshot_delimiter": str(
                self._config_value(config, "fewshot_delimiter") or ""
            ),
            "target_delimiter": str(
                self._config_value(config, "target_delimiter") or ""
            ),
        }
        paths = set()
        for value in (
            task.__class__,
            getattr(task, "fewshot_context", None),
            getattr(task, "doc_to_text", None),
            getattr(task, "doc_to_target", None),
            self._config_value(config, "process_docs"),
        ):
            try:
                path = inspect.getsourcefile(value)
            except (TypeError, OSError):
                path = None
            if path:
                resolved = Path(path).resolve()
                if package_root == resolved or package_root in resolved.parents:
                    paths.add(resolved)
        yaml_matches = sorted(package_root.rglob(task_name + ".yaml"))
        if len(yaml_matches) != 1:
            raise RendererVerificationError(
                "expected one installed task YAML for %s, found %d"
                % (task_name, len(yaml_matches))
            )
        paths.add(yaml_matches[0].resolve())
        projection["config_sha256"] = _sha256_object(projection)
        projection["config_file"] = str(yaml_matches[0].resolve().relative_to(package_root))
        projection["config_file_sha256"] = hashlib.sha256(
            yaml_matches[0].read_bytes()
        ).hexdigest()
        return projection, paths

    @staticmethod
    def _call_fewshot_context(task: Any, doc: Mapping[str, Any], rnd: Any) -> str:
        method = task.fewshot_context
        supported = inspect.signature(method).parameters
        candidates = {
            "doc": doc,
            "num_fewshot": 5,
            "rnd": rnd,
            "description": None,
            "system_instruction": None,
            "apply_chat_template": False,
            "chat_template": None,
            "fewshot_as_multiturn": False,
            "multiturn": False,
            "gen_prefix": None,
        }
        kwargs = {key: value for key, value in candidates.items() if key in supported}
        try:
            rendered = method(**kwargs)
        except TypeError as exc:
            raise RendererVerificationError(
                "lm_eval fewshot_context signature is incompatible with the frozen adapter"
            ) from exc
        if not isinstance(rendered, str) or not rendered.rstrip().endswith("Answer:"):
            raise RendererVerificationError(
                "actual lm_eval five-shot context must end at the target Answer: boundary"
            )
        return rendered

    def _render_with_captured_demos(
        self,
        task: Any,
        target_doc: Mapping[str, Any],
        fewshot_docs: Any,
        numpy_module: Any,
        fewshot_seed: int,
    ) -> Any:
        seed_setter = getattr(task, "set_fewshot_seed", None)
        if callable(seed_setter):
            seed_setter(fewshot_seed)
        rnd = numpy_module.random.default_rng(fewshot_seed)
        original_sampler = getattr(task, "sampler", None)
        recorder = _RecordingSampler(original_sampler) if original_sampler is not None else None
        original_rnd = getattr(original_sampler, "rnd", None) if original_sampler is not None else None
        if original_sampler is not None and hasattr(original_sampler, "rnd"):
            original_sampler.rnd = rnd
        if recorder is not None:
            try:
                task.sampler = recorder
            except (AttributeError, TypeError):
                recorder = None
        try:
            prompt = self._call_fewshot_context(task, target_doc, rnd)
        finally:
            if recorder is not None:
                task.sampler = original_sampler
            if original_sampler is not None and hasattr(original_sampler, "rnd"):
                original_sampler.rnd = original_rnd
        demos = recorder.last_sample if recorder is not None else None
        if not demos:
            demos = self._infer_demos_from_prompt(task, prompt, fewshot_docs)
        if len(demos) != 5:
            raise RendererVerificationError("actual lm_eval context did not expose five demos")
        return prompt, list(demos)

    @staticmethod
    def _infer_demos_from_prompt(task: Any, prompt: str, fewshot_docs: Any) -> List[Any]:
        matches = []
        for index in range(len(fewshot_docs)):
            doc = dict(fewshot_docs[index])
            snippet = str(task.doc_to_text(doc)) + str(task.doc_to_target(doc))
            position = prompt.find(snippet)
            if position >= 0:
                matches.append((position, doc))
        matches.sort(key=lambda value: value[0])
        return [doc for _, doc in matches[:5]]

    @staticmethod
    def _demo_evidence(
        task: Any,
        demos: Sequence[Mapping[str, Any]],
        fewshot_docs: Any,
        source: str,
        subject: str,
    ) -> List[Dict[str, Any]]:
        by_hash = {
            _sha256_object(dict(fewshot_docs[index])): index
            for index in range(len(fewshot_docs))
        }
        result = []
        for demo in demos:
            normalized = dict(demo)
            doc_hash = _sha256_object(normalized)
            if doc_hash not in by_hash:
                raise RendererVerificationError("selected demo is absent from frozen dev split")
            rendered = str(task.doc_to_text(normalized)) + str(task.doc_to_target(normalized))
            gold = str(task.doc_to_target(normalized))
            demo_index = by_hash[doc_hash]
            result.append(
                {
                    "id": "%s:dev:%d" % (subject, demo_index),
                    "doc_index": demo_index,
                    "source": source,
                    "split": "dev",
                    "subject": subject,
                    "doc_sha256": doc_hash,
                    "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                    "gold_sha256": hashlib.sha256(gold.encode("utf-8")).hexdigest(),
                }
            )
        return result


class _RecordingSampler:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.last_sample: Optional[List[Any]] = None

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        result = self.delegate.sample(*args, **kwargs)
        self.last_sample = list(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _normalize_projection_records(
    value: Any,
    renderer: Mapping[str, Any],
    source: str,
    target_split: str,
    fewshot_split: str,
    task_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RendererVerificationError("renderer backend emitted no projection records")
    result = []
    seen = set()
    fixed_demos: Dict[str, bytes] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise RendererVerificationError("projection record %d is not an object" % index)
        record = dict(raw)
        record_id = str(record.get("id") or "")
        if not record_id or record_id in seen:
            raise RendererVerificationError("projection record IDs must be non-empty and unique")
        seen.add(record_id)
        task_name = str(record.get("task_name") or "")
        if task_name not in task_names:
            raise RendererVerificationError("projection record task is not frozen")
        if record.get("task_group") != "mmlu" or record.get("num_fewshot") != 5:
            raise RendererVerificationError("projection record is not MMLU five-shot")
        if record.get("uses_target_gold_labels") is not False:
            raise RendererVerificationError("projection record may use target gold")
        if record.get("fewshot_answers_present") is not True:
            raise RendererVerificationError("projection record lacks demo answers")
        if str(record.get("source") or "") != source or record.get("split") != target_split:
            raise RendererVerificationError("projection record source/split mismatch")
        if _forbidden_target_paths(record):
            raise RendererVerificationError("projection record contains target gold fields")
        prompt = str(record.get("text") or "")
        if not prompt.rstrip().endswith("Answer:"):
            raise RendererVerificationError("projection prompt must end at Answer:")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if record.get("render_sha256") != prompt_hash:
            raise RendererVerificationError("projection prompt hash mismatch")
        for key in ("target_doc_sha256", "dataset_fingerprint"):
            _require_sha256(record.get(key), "projection.%s" % key)
        demos = record.get("demonstrations")
        if not isinstance(demos, list) or len(demos) != 5:
            raise RendererVerificationError("projection record needs exactly five demos")
        normalized_demos = []
        for demo in demos:
            if not isinstance(demo, Mapping):
                raise RendererVerificationError("projection demo must be an object")
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
            if any(key not in demo for key in required):
                raise RendererVerificationError("projection demo provenance is incomplete")
            if demo["split"] != fewshot_split or demo["subject"] != record.get("subject"):
                raise RendererVerificationError("projection demo split/subject mismatch")
            for key in ("doc_sha256", "rendered_sha256", "gold_sha256"):
                _require_sha256(demo[key], "projection demo %s" % key)
            normalized_demos.append({key: demo[key] for key in required})
        demo_ids = [str(demo["id"]) for demo in normalized_demos]
        if demo_ids != [str(value) for value in record.get("fewshot_sample_ids", [])]:
            raise RendererVerificationError("projection demo IDs/order mismatch")
        if len(set(demo_ids)) != 5 or str(record.get("target_doc_id")) in demo_ids:
            raise RendererVerificationError("projection target/demo IDs are not disjoint")
        demo_contract = canonical_json_bytes(normalized_demos)
        previous = fixed_demos.setdefault(str(record.get("subject")), demo_contract)
        if previous != demo_contract:
            raise RendererVerificationError("fixed demos changed within one subject")
        record.update(
            {
                "schema_version": PROJECTION_RECORD_VERSION,
                "template_sha256": renderer["template_sha256"],
                "render_contract_sha256": renderer["render_contract_sha256"],
                "dataset_revision": renderer["dataset_revision"],
                "dataset_fingerprint_sha256": renderer["dataset_fingerprint_sha256"],
                "renderer_source_sha256": renderer["renderer_source_sha256"],
                "demonstrations": normalized_demos,
            }
        )
        body = {
            key: item for key, item in record.items() if key != "projection_record_sha256"
        }
        record["projection_record_sha256"] = _sha256_object(body)
        result.append(record)
    return result


def _normalize_hash_records(value: Any, context: str) -> List[Dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise RendererVerificationError("%s must be a non-empty list" % context)
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise RendererVerificationError("%s entries must be objects" % context)
        path = str(item.get("path") or "")
        if not path or path.startswith("/") or ".." in Path(path).parts or path in seen:
            raise RendererVerificationError("%s paths must be unique package-relative paths" % context)
        seen.add(path)
        result.append({"path": path, "sha256": _require_sha256(item.get("sha256"), context)})
    return sorted(result, key=lambda item: item["path"])


def _normalize_task_configs(value: Any, task_names: Sequence[str]) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise RendererVerificationError("task_configs must be a list")
    result = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(result) != len(task_names):
        raise RendererVerificationError("task_configs must cover every task exactly once")
    if sorted(str(item.get("task_name")) for item in result) != sorted(task_names):
        raise RendererVerificationError("task_configs task names mismatch")
    for item in result:
        _require_sha256(item.get("config_sha256"), "task config hash")
        _require_sha256(item.get("config_file_sha256"), "task config file hash")
    return sorted(result, key=lambda item: str(item["task_name"]))


def _normalize_dataset_tasks(value: Any, task_names: Sequence[str]) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise RendererVerificationError("dataset tasks must be a list")
    result = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(result) != len(task_names):
        raise RendererVerificationError("dataset evidence must cover every task exactly once")
    if sorted(str(item.get("task_name")) for item in result) != sorted(task_names):
        raise RendererVerificationError("dataset evidence task names mismatch")
    for item in result:
        if not str(item.get("revision") or ""):
            raise RendererVerificationError("dataset task revision is missing")
        raw = item.get("raw_split_fingerprints")
        if not isinstance(raw, Mapping) or not raw:
            raise RendererVerificationError("dataset raw split fingerprints are missing")
        if any(not str(value) for value in raw.values()):
            raise RendererVerificationError("dataset raw split fingerprint is empty")
        for key in ("processed_target_fingerprint", "processed_fewshot_fingerprint"):
            if not str(item.get(key) or ""):
                raise RendererVerificationError("dataset %s is missing" % key)
    return sorted(result, key=lambda item: str(item["task_name"]))


def _strip_target_gold(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_target_gold(item)
            for key, item in value.items()
            if _normalized_key(key) not in FORBIDDEN_TARGET_KEYS
        }
    if isinstance(value, list):
        return [_strip_target_gold(item) for item in value]
    return value


def _forbidden_target_paths(value: Any, path: Sequence[str] = ()) -> List[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = tuple(path) + (str(key),)
            if _normalized_key(key) in FORBIDDEN_TARGET_KEYS:
                found.append(".".join(current))
            found.extend(_forbidden_target_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_target_paths(item, tuple(path) + (str(index),)))
    return found


def _normalized_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_")


def _validate_exact_revision(value: Any) -> str:
    revision = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise RendererVerificationError(
            "dataset_revision must be an exact lowercase 40-64 hex commit"
        )
    return revision


def _require_sha256(value: Any, context: str) -> str:
    digest = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RendererVerificationError("%s must be a lowercase SHA256" % context)
    return digest


def _sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise RendererVerificationError("missing renderer manifest field: %s" % ".".join(keys))
        value = value[key]
    return value


def _subject_seed(seed: int, subject: str) -> int:
    digest = hashlib.sha256((str(seed) + "\0" + subject).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
