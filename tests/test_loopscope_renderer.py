import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from tflt.loopscope.mmlu_renderer import (
    LmEvalMMLURendererBackend,
    RENDERER_SCHEMA_VERSION,
    RendererVerificationError,
    canonical_json_bytes,
    create_renderer_bundle,
    manifest_sha256,
    projection_jsonl_bytes,
    validate_renderer_manifest_payload,
    verify_export_bundle,
)
try:
    from loopscope_fixtures import FAKE_DATASET_REVISION, FakeRendererBackend
except ModuleNotFoundError:
    from tests.loopscope_fixtures import FAKE_DATASET_REVISION, FakeRendererBackend


def _load_export_script():
    path = Path(__file__).resolve().parents[1] / "scripts/loopscope/export_mmlu_renderer.py"
    spec = importlib.util.spec_from_file_location("export_mmlu_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(count=3):
    return create_renderer_bundle(
        dataset_revision=FAKE_DATASET_REVISION,
        task_names=["mmlu_math"],
        max_targets_per_task=count,
        backend=FakeRendererBackend(count),
    )


def _object_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _rehash_projection(manifest, records):
    for record in records:
        body = {
            key: value
            for key, value in record.items()
            if key != "projection_record_sha256"
        }
        record["projection_record_sha256"] = _object_hash(body)
    projection_bytes = projection_jsonl_bytes(records)
    projection_hash = hashlib.sha256(projection_bytes).hexdigest()
    manifest["projection"]["sha256"] = projection_hash
    manifest["projection"]["record_hashes"] = [
        record["projection_record_sha256"] for record in records
    ]
    manifest["renderer"]["source_projection_sha256"] = projection_hash
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return projection_bytes


class _HybridTaskConfig(Mapping):
    """Minimal lm-eval 0.4.11 TaskConfig shape for dependency-light tests."""

    def __init__(self, attribute_value=None, mapping_value=None, has_attribute=True):
        if has_attribute:
            self.dataset_path = attribute_value
        self._mapping_value = mapping_value

    def __getitem__(self, key):
        if key == "dataset_path" and self._mapping_value is not None:
            return self._mapping_value
        raise KeyError(key)

    def __iter__(self):
        if self._mapping_value is not None:
            yield "dataset_path"

    def __len__(self):
        return int(self._mapping_value is not None)

    def get(self, key, default=None):
        if key == "dataset_path":
            return self._mapping_value
        return default


class _FakeTaskManager:
    def __init__(self, task_index):
        self.task_index = task_index


class _FakeTask:
    def __init__(self, task_name):
        self.config = type(
            "FakeTaskConfig",
            (),
            {
                "task": task_name,
                "dataset_path": "cais/mmlu",
                "dataset_name": task_name[len("mmlu_") :],
                "fewshot_split": "dev",
                "doc_to_text": "Question\nAnswer:",
                "doc_to_target": "answer",
                "fewshot_delimiter": "\n\n",
                "target_delimiter": " ",
                "process_docs": None,
            },
        )()

    def fewshot_context(self):
        raise AssertionError("not called")


class LoopScopeRendererTest(unittest.TestCase):
    def test_registry_yaml_selects_loaded_task_among_duplicate_basenames(self):
        task_name = "mmlu_abstract_algebra"
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp) / "lm_eval"
            registered = (
                package_root / "tasks" / "mmlu" / "default" / (task_name + ".yaml")
            )
            registered.parent.mkdir(parents=True)
            registered.write_text("task: mmlu_abstract_algebra\n", encoding="utf-8")
            for family in ("generative", "continuation"):
                decoy = package_root / "tasks" / "mmlu" / family / (task_name + ".yaml")
                decoy.parent.mkdir(parents=True)
                decoy.write_text("task: decoy\n", encoding="utf-8")
            manager = _FakeTaskManager(
                {
                    task_name: {
                        "type": "task",
                        "yaml_path": str(registered),
                    }
                }
            )
            resolved = LmEvalMMLURendererBackend._registered_task_yaml(
                manager, task_name, package_root
            )
            self.assertEqual(resolved, registered.resolve())
            projection, paths = LmEvalMMLURendererBackend()._task_config_evidence(
                _FakeTask(task_name), task_name, package_root, resolved
            )
        self.assertEqual(
            projection["config_file"],
            "tasks/mmlu/default/mmlu_abstract_algebra.yaml",
        )
        self.assertEqual(
            projection["config_file_sha256"],
            hashlib.sha256(b"task: mmlu_abstract_algebra\n").hexdigest(),
        )
        self.assertIn(registered.resolve(), paths)

    def test_registry_yaml_fail_closed_contract(self):
        task_name = "mmlu_abstract_algebra"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_root = root / "lm_eval"
            tasks_root = package_root / "tasks" / "mmlu" / "default"
            tasks_root.mkdir(parents=True)
            valid = tasks_root / (task_name + ".yaml")
            valid.write_text("task: mmlu_abstract_algebra\n", encoding="utf-8")
            wrong_stem = tasks_root / "mmlu_anatomy.yaml"
            wrong_stem.write_text("task: mmlu_anatomy\n", encoding="utf-8")
            non_yaml = tasks_root / (task_name + ".json")
            non_yaml.write_text("{}\n", encoding="utf-8")
            outside_tasks = package_root / "other" / (task_name + ".yaml")
            outside_tasks.parent.mkdir(parents=True)
            outside_tasks.write_text("task: escaped\n", encoding="utf-8")
            escaped = root / (task_name + ".yaml")
            escaped.write_text("task: escaped\n", encoding="utf-8")
            cases = {
                "registry-not-mapping": _FakeTaskManager([]),
                "task-missing": _FakeTaskManager({}),
                "entry-not-mapping": _FakeTaskManager({task_name: []}),
                "wrong-entry-type": _FakeTaskManager(
                    {task_name: {"type": "group", "yaml_path": str(valid)}}
                ),
                "yaml-path-missing": _FakeTaskManager(
                    {task_name: {"type": "task"}}
                ),
                "yaml-path-sentinel": _FakeTaskManager(
                    {task_name: {"type": "task", "yaml_path": -1}}
                ),
                "not-yaml": _FakeTaskManager(
                    {task_name: {"type": "task", "yaml_path": str(non_yaml)}}
                ),
                "file-missing": _FakeTaskManager(
                    {
                        task_name: {
                            "type": "task",
                            "yaml_path": str(
                                package_root
                                / "tasks"
                                / "mmlu"
                                / "missing"
                                / (task_name + ".yaml")
                            ),
                        }
                    }
                ),
                "outside-package": _FakeTaskManager(
                    {task_name: {"type": "task", "yaml_path": str(escaped)}}
                ),
                "outside-tasks": _FakeTaskManager(
                    {task_name: {"type": "task", "yaml_path": str(outside_tasks)}}
                ),
                "wrong-stem": _FakeTaskManager(
                    {task_name: {"type": "task", "yaml_path": str(wrong_stem)}}
                ),
            }
            for label, manager in cases.items():
                with self.subTest(label=label), self.assertRaises(
                    RendererVerificationError
                ):
                    LmEvalMMLURendererBackend._registered_task_yaml(
                        manager, task_name, package_root
                    )

    def test_task_config_evidence_rejects_mismatched_loaded_task_name(self):
        task_name = "mmlu_abstract_algebra"
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp) / "lm_eval"
            registered = (
                package_root / "tasks" / "mmlu" / "default" / (task_name + ".yaml")
            )
            registered.parent.mkdir(parents=True)
            registered.write_text("task: mmlu_abstract_algebra\n", encoding="utf-8")
            task = _FakeTask("mmlu_anatomy")
            with self.assertRaisesRegex(RendererVerificationError, "task name"):
                LmEvalMMLURendererBackend()._task_config_evidence(
                    task, task_name, package_root, registered
                )

    def test_phase_one_renderer_only_accepts_validation_targets(self):
        bundle = create_renderer_bundle(
            dataset_revision=FAKE_DATASET_REVISION,
            target_split="validation",
            task_names=["mmlu_math"],
            max_targets_per_task=1,
            backend=FakeRendererBackend(1),
        )
        self.assertEqual(bundle["manifest"]["target_split"], "validation")
        for split in ("auxiliary_train", "dev", "test"):
            with self.subTest(split=split), self.assertRaisesRegex(
                RendererVerificationError, "target_split=validation"
            ):
                create_renderer_bundle(
                    dataset_revision=FAKE_DATASET_REVISION,
                    target_split=split,
                    task_names=["mmlu_math"],
                    max_targets_per_task=1,
                    backend=FakeRendererBackend(1),
                )

    def test_config_value_prefers_populated_hybrid_attribute(self):
        config = _HybridTaskConfig(
            attribute_value="cais/mmlu",
            mapping_value=None,
        )
        self.assertEqual(
            LmEvalMMLURendererBackend._config_value(config, "dataset_path"),
            "cais/mmlu",
        )

    def test_config_value_preserves_mapping_and_attribute_only_objects(self):
        self.assertEqual(
            LmEvalMMLURendererBackend._config_value(
                {"dataset_path": "mapping-only"}, "dataset_path"
            ),
            "mapping-only",
        )

        class AttributeOnly:
            dataset_path = "attribute-only"

        self.assertEqual(
            LmEvalMMLURendererBackend._config_value(
                AttributeOnly(), "dataset_path"
            ),
            "attribute-only",
        )

    def test_config_value_uses_explicit_none_precedence(self):
        cases = [
            (_HybridTaskConfig("attribute", "mapping"), "attribute"),
            (_HybridTaskConfig(None, "mapping"), "mapping"),
            (_HybridTaskConfig(mapping_value="mapping", has_attribute=False), "mapping"),
            (_HybridTaskConfig("", "mapping"), ""),
            (_HybridTaskConfig(False, "mapping"), False),
            (_HybridTaskConfig(0, "mapping"), 0),
            (_HybridTaskConfig(None, "", has_attribute=False), ""),
        ]
        for config, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    LmEvalMMLURendererBackend._config_value(
                        config, "dataset_path"
                    ),
                    expected,
                )
        self.assertIsNone(
            LmEvalMMLURendererBackend._config_value(None, "dataset_path")
        )

    def test_export_and_independent_rerender_close_all_provenance(self):
        bundle = _bundle(3)
        manifest = bundle["manifest"]
        self.assertEqual(manifest["schema_version"], RENDERER_SCHEMA_VERSION)
        self.assertEqual(manifest["renderer"]["lm_eval_version"], "0.4.11")
        self.assertEqual(manifest["dataset"]["revision"], FAKE_DATASET_REVISION)
        self.assertEqual(len(bundle["records"][0]["demonstrations"]), 5)
        self.assertTrue(bundle["records"][0]["demonstrations"][0]["gold_sha256"])
        self.assertTrue(bundle["records"][0]["text"].endswith("Answer:"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projection = root / "projection.jsonl"
            evidence = root / "renderer.json"
            projection.write_bytes(bundle["projection_bytes"])
            evidence.write_text(json.dumps(manifest), encoding="utf-8")
            verified = verify_export_bundle(
                projection, evidence, backend=FakeRendererBackend(3)
            )
        self.assertEqual(verified, manifest)

    def test_handwritten_prompt_fails_even_after_all_self_hashes_are_repaired(self):
        bundle = _bundle(2)
        records = json.loads(json.dumps(bundle["records"]))
        records[0]["text"] = "Hand-written five-shot impostor\nAnswer:"
        records[0]["render_sha256"] = hashlib.sha256(
            records[0]["text"].encode("utf-8")
        ).hexdigest()
        manifest = json.loads(json.dumps(bundle["manifest"]))
        projection_bytes = _rehash_projection(manifest, records)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projection = root / "projection.jsonl"
            evidence = root / "renderer.json"
            projection.write_bytes(projection_bytes)
            evidence.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                RendererVerificationError,
                "does not reproduce|differs",
            ):
                verify_export_bundle(
                    projection, evidence, backend=FakeRendererBackend(2)
                )

    def test_source_template_demo_and_target_tampering_are_all_rejected(self):
        for mutation in ("source", "template", "demo_gold", "target_id"):
            with self.subTest(mutation=mutation):
                bundle = _bundle(2)
                manifest = json.loads(json.dumps(bundle["manifest"]))
                records = json.loads(json.dumps(bundle["records"]))
                if mutation == "source":
                    manifest["source_files"][0]["sha256"] = "f" * 64
                    source_hash = _object_hash(manifest["source_files"])
                    manifest["renderer"]["source_files_sha256"] = source_hash
                    manifest["renderer"]["renderer_source_sha256"] = source_hash
                    manifest["render_contract"]["renderer_source_sha256"] = source_hash
                    for record in records:
                        record["renderer_source_sha256"] = source_hash
                elif mutation == "template":
                    manifest["task_configs"][0]["doc_to_text"] = "handwritten-template"
                    manifest["task_configs"][0]["config_sha256"] = "e" * 64
                    template_hash = _object_hash(manifest["task_configs"])
                    manifest["renderer"]["task_configs_sha256"] = template_hash
                    manifest["renderer"]["template_sha256"] = template_hash
                    manifest["render_contract"]["template_sha256"] = template_hash
                    for record in records:
                        record["template_sha256"] = template_hash
                elif mutation == "demo_gold":
                    records[0]["demonstrations"][0]["gold_sha256"] = "d" * 64
                else:
                    records[0]["target_doc_id"] += "-changed"
                if mutation in ("source", "template"):
                    contract_hash = _object_hash(manifest["render_contract"])
                    manifest["renderer"]["render_contract_sha256"] = contract_hash
                    for record in records:
                        record["render_contract_sha256"] = contract_hash
                projection_bytes = _rehash_projection(manifest, records)
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    projection = root / "projection.jsonl"
                    evidence = root / "renderer.json"
                    projection.write_bytes(projection_bytes)
                    evidence.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(RendererVerificationError):
                        verify_export_bundle(
                            projection,
                            evidence,
                            backend=FakeRendererBackend(2),
                        )

    def test_old_arbitrary_hex_manifest_is_rejected(self):
        payload = {
            "schema_version": "loopscope.mmlu-renderer-manifest.v1",
            "task_group": "mmlu",
            "num_fewshot": 5,
            "fewshot_split": "dev",
            "chat_template": False,
            "multiturn": False,
            "renderer": {
                "renderer_entrypoint": "lm_eval.tasks.mmlu",
                "lm_eval_version": "0.4.11",
                "renderer_source_sha256": "1" * 64,
                "template_sha256": "2" * 64,
                "render_contract_sha256": "3" * 64,
            },
        }
        payload["manifest_sha256"] = manifest_sha256(payload)
        with self.assertRaisesRegex(RendererVerificationError, "unsupported"):
            validate_renderer_manifest_payload(payload)

    def test_export_cli_has_no_bypass_and_is_write_once(self):
        exporter = _load_export_script()
        defaults = exporter.parse_args(
            [
                "--output-jsonl",
                "projection.jsonl",
                "--manifest",
                "renderer.json",
                "--dataset-revision",
                FAKE_DATASET_REVISION,
            ]
        )
        self.assertEqual(defaults.target_split, "validation")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projection = root / "projection.jsonl"
            evidence = root / "renderer.json"
            argv = [
                "--output-jsonl",
                str(projection),
                "--manifest",
                str(evidence),
                "--dataset-revision",
                FAKE_DATASET_REVISION,
                "--task-name",
                "mmlu_math",
                "--max-targets-per-task",
                "2",
            ]
            self.assertEqual(exporter.main(argv, backend=FakeRendererBackend(2)), 0)
            self.assertNotIn("trust", exporter.parse_args(argv).__dict__)
            with self.assertRaises(FileExistsError):
                exporter.main(argv, backend=FakeRendererBackend(2))


if __name__ == "__main__":
    unittest.main()
