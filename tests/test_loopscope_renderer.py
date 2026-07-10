import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.mmlu_renderer import (
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


class LoopScopeRendererTest(unittest.TestCase):
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
