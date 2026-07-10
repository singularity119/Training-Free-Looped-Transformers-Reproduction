import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.grid import generate_window_grid
from tflt.loopscope.mmlu_renderer import create_renderer_bundle, verify_export_bundle
try:
    from loopscope_fixtures import FAKE_DATASET_REVISION, FakeRendererBackend
except ModuleNotFoundError:
    from tests.loopscope_fixtures import FAKE_DATASET_REVISION, FakeRendererBackend


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / "loopscope" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_script("build_mmlu_probe_pool.py")
prepare = _load_script("prepare_qwen17_phase1.py")


def _renderer_bundle(count=8):
    return create_renderer_bundle(
        dataset_revision=FAKE_DATASET_REVISION,
        task_names=["mmlu_math"],
        max_targets_per_task=count,
        backend=FakeRendererBackend(count),
    )


def _source_record(index, subject="math"):
    if subject != "math":
        raise ValueError("fixture backend only exposes math")
    return dict(_renderer_bundle(max(index + 1, 1))["records"][index])


class LoopScopePhaseOneScriptTest(unittest.TestCase):
    def test_builder_requires_lm_eval_five_shot_contract_and_prepare_accepts_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _renderer_bundle(4)
            renderer = root / "renderer.json"
            renderer.write_text(json.dumps(bundle["manifest"]), encoding="utf-8")
            source = root / "source.jsonl"
            source.write_bytes(bundle["projection_bytes"])
            pool = root / "pool.jsonl"
            manifest = root / "pool-manifest.json"
            code = build.main(
                [
                    "--input-jsonl", str(source),
                    "--output-jsonl", str(pool),
                    "--manifest", str(manifest),
                    "--renderer-manifest", str(renderer),
                    "--source", bundle["manifest"]["dataset"]["source"],
                    "--split", "auxiliary_train",
                    "--count", "4",
                ],
                renderer_verifier=lambda projection, evidence: verify_export_bundle(
                    projection, evidence, backend=FakeRendererBackend(4)
                ),
            )
            self.assertEqual(code, 0)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertFalse(payload["uses_target_gold_labels"])
            self.assertTrue(payload["fewshot_answers_present"])
            self.assertEqual(payload["renderer"]["lm_eval_version"], "0.4.11")
            self.assertEqual(len(payload["rendering_records"][0]["demonstrations"]), 5)
            prepare._validate_pool(pool, payload)

    def test_builder_rejects_structured_zero_shot_path(self):
        bundle = _renderer_bundle(1)
        renderer = bundle["manifest"]
        item = _source_record(0)
        item["question"] = "forbidden"
        body = {
            key: value for key, value in item.items() if key != "projection_record_sha256"
        }
        item["projection_record_sha256"] = build.hashlib.sha256(
            build.canonical_json_bytes(body)
        ).hexdigest()
        raw = (json.dumps(item) + "\n").encode("utf-8")
        with self.assertRaises(build.PoolBuildError):
            build._load_source_records(
                raw,
                bundle["manifest"]["dataset"]["source"],
                "auxiliary_train",
                renderer,
            )

    def test_exact_hpc2_paths_are_not_configurable(self):
        run = prepare.DEFAULT_RUN_BASE / (
            "loopscope-qwen17-mmlu-phase1-20260710-120000"
        )
        venv = prepare.DEFAULT_REMOTE_REPO / ".venv-loopscope-cu121-20260710"
        prepare._validate_hpc2_paths(
            prepare.DEFAULT_REMOTE_REPO, prepare.DEFAULT_RUN_BASE, run, venv
        )
        with self.assertRaises(prepare.PreparationError):
            prepare._validate_hpc2_paths(
                Path(str(prepare.DEFAULT_REMOTE_REPO) + "-wrong"),
                prepare.DEFAULT_RUN_BASE,
                run,
                venv,
            )
        with self.assertRaises(prepare.PreparationError):
            prepare._validate_hpc2_paths(
                prepare.DEFAULT_REMOTE_REPO,
                prepare.DEFAULT_RUN_BASE,
                run,
                prepare.DEFAULT_REMOTE_REPO / ".venv",
            )

    def test_gate_e_jobs_include_full_probe_then_offline_score(self):
        root = prepare.DEFAULT_RUN_BASE / "loopscope-qwen17-mmlu-phase1-20260710-120000"
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/loopscope/qwen17_mmlu_phase1.json")
            .read_text(encoding="utf-8")
        )
        jobs = prepare._build_jobs(
            root,
            prepare.DEFAULT_REMOTE_REPO,
            prepare.DEFAULT_REMOTE_REPO / ".venv-loopscope-cu121-20260710",
            config,
            generate_window_grid(28),
            root / "manifests/criterion_v0.json",
        )
        self.assertEqual({"probe-layers-full", "probe-windows-full"}, {
            job["job_id"] for job in jobs["gate-e-probe"]
        })
        self.assertNotIn("--max-examples", jobs["gate-e-probe"][0]["argv"])
        self.assertEqual(jobs["gate-e-score"][0]["argv"][3], "score-windows")
        self.assertIn("--criterion", jobs["gate-e-score"][0]["argv"])
        self.assertTrue(jobs["gate-e-full"])

    def test_revision_report_is_bound_to_run_stage_jobs_and_observations(self):
        root = prepare.DEFAULT_RUN_BASE / "loopscope-qwen17-mmlu-phase1-20260710-120000"
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/loopscope/qwen17_mmlu_phase1.json")
            .read_text(encoding="utf-8")
        )
        jobs = prepare._build_jobs(
            root,
            prepare.DEFAULT_REMOTE_REPO,
            prepare.DEFAULT_REMOTE_REPO / ".venv-loopscope-cu121-20260710",
            config,
            generate_window_grid(28),
            root / "manifests/criterion_v0.json",
        )
        manifest = {
            "schema_version": prepare.RUN_SCHEMA_VERSION,
            "stages": {stage: {"jobs": values} for stage, values in jobs.items()},
        }
        manifest["manifest_sha256"] = prepare.manifest_sha256(manifest)
        expected_jobs = prepare._revision_jobs_for_stage(manifest, "gate-d-limit")
        report = {
            "schema_version": "loopscope.model-revision-check.v1",
            "run_manifest_sha256": manifest["manifest_sha256"],
            "stage": "gate-d-limit",
            "reference_policy": "all actual commit hashes must exactly match",
            "expected_jobs": [prepare._revision_job_contract(job) for job in expected_jobs],
            "observations": [
                {
                    "job_id": job["job_id"],
                    "stage": job["stage"],
                    "artifact": job["revision_artifact"],
                    "revision": "revision-a",
                }
                for job in expected_jobs
            ],
            "unique_revisions": ["revision-a"],
            "match": True,
        }
        report["manifest_sha256"] = prepare.manifest_sha256(report)
        self.assertEqual(
            prepare.validate_revision_report_payload(report, manifest, "gate-d-limit"),
            "revision-a",
        )
        report["observations"] = report["observations"][:-1]
        report["manifest_sha256"] = prepare.manifest_sha256(report)
        with self.assertRaises(prepare.PreparationError):
            prepare.validate_revision_report_payload(report, manifest, "gate-d-limit")

    def test_score_freeze_rejects_four_sample_prefix_of_larger_frozen_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _renderer_bundle(6)
            renderer = root / "renderer.json"
            renderer.write_text(json.dumps(bundle["manifest"]), encoding="utf-8")
            source = root / "source.jsonl"
            source.write_bytes(bundle["projection_bytes"])
            pool = root / "probe_pool.jsonl"
            pool_manifest_path = root / "probe_pool_manifest.json"
            build.main(
                [
                    "--input-jsonl", str(source),
                    "--output-jsonl", str(pool),
                    "--manifest", str(pool_manifest_path),
                    "--renderer-manifest", str(renderer),
                    "--source", bundle["manifest"]["dataset"]["source"],
                    "--split", "auxiliary_train",
                    "--count", "6",
                ],
                renderer_verifier=lambda projection, evidence: verify_export_bundle(
                    projection, evidence, backend=FakeRendererBackend(6)
                ),
            )
            pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
            layer_output = root / "gate-e-probe" / "probe-layers-full"
            window_output = root / "gate-e-probe" / "probe-windows-full"
            layer_output.mkdir(parents=True)
            window_output.mkdir(parents=True)
            prefix_pool = {
                "count": 4,
                "source_manifest_count": 6,
                "sample_ids": pool_manifest["sample_ids"][:4],
            }
            (layer_output / "probe_report.json").write_text(
                json.dumps({"probe_pool": prefix_pool}), encoding="utf-8"
            )
            (window_output / "probe_report.json").write_text(
                json.dumps({"probe_pool": prefix_pool}), encoding="utf-8"
            )
            run_manifest = {
                "inputs": {
                    "probe_pool": {
                        "snapshot_path": str(pool),
                        "manifest_snapshot_path": str(pool_manifest_path),
                        "count": 6,
                    }
                },
                "stages": {
                    "gate-e-probe": {
                        "jobs": [
                            {
                                "job_id": "probe-layers-full",
                                "output_dir": str(layer_output),
                            },
                            {
                                "job_id": "probe-windows-full",
                                "output_dir": str(window_output),
                            },
                        ]
                    }
                },
            }
            score = {
                "inputs": {
                    "paths": {
                        "layer_probe": str(layer_output / "probe_report.json"),
                        "window_probes": [str(window_output / "probe_report.json")],
                    }
                }
            }
            with self.assertRaisesRegex(prepare.PreparationError, "complete frozen pool"):
                prepare._validate_full_pool_score_inputs(run_manifest, score)


if __name__ == "__main__":
    unittest.main()
