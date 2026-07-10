import importlib.util
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from tflt.loopscope.grid import generate_window_grid
from tflt.loopscope.mmlu_renderer import create_renderer_bundle, verify_export_bundle
from tflt.loopscope.probe import load_probe_records, probe_pool_metadata
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
exporter = _load_script("export_mmlu_renderer.py")
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


def _write_full_freeze_fixture(root):
    run_root = root / "loopscope-qwen17-mmlu-phase1-fixture"
    manifests = run_root / "manifests"
    provenance = run_root / "control" / "provenance"
    manifests.mkdir(parents=True)
    provenance.mkdir(parents=True)

    bundle = _renderer_bundle(6)
    renderer_path = manifests / "renderer.json"
    renderer_path.write_text(json.dumps(bundle["manifest"]), encoding="utf-8")
    source_path = manifests / "source.jsonl"
    source_path.write_bytes(bundle["projection_bytes"])
    pool_path = manifests / "probe_pool.jsonl"
    pool_manifest_path = manifests / "probe_pool_manifest.json"
    build.main(
        [
            "--input-jsonl", str(source_path),
            "--output-jsonl", str(pool_path),
            "--manifest", str(pool_manifest_path),
            "--renderer-manifest", str(renderer_path),
            "--source", bundle["manifest"]["dataset"]["source"],
            "--split", "validation",
            "--count", "6",
        ],
        renderer_verifier=lambda projection, evidence: verify_export_bundle(
            projection, evidence, backend=FakeRendererBackend(6)
        ),
    )
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    records = load_probe_records(pool_path)
    pool_metadata, warnings = probe_pool_metadata(records, str(pool_manifest_path))
    if warnings:
        raise AssertionError("fixture unexpectedly emitted probe warnings: %r" % warnings)

    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs/loopscope/qwen17_mmlu_phase1.json").read_text(
            encoding="utf-8"
        )
    )
    criterion = json.loads(
        (repo_root / "configs/loopscope/criterion_v0.json").read_text(encoding="utf-8")
    )
    criterion_path = manifests / "criterion_v0.json"
    criterion_path.write_text(json.dumps(criterion), encoding="utf-8")
    grid = generate_window_grid(28)
    grid_path = manifests / "window_grid.json"
    grid_path.write_text(json.dumps(grid), encoding="utf-8")
    jobs = prepare._build_jobs(
        run_root,
        prepare.DEFAULT_REMOTE_REPO,
        prepare.DEFAULT_REMOTE_REPO / ".venv-loopscope-cu121-20260710",
        config,
        grid,
        criterion_path,
    )

    commit = prepare.MODEL_SNAPSHOT_COMMIT
    closure = {
        "schema_version": "loopscope.revision-closure.v1",
        "manifest_commit": commit,
        "model_commit": commit,
        "tokenizer_commit": commit,
        "match": True,
    }
    probe_paths = {}
    for job in jobs["gate-e-probe"]:
        path = Path(job["revision_artifact"])
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model": {"revision": commit},
                    "tokenizer": {"revision": commit},
                    "revision_closure": closure,
                    "probe_pool": pool_metadata,
                }
            ),
            encoding="utf-8",
        )
        probe_paths[job["job_id"]] = path

    score_job = jobs["gate-e-score"][0]
    score_path = Path(score_job["output_dir"]) / "window_scores.json"
    score_path.parent.mkdir(parents=True)
    score = {
        "schema_version": "loopscope.selection.v1",
        "criterion_sha256": prepare.manifest_sha256(criterion),
        "window_grid": {
            "manifest_sha256": grid["manifest_sha256"],
            "candidate_windows": [str(item["window"]) for item in grid["windows"]],
        },
        "inputs": {
            "paths": {
                "layer_probe": str(probe_paths["probe-layers-full"].resolve()),
                "window_probes": [str(probe_paths["probe-windows-full"].resolve())],
            },
            "layer_probe_manifest_sha256": pool_metadata["manifest_sha256"],
            "window_probe_manifest_sha256": [pool_metadata["manifest_sha256"]],
        },
        "probe_provenance": {
            "model": {
                "alias": "qwen3-1.7b-base",
                "repo_id": "Qwen/Qwen3-1.7B-Base",
                "revision": commit,
            },
            "tokenizer_revision": commit,
            "revision_closure": closure,
            "probe_pool": pool_metadata,
        },
    }
    score["manifest_sha256"] = prepare.manifest_sha256(score)
    score_path.write_text(json.dumps(score), encoding="utf-8")

    manifest = {
        "schema_version": prepare.RUN_SCHEMA_VERSION,
        "run_root": str(run_root.resolve()),
        "frozen_recipe": {"revision": commit},
        "revision_policy": {"cli_pins_revision": True, "manifest_commit": commit},
        "inputs": {
            "window_grid": {
                "snapshot_path": str(grid_path.resolve()),
                "manifest_sha256": grid["manifest_sha256"],
            },
            "criterion": {
                "snapshot_path": str(criterion_path.resolve()),
                "file_sha256": hashlib.sha256(criterion_path.read_bytes()).hexdigest(),
                "criterion_sha256": prepare.manifest_sha256(criterion),
            },
            "probe_pool": {
                "snapshot_path": str(pool_path.resolve()),
                "manifest_snapshot_path": str(pool_manifest_path.resolve()),
                "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                "manifest_sha256": pool_manifest["manifest_sha256"],
                "count": pool_manifest["count"],
                "render_contract_sha256": pool_manifest["renderer"][
                    "render_contract_sha256"
                ],
                "render_contract_subset_sha256": pool_manifest[
                    "render_contract_subset_sha256"
                ],
            },
        },
        "stages": {stage: {"jobs": stage_jobs} for stage, stage_jobs in jobs.items()},
    }
    manifest["manifest_sha256"] = prepare.manifest_sha256(manifest)
    manifest_path = run_root / "control" / "phase1_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    expected_jobs = prepare._revision_jobs_for_stage(manifest, "gate-e-probe")
    revision_report = {
        "schema_version": prepare.REVISION_REPORT_SCHEMA_VERSION,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "stage": "gate-e-probe",
        "manifest_commit": commit,
        "reference_policy": "model, tokenizer, and manifest commits must all exactly match",
        "expected_jobs": [prepare._revision_job_contract(job) for job in expected_jobs],
        "observations": [
            {
                "job_id": job["job_id"],
                "stage": job["stage"],
                "artifact": job["revision_artifact"],
                "model_commit": commit,
                "tokenizer_commit": commit,
                "manifest_commit": commit,
                "match": True,
            }
            for job in expected_jobs
        ],
        "unique_revisions": [commit],
        "match": True,
    }
    revision_report["manifest_sha256"] = prepare.manifest_sha256(revision_report)
    (provenance / "gate-e-probe-model-revisions.json").write_text(
        json.dumps(revision_report), encoding="utf-8"
    )
    prepare.freeze_score(run_root.resolve())
    return run_root.resolve(), probe_paths


class LoopScopePhaseOneScriptTest(unittest.TestCase):
    PLANNING_THREAD_ID = "019f4c7b-e5eb-77f2-b007-59d004896550"

    def _prepare_cli_args(self):
        return [
            "prepare",
            "--planning-thread-id", self.PLANNING_THREAD_ID,
            "--venv", str(
                prepare.DEFAULT_REMOTE_REPO / ".venv-loopscope-cu121-20260711"
            ),
            "--window-grid", "/tmp/window_grid.json",
            "--probe-pool-jsonl", "/tmp/probe_pool.jsonl",
            "--probe-pool-manifest", "/tmp/probe_pool_manifest.json",
            "--timestamp", "20260711-120000",
            "--dry-run",
        ]

    def test_prepare_requires_planning_thread_id(self):
        argv = self._prepare_cli_args()
        index = argv.index("--planning-thread-id")
        del argv[index:index + 2]
        with self.assertRaises(SystemExit):
            prepare.parse_args(argv)

    def test_prepare_rejects_noncanonical_planning_thread_id(self):
        argv = self._prepare_cli_args()
        argv[argv.index("--planning-thread-id") + 1] = self.PLANNING_THREAD_ID.upper()
        with self.assertRaises(SystemExit):
            prepare.parse_args(argv)

    def test_prepare_requires_explicit_venv(self):
        argv = self._prepare_cli_args()
        index = argv.index("--venv")
        del argv[index:index + 2]
        with self.assertRaises(SystemExit):
            prepare.parse_args(argv)

    def test_prepare_preserves_exact_planning_thread_id_in_manifest(self):
        args = prepare.parse_args(self._prepare_cli_args())
        repo_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (repo_root / "configs/loopscope/qwen17_mmlu_phase1.json").read_text(
                encoding="utf-8"
            )
        )
        criterion = json.loads(
            (repo_root / "configs/loopscope/criterion_v0.json").read_text(
                encoding="utf-8"
            )
        )
        grid = generate_window_grid(28)
        pool_manifest = {
            "count": 1,
            "seed": 20260710,
            "split": "validation",
            "manifest_sha256": "a" * 64,
            "render_contract_subset_sha256": "b" * 64,
            "renderer": {"render_contract_sha256": "c" * 64},
        }

        def read_json(path):
            name = Path(path).name
            if name == "qwen17_mmlu_phase1.json":
                return config
            if name == "criterion_v0.json":
                return criterion
            if name == "window_grid.json":
                return grid
            if name == "probe_pool_manifest.json":
                return pool_manifest
            raise AssertionError("unexpected JSON path: %s" % path)

        jobs = {
            stage: []
            for stage in (
                "gate-c", "gate-d-limit", "gate-e-probe", "gate-e-score", "gate-e-full"
            )
        }
        output = io.StringIO()
        with patch.object(prepare, "_git_provenance", return_value={
            "root": str(prepare.DEFAULT_REMOTE_REPO),
            "branch": "loopscope",
            "commit": "d" * 40,
            "dirty": False,
        }), patch.object(
            prepare,
            "_resolve_repo_path",
            side_effect=lambda root, requested: root / requested,
        ), patch.object(prepare, "_read_json", side_effect=read_json), patch.object(
            prepare, "_validate_pool", return_value=b"{}\n"
        ), patch.object(prepare, "_file_sha256", return_value="e" * 64), patch.object(
            prepare, "_build_jobs", return_value=jobs
        ), redirect_stdout(output):
            self.assertEqual(prepare.prepare_run(args), 0)
        manifest = json.loads(output.getvalue())
        self.assertEqual(manifest["planning_thread_id"], self.PLANNING_THREAD_ID)

    def test_approval_contract_accepts_exact_thread_match(self):
        manifest = {"planning_thread_id": self.PLANNING_THREAD_ID}
        approval = {"planning_thread_id": self.PLANNING_THREAD_ID}
        self.assertEqual(
            prepare.validate_approval_contract(manifest, approval),
            self.PLANNING_THREAD_ID,
        )

    def test_approval_contract_rejects_mismatch(self):
        manifest = {"planning_thread_id": self.PLANNING_THREAD_ID}
        approval = {"planning_thread_id": "019f4d58-42a4-73b2-9b5d-6c64ac68f43b"}
        with self.assertRaisesRegex(prepare.PreparationError, "exactly match"):
            prepare.validate_approval_contract(manifest, approval)

    def test_approval_contract_rejects_missing_manifest_thread(self):
        with self.assertRaisesRegex(prepare.PreparationError, "canonical lowercase UUID"):
            prepare.validate_approval_contract(
                {}, {"planning_thread_id": self.PLANNING_THREAD_ID}
            )

    def test_historical_planning_thread_id_is_absent_from_source_docs_and_tests(self):
        historical = "019f4b5a" + "-79ac-78c1-9196-c7fd733cf04d"
        repo_root = Path(__file__).resolve().parents[1]
        for root_name in ("src", "scripts", "docs", "tests"):
            for path in (repo_root / root_name).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".sh", ".md", ".toml", ".json"}:
                    self.assertNotIn(
                        historical,
                        path.read_text(encoding="utf-8"),
                        str(path),
                    )

    def _assert_post_freeze_mutation_blocks_submission(self, job_id):
        with tempfile.TemporaryDirectory() as tmp:
            run_root, probe_paths = _write_full_freeze_fixture(Path(tmp))
            prepare.validate_full_submission_freeze(run_root)
            with probe_paths[job_id].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            attempt = run_root / "control" / "submissions" / "gate-e-full-attempt.json"
            attempt.parent.mkdir()
            sbatch = Mock()

            def guarded_submit():
                prepare.validate_full_submission_freeze(run_root)
                attempt.write_text("attempt", encoding="utf-8")
                sbatch()

            with self.assertRaisesRegex(
                prepare.PreparationError, "full-probe evidence mismatch"
            ):
                guarded_submit()
            self.assertFalse(attempt.exists())
            sbatch.assert_not_called()

    def test_gate_e_full_rejects_post_freeze_layer_probe_mutation_before_attempt_or_sbatch(self):
        self._assert_post_freeze_mutation_blocks_submission("probe-layers-full")

    def test_gate_e_full_rejects_post_freeze_window_probe_mutation_before_attempt_or_sbatch(self):
        self._assert_post_freeze_mutation_blocks_submission("probe-windows-full")

    def test_submit_shell_calls_freeze_helper_before_attempt_and_sbatch(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/loopscope/submit_qwen17_phase1.sh"
        ).read_text(encoding="utf-8")
        helper_index = script.index("validate-full-freeze --run-root")
        contract_index = script.index("validate-approval-contract")
        attempt_index = script.index('"$attempt" "$stage"')
        sbatch_index = script.index('sbatch --parsable "$runner"')
        self.assertLess(contract_index, attempt_index)
        self.assertLess(contract_index, sbatch_index)
        self.assertLess(helper_index, attempt_index)
        self.assertLess(helper_index, sbatch_index)
        self.assertNotIn('freeze.get("full_probe_evidence")', script)

    def test_phase_config_requires_exact_model_revision(self):
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/loopscope/qwen17_mmlu_phase1.json")
            .read_text(encoding="utf-8")
        )
        del config["model"]["revision"]
        with self.assertRaises(prepare.PreparationError):
            prepare._validate_phase_config(config)

    def test_phase_config_freezes_validation_calibration_targets(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/loopscope/qwen17_mmlu_phase1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["probe_pool"].get("target_split"), "validation")
        for value in (None, "auxiliary_train", "dev", "test"):
            with self.subTest(value=value):
                mutated = json.loads(json.dumps(config))
                if value is None:
                    mutated["probe_pool"].pop("target_split")
                else:
                    mutated["probe_pool"]["target_split"] = value
                with self.assertRaisesRegex(
                    prepare.PreparationError, "probe_pool.target_split"
                ):
                    prepare._validate_phase_config(mutated)

    def test_exporter_and_builder_cli_freeze_validation_target_split(self):
        export_args = exporter.parse_args(
            [
                "--output-jsonl",
                "projection.jsonl",
                "--manifest",
                "renderer.json",
                "--dataset-revision",
                FAKE_DATASET_REVISION,
            ]
        )
        self.assertEqual(export_args.target_split, "validation")
        builder_args = [
            "--input-jsonl", "projection.jsonl",
            "--output-jsonl", "pool.jsonl",
            "--renderer-manifest", "renderer.json",
            "--source", "cais/mmlu@" + FAKE_DATASET_REVISION,
            "--split", "validation",
        ]
        self.assertEqual(build.parse_args(builder_args).split, "validation")
        split_index = builder_args.index("--split") + 1
        for split in ("", "auxiliary_train", "dev", "test"):
            with self.subTest(split=split):
                invalid = list(builder_args)
                invalid[split_index] = split
                with self.assertRaises(SystemExit):
                    build.parse_args(invalid)

    def test_prepare_pool_rejects_every_nonvalidation_manifest_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_renderer_bundle(
                dataset_revision=FAKE_DATASET_REVISION,
                target_split="validation",
                task_names=["mmlu_math"],
                max_targets_per_task=4,
                backend=FakeRendererBackend(4),
            )
            renderer = root / "renderer.json"
            renderer.write_text(json.dumps(bundle["manifest"]), encoding="utf-8")
            source = root / "source.jsonl"
            source.write_bytes(bundle["projection_bytes"])
            pool = root / "pool.jsonl"
            manifest_path = root / "pool-manifest.json"
            self.assertEqual(
                build.main(
                    [
                        "--input-jsonl", str(source),
                        "--output-jsonl", str(pool),
                        "--manifest", str(manifest_path),
                        "--renderer-manifest", str(renderer),
                        "--source", bundle["manifest"]["dataset"]["source"],
                        "--split", "validation",
                        "--count", "4",
                    ],
                    renderer_verifier=lambda projection, evidence: verify_export_bundle(
                        projection, evidence, backend=FakeRendererBackend(4)
                    ),
                ),
                0,
            )
            valid = json.loads(manifest_path.read_text(encoding="utf-8"))
            prepare._validate_pool(pool, valid)
            for split in ("auxiliary_train", "dev", "test"):
                with self.subTest(split=split):
                    invalid = json.loads(json.dumps(valid))
                    invalid["split"] = split
                    invalid["manifest_sha256"] = build.manifest_sha256(invalid)
                    with self.assertRaisesRegex(
                        prepare.PreparationError, "split.*validation|validation.*split"
                    ):
                        prepare._validate_pool(pool, invalid)

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
                    "--split", "validation",
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
                "validation",
                renderer,
            )

    def test_builder_preserves_unicode_next_line_inside_json_strings(self):
        bundle = _renderer_bundle(1)
        item = _source_record(0)
        item["text"] = item["text"].replace("Answer:", "next\u0085line\nAnswer:", 1)
        item["render_sha256"] = hashlib.sha256(
            item["text"].encode("utf-8")
        ).hexdigest()
        body = {
            key: value for key, value in item.items() if key != "projection_record_sha256"
        }
        item["projection_record_sha256"] = hashlib.sha256(
            build.canonical_json_bytes(body)
        ).hexdigest()
        raw = build.canonical_json_bytes(item) + b"\n"
        records = build._load_source_records(
            raw,
            bundle["manifest"]["dataset"]["source"],
            "validation",
            bundle["manifest"],
        )
        self.assertEqual(len(records), 1)
        self.assertIn("\u0085", records[0]["text"])

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
        for stage in ("gate-c", "gate-d-limit", "gate-e-probe", "gate-e-full"):
            for job in jobs[stage]:
                self.assertIn("--revision", job["argv"])
                revision_index = job["argv"].index("--revision")
                self.assertEqual(
                    job["argv"][revision_index + 1], prepare.MODEL_SNAPSHOT_COMMIT
                )
                self.assertEqual(
                    set(job["revision_keys"]),
                    {"model_commit", "tokenizer_commit", "manifest_commit"},
                )

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
            "frozen_recipe": {"revision": prepare.MODEL_SNAPSHOT_COMMIT},
            "revision_policy": {
                "cli_pins_revision": True,
                "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
            },
            "stages": {stage: {"jobs": values} for stage, values in jobs.items()},
        }
        manifest["manifest_sha256"] = prepare.manifest_sha256(manifest)
        expected_jobs = prepare._revision_jobs_for_stage(manifest, "gate-d-limit")
        report = {
            "schema_version": prepare.REVISION_REPORT_SCHEMA_VERSION,
            "run_manifest_sha256": manifest["manifest_sha256"],
            "stage": "gate-d-limit",
            "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
            "reference_policy": "model, tokenizer, and manifest commits must all exactly match",
            "expected_jobs": [prepare._revision_job_contract(job) for job in expected_jobs],
            "observations": [
                {
                    "job_id": job["job_id"],
                    "stage": job["stage"],
                    "artifact": job["revision_artifact"],
                    "model_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                    "tokenizer_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                    "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                    "match": True,
                }
                for job in expected_jobs
            ],
            "unique_revisions": [prepare.MODEL_SNAPSHOT_COMMIT],
            "match": True,
        }
        report["manifest_sha256"] = prepare.manifest_sha256(report)
        self.assertEqual(
            prepare.validate_revision_report_payload(report, manifest, "gate-d-limit"),
            prepare.MODEL_SNAPSHOT_COMMIT,
        )
        report["observations"] = report["observations"][:-1]
        report["manifest_sha256"] = prepare.manifest_sha256(report)
        with self.assertRaises(prepare.PreparationError):
            prepare.validate_revision_report_payload(report, manifest, "gate-d-limit")

    def test_revision_report_rejects_probe_eval_mismatch(self):
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
            "frozen_recipe": {"revision": prepare.MODEL_SNAPSHOT_COMMIT},
            "revision_policy": {
                "cli_pins_revision": True,
                "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
            },
            "stages": {stage: {"jobs": values} for stage, values in jobs.items()},
        }
        manifest["manifest_sha256"] = prepare.manifest_sha256(manifest)
        expected_jobs = prepare._revision_jobs_for_stage(manifest, "gate-d-limit")
        observations = []
        for job in expected_jobs:
            observations.append(
                {
                    "job_id": job["job_id"],
                    "stage": job["stage"],
                    "artifact": job["revision_artifact"],
                    "model_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                    "tokenizer_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                    "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                    "match": True,
                }
            )
        observations[-1]["tokenizer_commit"] = "b" * 40
        observations[-1]["match"] = False
        report = {
            "schema_version": prepare.REVISION_REPORT_SCHEMA_VERSION,
            "run_manifest_sha256": manifest["manifest_sha256"],
            "stage": "gate-d-limit",
            "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
            "reference_policy": "model, tokenizer, and manifest commits must all exactly match",
            "expected_jobs": [prepare._revision_job_contract(job) for job in expected_jobs],
            "observations": observations,
            "unique_revisions": [prepare.MODEL_SNAPSHOT_COMMIT, "b" * 40],
            "match": False,
        }
        report["manifest_sha256"] = prepare.manifest_sha256(report)
        with self.assertRaisesRegex(prepare.PreparationError, "do not match"):
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
                    "--split", "validation",
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
                "frozen_recipe": {"revision": prepare.MODEL_SNAPSHOT_COMMIT},
                "revision_policy": {
                    "cli_pins_revision": True,
                    "manifest_commit": prepare.MODEL_SNAPSHOT_COMMIT,
                },
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
