import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tflt.loopscope import phase2_analysis as p2a
from tflt.loopscope import phase2_reuse as p2r
from tflt.loopscope import phase2_schema as p2s
from tflt.loopscope import phase2_trajectory as p2t
from tflt.loopscope.phase2_analysis import choice_output
from tflt.loopscope.phase2_schema import (
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    SchemaError,
    atomic_write_new_json,
    file_sha256,
    make_full_final_output_envelope,
    make_hashed_manifest,
    make_identity_manifest,
    make_source_provenance,
    protocol_cell_id,
)
from tflt.loopscope.schema import attach_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json"


def _identity(index):
    return {
        "task": "mmlu_subject",
        "doc_id": str(index),
        "doc_hash": "%064x" % (index + 1),
    }


def _producer(kind):
    payload = {
        "producer_kind": kind,
        "attempt_manifest_sha256": "1" * 64,
        "receipt_manifest_sha256": "2" * 64,
        "command_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
        "revision_report_sha256": "5" * 64,
    }
    if kind == "lm_eval_logged_samples_adapter":
        payload["results_sha256"] = "6" * 64
    elif kind == "phase1_immutable_reuse_adapter":
        payload["results_sha256"] = "6" * 64
        payload["source_manifest_sha256"] = "7" * 64
    return payload


def _evidence(kind):
    root = (
        "/hpc2hdd/home/xhuang225/workspaces/"
        "training_free_looped_transformers_loopscope/runs/provenance-red"
    )
    ref = lambda name, digest: {"path": "%s/%s" % (root, name), "sha256": digest}
    if kind == "lm_eval_logged_samples_adapter":
        results = ref("results.json", "6" * 64)
        return {
            "adapter_request": ref("provenance/request.json", "8" * 64),
            "attempt_manifest": ref("provenance/attempt.json", "1" * 64),
            "receipt_manifest": ref("provenance/receipt.json", "2" * 64),
            "command_args": ref("command_args.json", "3" * 64),
            "environment": ref("env.json", "4" * 64),
            "revision_report": ref("model_revision.json", "5" * 64),
            "results": results,
            "sample_source": {"kind": "results_inline_samples", "artifacts": [results]},
        }
    phase1 = (
        "/hpc2hdd/home/xhuang225/workspaces/"
        "training_free_looped_transformers/runs/"
        "loopscope-qwen17-mmlu-phase1-20260711-053022"
    )
    source = lambda name, digest: {"path": "%s/%s" % (phase1, name), "sha256": digest}
    results = source("gate-e-full/baseline-full/results.json", "6" * 64)
    return {
        "reuse_request": ref("provenance/reuse-request.json", "8" * 64),
        "attempt_manifest": ref("provenance/attempt.json", "1" * 64),
        "receipt_manifest": ref("provenance/receipt.json", "2" * 64),
        "command_args": ref("command_args.json", "3" * 64),
        "environment": ref("env.json", "4" * 64),
        "source_run_manifest": source("control/phase1_run_manifest.json", "7" * 64),
        "source_command_args": source("gate-e-full/baseline-full/command_args.json", "9" * 64),
        "source_environment": source("gate-e-full/baseline-full/env.json", "a" * 64),
        "source_revision_report": source("gate-e-full/baseline-full/model_revision.json", "5" * 64),
        "source_results": results,
        "sample_source": {"kind": "results_inline_samples", "artifacts": [results]},
    }


def _cell(protocol, window, k, alpha):
    return {
        "cell_id": protocol_cell_id(protocol, window, k, alpha),
        "protocol": protocol,
        "window": window,
        "k": k,
        "alpha": alpha,
        "step_size": alpha / k,
    }


def _reuse_cells():
    return [
        _cell("baseline_no_loop", "none", 1, 1.0),
        _cell("shared_k2_anchor", "11:14", 2, 1.0),
        _cell("shared_k2_anchor", "12:15", 2, 1.0),
        _cell("shared_k2_anchor", "13:16", 2, 1.0),
    ]


def _new_cells():
    cells = []
    for window in ("11:14", "12:15", "13:16"):
        cells.extend(
            (_cell("fixed_step", window, 3, 1.5), _cell("fixed_step", window, 4, 2.0))
        )
    cells.extend(
        (_cell("fixed_horizon", "12:15", 3, 1.0), _cell("fixed_horizon", "12:15", 4, 1.0))
    )
    return cells


def _card_for_roots(card, *, phase1_root=None, workspace=None):
    result = copy.deepcopy(card)
    if phase1_root is not None:
        result["phase1_reuse_policy"]["phase1_run_manifest_path"] = str(
            Path(phase1_root) / "control" / "phase1_run_manifest.json"
        )
    if workspace is not None:
        result["write_once_contract"]["workspace_root"] = str(workspace)
    result.pop("manifest_sha256", None)
    attach_manifest_sha256(result)
    return result


def _phase1_job_argv(card, cell, output_root):
    argv = [
        "python", "-m", "tflt.eval_runner", "--model", "qwen3-1.7b-base",
        "--revision", card["science"]["revision"], "--tasks", "mmlu",
        "--output-dir", str(output_root), "--num-fewshot", "5",
        "--batch-size", "auto", "--dtype", "float16",
    ]
    if cell["protocol"] != "baseline_no_loop":
        argv.extend(
            [
                "--loop", "--window", cell["window"], "--k", "2",
                "--iteration-mode", "block", "--strategy", "damped_euler",
                "--alpha", "1.0", "--beta", "0.0", "--cache-strategy", "last",
                "--decode-mode", "bypass",
            ]
        )
    return argv


def _phase1_command(card, cell, output_root):
    command = {
        "model": "qwen3-1.7b-base",
        "revision": card["science"]["revision"],
        "tasks": "mmlu",
        "output_dir": str(output_root),
        "limit": None,
        "num_fewshot": 5,
        "batch_size": "auto",
        "dtype": "float16",
        "loop": cell["protocol"] != "baseline_no_loop",
        "window": "12:15",
        "k": 2,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "first_n": None,
    }
    if cell["protocol"] != "baseline_no_loop":
        command["window"] = cell["window"]
    return command


def _build_phase1_fixture(root, card, identities, raw_rows):
    run_root = root / "phase1-run"
    results_payload = {"results": {"mmlu": {"acc,none": 1.0}}, "samples": {"mmlu_subject": raw_rows}}
    revision = {
        "repo_id": card["science"]["model"],
        "model_commit": card["science"]["revision"],
        "tokenizer_commit": card["science"]["revision"],
        "manifest_commit": card["science"]["revision"],
        "match": True,
    }
    jobs = []
    baseline_results = None
    for cell in _reuse_cells():
        job_id = p2r.PHASE1_REUSE_JOB_BY_CELL[cell["cell_id"]]
        output = run_root / "gate-e-full" / job_id
        output.mkdir(parents=True)
        atomic_write_new_json(output / "command_args.json", _phase1_command(card, cell, output))
        atomic_write_new_json(output / "env.json", {"VIRTUAL_ENV": "/fixture"})
        atomic_write_new_json(output / "model_revision.json", revision)
        if baseline_results is None:
            atomic_write_new_json(output / "results.json", results_payload)
            baseline_results = output / "results.json"
        else:
            os.link(baseline_results, output / "results.json")
        argv = _phase1_job_argv(card, cell, output)
        jobs.append(
            {
                "job_id": job_id,
                "stage": "gate-e-full",
                "output_dir": str(output),
                "claim_dir": str(output) + ".claim",
                "argv": argv,
                "command": " ".join(argv),
                "automatic_retry": False,
                "revision_artifact": str(output / "model_revision.json"),
                "revision_keys": {
                    "model_commit": ["model_commit"],
                    "tokenizer_commit": ["tokenizer_commit"],
                    "manifest_commit": ["manifest_commit"],
                },
            }
        )
    manifest = make_hashed_manifest({"stages": {"gate-e-full": {"jobs": jobs}}})
    manifest_path = run_root / "control" / "phase1_run_manifest.json"
    atomic_write_new_json(manifest_path, manifest)
    source = make_source_provenance(
        identity_namespace=FULL_IDENTITY_NAMESPACE,
        verification_status="live_verified",
        source_manifest_path=str(manifest_path),
        source_manifest_sha256=manifest["manifest_sha256"],
        renderer_subset_sha256="b" * 64,
        identity_artifacts=[{"path": str(baseline_results), "sha256": file_sha256(baseline_results)}],
        phase1_run_root=str(run_root),
    )
    identity_manifest = make_identity_manifest(
        card,
        identity_namespace=FULL_IDENTITY_NAMESPACE,
        split="mmlu_test_full",
        identities=identities,
        source_provenance=source,
    )
    return run_root, manifest_path, identity_manifest


def _build_new_full_fixture(root, workspace, card, identity_manifest, cell, raw_rows, samples):
    authorization_root = workspace / "staging" / cell["cell_id"]
    provenance = authorization_root / "provenance"
    output = authorization_root / "output"
    provenance.mkdir(parents=True)
    output.mkdir()
    card_path = provenance / "card.json"
    identity_path = provenance / "identity.json"
    atomic_write_new_json(card_path, card)
    atomic_write_new_json(identity_path, identity_manifest)
    common = {
        "producer_kind": "lm_eval_logged_samples_adapter",
        "card_manifest_sha256": card["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "revision": card["science"]["revision"],
        "executor_thread_id": "executor-fixture",
        "authorization_root": str(authorization_root.resolve()),
        "output_root": str(output.resolve()),
    }
    attempt = make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-full-attempt.v2",
            "artifact_kind": "full_final_output_attempt",
            **common,
            "created_at_utc": "2026-07-14T03:00:00Z",
        }
    )
    attempt_path = provenance / "attempt.json"
    atomic_write_new_json(attempt_path, attempt)
    receipt = make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-full-receipt.v2",
            "artifact_kind": "full_final_output_execution_receipt",
            **common,
            "attempt_manifest_sha256": attempt["manifest_sha256"],
            "created_at_utc": "2026-07-14T03:01:00Z",
        }
    )
    receipt_path = provenance / "receipt.json"
    atomic_write_new_json(receipt_path, receipt)
    request = make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-final-output-adapter-manifest.v2",
            "artifact_kind": "full_final_output_adapter_request",
            "card": {"path": str(card_path.resolve()), "sha256": card["manifest_sha256"]},
            "identity_manifest": {
                "path": str(identity_path.resolve()),
                "sha256": identity_manifest["manifest_sha256"],
            },
            "cell": cell,
            "attempt_manifest": {
                "path": str(attempt_path.resolve()),
                "sha256": attempt["manifest_sha256"],
            },
            "receipt_manifest": {
                "path": str(receipt_path.resolve()),
                "sha256": receipt["manifest_sha256"],
            },
            "identity_join": "unordered_exact_task_doc_id_doc_hash_to_canonical_manifest",
            "score_source": FULL_LM_EVAL_SCORE_SOURCE,
            "output_filename": "phase2_final_outputs.json",
        }
    )
    request_path = provenance / "request.json"
    atomic_write_new_json(request_path, request)
    command = {
        "model": "qwen3-1.7b-base",
        "revision": card["science"]["revision"],
        "tasks": "mmlu",
        "output_dir": str(output.resolve()),
        "limit": None,
        "num_fewshot": 5,
        "batch_size": "auto",
        "dtype": "float16",
        "loop": True,
        "window": cell["window"],
        "k": cell["k"],
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": cell["alpha"],
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "first_n": None,
        "phase2_final_output_manifest": str(request_path.resolve()),
    }
    atomic_write_new_json(output / "command_args.json", command)
    atomic_write_new_json(output / "env.json", {"VIRTUAL_ENV": "/fixture"})
    atomic_write_new_json(
        output / "model_revision.json",
        {
            "repo_id": card["science"]["model"],
            "model_commit": card["science"]["revision"],
            "tokenizer_commit": card["science"]["revision"],
            "manifest_commit": card["science"]["revision"],
            "match": True,
        },
    )
    atomic_write_new_json(
        output / "results.json",
        {"results": {"mmlu": {"acc,none": 1.0}}, "samples": {"mmlu_subject": raw_rows}},
    )
    ref = lambda path: {"path": str(path.resolve()), "sha256": file_sha256(path)}
    results_ref = ref(output / "results.json")
    evidence = {
        "adapter_request": {"path": str(request_path.resolve()), "sha256": request["manifest_sha256"]},
        "attempt_manifest": {"path": str(attempt_path.resolve()), "sha256": attempt["manifest_sha256"]},
        "receipt_manifest": {"path": str(receipt_path.resolve()), "sha256": receipt["manifest_sha256"]},
        "command_args": ref(output / "command_args.json"),
        "environment": ref(output / "env.json"),
        "revision_report": ref(output / "model_revision.json"),
        "results": results_ref,
        "sample_source": {"kind": "results_inline_samples", "artifacts": [results_ref]},
    }
    producer = {
        "producer_kind": "lm_eval_logged_samples_adapter",
        "attempt_manifest_sha256": attempt["manifest_sha256"],
        "receipt_manifest_sha256": receipt["manifest_sha256"],
        "command_sha256": evidence["command_args"]["sha256"],
        "environment_sha256": evidence["environment"]["sha256"],
        "revision_report_sha256": evidence["revision_report"]["sha256"],
        "results_sha256": results_ref["sha256"],
    }
    envelope = make_full_final_output_envelope(
        card,
        identity_manifest,
        cell=cell,
        samples=samples,
        producer=producer,
        artifact_root=str(output.resolve()),
        producer_evidence=evidence,
    )
    sidecar_path = output / "phase2_final_outputs.json"
    atomic_write_new_json(sidecar_path, envelope)
    return {
        "authorization_root": authorization_root,
        "output": output,
        "sidecar_path": sidecar_path,
        "envelope": envelope,
        "evidence": evidence,
    }


class Phase2ProvenanceRepairRedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
        identities = [_identity(index) for index in range(14042)]
        cls.identities = identities
        cls.raw_rows = [
            {
                "doc_id": item["doc_id"],
                "doc_hash": item["doc_hash"],
                "filtered_resps": [4.0, 3.0, 2.0, 1.0],
                "target": 0,
                "metrics": {"acc,none": 1},
            }
            for item in identities
        ]
        cls.identity_manifest = make_identity_manifest(
            cls.card,
            identity_namespace=FULL_IDENTITY_NAMESPACE,
            split="mmlu_test_full",
            identities=identities,
            source_provenance=make_source_provenance(
                identity_namespace=FULL_IDENTITY_NAMESPACE,
                verification_status="live_verified",
                source_manifest_path=(
                    "/hpc2hdd/home/xhuang225/workspaces/"
                    "training_free_looped_transformers/runs/"
                    "loopscope-qwen17-mmlu-phase1-20260711-053022/"
                    "control/phase1_run_manifest.json"
                ),
                source_manifest_sha256="a" * 64,
                renderer_subset_sha256="b" * 64,
                identity_artifacts=[{
                    "path": (
                        "/hpc2hdd/home/xhuang225/workspaces/"
                        "training_free_looped_transformers/runs/"
                        "loopscope-qwen17-mmlu-phase1-20260711-053022/"
                        "gate-e-full/baseline-full/results.json"
                    ),
                    "sha256": "c" * 64,
                }],
            ),
        )
        cls.samples = [
            choice_output(
                [4.0, 3.0, 2.0, 1.0],
                identity=item,
                score_source=FULL_LM_EVAL_SCORE_SOURCE,
                gold_index=0,
                evaluator_acc=True,
            )
            for item in identities
        ]

    def test_new_run_cell_cannot_masquerade_as_phase1_reuse(self):
        cell = {
            "cell_id": protocol_cell_id("fixed_step", "11:14", 3, 1.5),
            "protocol": "fixed_step",
            "window": "11:14",
            "k": 3,
            "alpha": 1.5,
            "step_size": 0.5,
        }
        with self.assertRaises(SchemaError):
            make_full_final_output_envelope(
                self.card,
                self.identity_manifest,
                cell=cell,
                samples=self.samples,
                producer=_producer("phase1_immutable_reuse_adapter"),
                artifact_root=(
                    "/hpc2hdd/home/xhuang225/workspaces/"
                    "training_free_looped_transformers_loopscope/runs/provenance-red"
                ),
                producer_evidence=_evidence("phase1_immutable_reuse_adapter"),
            )

    def test_reuse_cell_cannot_masquerade_as_new_run(self):
        cell = {
            "cell_id": protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0),
            "protocol": "shared_k2_anchor",
            "window": "12:15",
            "k": 2,
            "alpha": 1.0,
            "step_size": 0.5,
        }
        with self.assertRaises(SchemaError):
            make_full_final_output_envelope(
                self.card,
                self.identity_manifest,
                cell=cell,
                samples=self.samples,
                producer=_producer("lm_eval_logged_samples_adapter"),
                artifact_root=(
                    "/hpc2hdd/home/xhuang225/workspaces/"
                    "training_free_looped_transformers_loopscope/runs/provenance-red"
                ),
                producer_evidence=_evidence("lm_eval_logged_samples_adapter"),
            )

    def test_partition_covers_exactly_all_four_reuse_and_eight_new_cells(self):
        partition = p2r.full_cell_producer_partition(self.card)
        self.assertEqual(
            partition["phase1_immutable_reuse_adapter"],
            tuple(cell["cell_id"] for cell in _reuse_cells()),
        )
        self.assertEqual(
            partition["lm_eval_logged_samples_adapter"],
            tuple(cell["cell_id"] for cell in _new_cells()),
        )
        for cell in _reuse_cells():
            self.assertEqual(p2s.expected_full_producer_kind(cell), "phase1_immutable_reuse_adapter")
        for cell in _new_cells():
            self.assertEqual(p2s.expected_full_producer_kind(cell), "lm_eval_logged_samples_adapter")

    def test_real_phase1_loader_closes_all_four_reuse_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "phase1-run"
            card = _card_for_roots(self.card, phase1_root=run_root)
            with mock.patch.object(p2s, "PHASE1_RUN_ROOT", str(run_root)), mock.patch.object(
                p2r, "PHASE1_RUN_ROOT", str(run_root)
            ):
                _, manifest_path, identity_manifest = _build_phase1_fixture(
                    Path(tmp), card, self.identities, self.raw_rows
                )
                for cell in _reuse_cells():
                    loaded = p2r.load_phase1_immutable_reuse_source(
                        card, identity_manifest, cell, manifest_path
                    )
                    self.assertEqual(loaded["job_id"], p2r.PHASE1_REUSE_JOB_BY_CELL[cell["cell_id"]])
                    self.assertEqual(len(loaded["samples"]), 14042)
                    self.assertEqual(
                        [row["sample_identity"] for row in loaded["samples"]],
                        self.identities,
                    )
                    self.assertEqual(
                        loaded["evidence"]["sample_source"]["kind"],
                        "results_inline_samples",
                    )
                with self.assertRaises(p2r.Phase2ReuseError):
                    p2r.load_phase1_immutable_reuse_source(
                        card, identity_manifest, _new_cells()[0], manifest_path
                    )

    def test_phase1_loader_uses_exact_sorted_sidecars_when_results_omits_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "phase1-run"
            card = _card_for_roots(self.card, phase1_root=run_root)
            with mock.patch.object(p2s, "PHASE1_RUN_ROOT", str(run_root)), mock.patch.object(
                p2r, "PHASE1_RUN_ROOT", str(run_root)
            ):
                _, manifest_path, _ = _build_phase1_fixture(
                    Path(tmp), card, self.identities, self.raw_rows
                )
                output = run_root / "gate-e-full" / "baseline-full"
                results_path = output / "results.json"
                results_path.write_text(
                    json.dumps({"results": {"mmlu_subject": {"acc,none": 1.0}}}),
                    encoding="utf-8",
                )
                sidecar_path = output / "samples_mmlu_subject_20260714.jsonl"
                sidecar_path.write_text(
                    "\n".join(json.dumps(row) for row in self.raw_rows) + "\n",
                    encoding="utf-8",
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                source = make_source_provenance(
                    identity_namespace=FULL_IDENTITY_NAMESPACE,
                    verification_status="live_verified",
                    source_manifest_path=str(manifest_path),
                    source_manifest_sha256=manifest["manifest_sha256"],
                    renderer_subset_sha256="b" * 64,
                    identity_artifacts=[
                        {"path": str(results_path), "sha256": file_sha256(results_path)},
                        {"path": str(sidecar_path), "sha256": file_sha256(sidecar_path)},
                    ],
                    phase1_run_root=str(run_root),
                )
                identity_manifest = make_identity_manifest(
                    card,
                    identity_namespace=FULL_IDENTITY_NAMESPACE,
                    split="mmlu_test_full",
                    identities=self.identities,
                    source_provenance=source,
                )
                loaded = p2r.load_phase1_immutable_reuse_source(
                    card, identity_manifest, _reuse_cells()[0], manifest_path
                )
                self.assertEqual(
                    loaded["evidence"]["sample_source"],
                    {
                        "kind": "exact_sorted_logged_sample_sidecars",
                        "artifacts": [
                            {"path": str(sidecar_path.resolve()), "sha256": file_sha256(sidecar_path)}
                        ],
                    },
                )
                self.assertEqual(len(loaded["samples"]), 14042)

    def test_phase1_loader_rejects_fake_digest_tamper_missing_and_wrong_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "phase1-run"
            card = _card_for_roots(self.card, phase1_root=run_root)
            with mock.patch.object(p2s, "PHASE1_RUN_ROOT", str(run_root)), mock.patch.object(
                p2r, "PHASE1_RUN_ROOT", str(run_root)
            ):
                _, manifest_path, identity_manifest = _build_phase1_fixture(
                    Path(tmp), card, self.identities, self.raw_rows
                )
                cell = _reuse_cells()[1]
                loaded = p2r.load_phase1_immutable_reuse_source(
                    card, identity_manifest, cell, manifest_path
                )
                fake = copy.deepcopy(loaded["evidence"])
                fake["source_results"]["sha256"] = "f" * 64
                with self.assertRaisesRegex(p2r.Phase2ReuseError, "frozen reuse request"):
                    p2r.load_phase1_immutable_reuse_source(
                        card,
                        identity_manifest,
                        cell,
                        manifest_path,
                        expected_source_evidence=fake,
                    )

                results_path = Path(loaded["evidence"]["source_results"]["path"])
                tampered = json.loads(results_path.read_text(encoding="utf-8"))
                tampered["samples"]["mmlu_subject"][0]["filtered_resps"][0] = -9.0
                results_path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(p2r.Phase2ReuseError, "frozen reuse request"):
                    p2r.load_phase1_immutable_reuse_source(
                        card,
                        identity_manifest,
                        cell,
                        manifest_path,
                        expected_source_evidence=loaded["evidence"],
                    )
                results_path.unlink()
                with self.assertRaisesRegex(p2r.Phase2ReuseError, "missing or unreadable"):
                    p2r.load_phase1_immutable_reuse_source(
                        card, identity_manifest, cell, manifest_path
                    )

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "phase1-run"
            card = _card_for_roots(self.card, phase1_root=run_root)
            with mock.patch.object(p2s, "PHASE1_RUN_ROOT", str(run_root)), mock.patch.object(
                p2r, "PHASE1_RUN_ROOT", str(run_root)
            ):
                _, manifest_path, identity_manifest = _build_phase1_fixture(
                    Path(tmp), card, self.identities, self.raw_rows
                )
                malformed = json.loads(manifest_path.read_text(encoding="utf-8"))
                malformed["stages"]["gate-e-full"]["jobs"][1]["output_dir"] = str(
                    run_root / "gate-e-full" / "wrong-output"
                )
                malformed.pop("manifest_sha256")
                attach_manifest_sha256(malformed)
                manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
                identity = copy.deepcopy(identity_manifest)
                identity["source_provenance"]["source_manifest_sha256"] = malformed["manifest_sha256"]
                identity["source_provenance"].pop("manifest_sha256")
                attach_manifest_sha256(identity["source_provenance"])
                identity.pop("manifest_sha256")
                attach_manifest_sha256(identity)
                with self.assertRaisesRegex(p2r.Phase2ReuseError, "cell-to-output mapping"):
                    p2r.load_phase1_immutable_reuse_source(
                        card, identity, _reuse_cells()[1], manifest_path
                    )

    def test_new_full_positive_fixture_and_source_aware_analysis_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            phase1_root = root / "phase1-run"
            card = _card_for_roots(self.card, phase1_root=phase1_root, workspace=workspace)
            with mock.patch.object(p2s, "PHASE2_WORKSPACE_ROOT", str(workspace)), mock.patch.object(
                p2s, "PHASE1_RUN_ROOT", str(phase1_root)
            ), mock.patch.object(p2r, "PHASE1_RUN_ROOT", str(phase1_root)):
                _, _, identity_manifest = _build_phase1_fixture(
                    root, card, self.identities, self.raw_rows
                )
                fixture = _build_new_full_fixture(
                    root,
                    workspace,
                    card,
                    identity_manifest,
                    _new_cells()[0],
                    self.raw_rows,
                    self.samples,
                )
                verified = p2r.verify_full_final_output_artifact(
                    fixture["sidecar_path"], card, identity_manifest
                )
                self.assertEqual(verified, fixture["envelope"])

                results_path = fixture["output"] / "results.json"
                tampered = json.loads(results_path.read_text(encoding="utf-8"))
                tampered["samples"]["mmlu_subject"][0]["filtered_resps"][0] = -8.0
                results_path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(p2r.Phase2ReuseError, "file hash differs"):
                    p2r.verify_full_final_output_artifact(
                        fixture["sidecar_path"], card, identity_manifest
                    )
                ref = {
                    "cell_id": _new_cells()[0]["cell_id"],
                    "path": str(fixture["sidecar_path"]),
                    "sha256": fixture["envelope"]["manifest_sha256"],
                }
                with self.assertRaisesRegex(SchemaError, "producer/source verification failed"):
                    p2a._load_verified_full_cell(ref, root, card, identity_manifest)
                results_path.unlink()
                with self.assertRaisesRegex(p2r.Phase2ReuseError, "missing or unreadable"):
                    p2r.verify_full_final_output_artifact(
                        fixture["sidecar_path"], card, identity_manifest
                    )

    def test_full_producer_rejects_outside_prefix_and_symlink_escape(self):
        attacks = ("outside", "prefix", "dotdot", "symlink")
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                workspace = root / "workspace"
                phase1_root = root / "phase1-run"
                card = _card_for_roots(self.card, phase1_root=phase1_root, workspace=workspace)
                with mock.patch.object(p2s, "PHASE2_WORKSPACE_ROOT", str(workspace)), mock.patch.object(
                    p2s, "PHASE1_RUN_ROOT", str(phase1_root)
                ), mock.patch.object(p2r, "PHASE1_RUN_ROOT", str(phase1_root)):
                    _, _, identity_manifest = _build_phase1_fixture(
                        root, card, self.identities, self.raw_rows
                    )
                    fixture = _build_new_full_fixture(
                        root,
                        workspace,
                        card,
                        identity_manifest,
                        _new_cells()[0],
                        self.raw_rows,
                        self.samples,
                    )
                    command_path = fixture["output"] / "command_args.json"
                    if attack == "symlink":
                        outside = root / "outside-command.json"
                        outside.write_bytes(command_path.read_bytes())
                        command_path.unlink()
                        command_path.symlink_to(outside)
                    elif attack == "dotdot":
                        forged = copy.deepcopy(fixture["envelope"])
                        forged["producer_evidence"]["command_args"]["path"] = str(
                            fixture["output"] / "nested" / ".." / "command_args.json"
                        )
                        forged.pop("manifest_sha256")
                        attach_manifest_sha256(forged)
                        fixture["sidecar_path"].write_text(json.dumps(forged), encoding="utf-8")
                    else:
                        outside_dir = (
                            root / "outside"
                            if attack == "outside"
                            else fixture["output"].with_name(fixture["output"].name + "-prefix")
                        )
                        outside_dir.mkdir(parents=True)
                        outside = outside_dir / "command_args.json"
                        outside.write_bytes(command_path.read_bytes())
                        forged = copy.deepcopy(fixture["envelope"])
                        forged["producer_evidence"]["command_args"]["path"] = str(outside.resolve())
                        forged.pop("manifest_sha256")
                        attach_manifest_sha256(forged)
                        fixture["sidecar_path"].write_text(json.dumps(forged), encoding="utf-8")
                    with self.assertRaises((p2r.Phase2ReuseError, SchemaError)):
                        p2r.verify_full_final_output_artifact(
                            fixture["sidecar_path"], card, identity_manifest
                        )

    def test_reuse_full_sidecar_reloads_frozen_request_and_actual_phase1_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            phase1_root = root / "phase1-run"
            card = _card_for_roots(self.card, phase1_root=phase1_root, workspace=workspace)
            with mock.patch.object(p2s, "PHASE2_WORKSPACE_ROOT", str(workspace)), mock.patch.object(
                p2s, "PHASE1_RUN_ROOT", str(phase1_root)
            ), mock.patch.object(p2r, "PHASE1_RUN_ROOT", str(phase1_root)):
                _, manifest_path, identity_manifest = _build_phase1_fixture(
                    root, card, self.identities, self.raw_rows
                )
                cell = _reuse_cells()[0]
                source = p2r.load_phase1_immutable_reuse_source(
                    card, identity_manifest, cell, manifest_path
                )
                authorization_root = workspace / "staging" / "reuse-baseline"
                provenance = authorization_root / "provenance"
                output = authorization_root / "output"
                provenance.mkdir(parents=True)
                output.mkdir()
                common = {
                    "producer_kind": "phase1_immutable_reuse_adapter",
                    "card_manifest_sha256": card["manifest_sha256"],
                    "cell_id": cell["cell_id"],
                    "revision": card["science"]["revision"],
                    "executor_thread_id": "executor-fixture",
                    "authorization_root": str(authorization_root.resolve()),
                    "output_root": str(output.resolve()),
                }
                attempt = make_hashed_manifest(
                    {
                        "schema_version": "loopscope.phase2-full-attempt.v2",
                        "artifact_kind": "full_final_output_attempt",
                        **common,
                        "created_at_utc": "2026-07-14T04:00:00Z",
                    }
                )
                attempt_path = provenance / "attempt.json"
                atomic_write_new_json(attempt_path, attempt)
                receipt = make_hashed_manifest(
                    {
                        "schema_version": "loopscope.phase2-full-receipt.v2",
                        "artifact_kind": "full_final_output_execution_receipt",
                        **common,
                        "attempt_manifest_sha256": attempt["manifest_sha256"],
                        "created_at_utc": "2026-07-14T04:01:00Z",
                    }
                )
                receipt_path = provenance / "receipt.json"
                atomic_write_new_json(receipt_path, receipt)
                attempt_ref = {"path": str(attempt_path.resolve()), "sha256": attempt["manifest_sha256"]}
                receipt_ref = {"path": str(receipt_path.resolve()), "sha256": receipt["manifest_sha256"]}
                request = make_hashed_manifest(
                    {
                        "schema_version": "loopscope.phase2-phase1-reuse-request.v1",
                        "artifact_kind": "phase1_immutable_reuse_request",
                        "card_manifest_sha256": card["manifest_sha256"],
                        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
                        "cell": cell,
                        "source_evidence": source["evidence"],
                        "attempt_manifest": attempt_ref,
                        "receipt_manifest": receipt_ref,
                        "output_root": str(output.resolve()),
                    }
                )
                request_path = provenance / "reuse-request.json"
                atomic_write_new_json(request_path, request)
                request_ref = {"path": str(request_path.resolve()), "sha256": request["manifest_sha256"]}
                command = {
                    "card_manifest_sha256": card["manifest_sha256"],
                    "identity_manifest_sha256": identity_manifest["manifest_sha256"],
                    "cell_id": cell["cell_id"],
                    "phase1_run_manifest": source["evidence"]["source_run_manifest"]["path"],
                    "output_dir": str(output.resolve()),
                    "attempt_manifest": attempt_ref["path"],
                    "receipt_manifest": receipt_ref["path"],
                    "reuse_request": request_ref["path"],
                }
                atomic_write_new_json(output / "command_args.json", command)
                atomic_write_new_json(output / "env.json", {"VIRTUAL_ENV": "/fixture"})
                command_ref = {"path": str((output / "command_args.json").resolve()), "sha256": file_sha256(output / "command_args.json")}
                environment_ref = {"path": str((output / "env.json").resolve()), "sha256": file_sha256(output / "env.json")}
                envelope = p2r.build_phase1_reuse_full_envelope(
                    card,
                    identity_manifest,
                    cell,
                    run_manifest_path=manifest_path,
                    artifact_root=output,
                    reuse_request=request_ref,
                    attempt_manifest=attempt_ref,
                    receipt_manifest=receipt_ref,
                    command_args=command_ref,
                    environment=environment_ref,
                )
                sidecar = output / "phase2_final_outputs.json"
                atomic_write_new_json(sidecar, envelope)
                self.assertEqual(
                    p2r.verify_full_final_output_artifact(sidecar, card, identity_manifest),
                    envelope,
                )


if __name__ == "__main__":
    unittest.main()
