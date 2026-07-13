import copy
import hashlib
import inspect
import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tflt import eval_runner
from tflt.loopscope import phase2_trajectory as p2t
from tflt.loopscope import phase2_analysis as p2a
from tflt.loopscope.phase2_analysis import (
    Phase2AnalysisError,
    classify_nca,
    exact_mcnemar_p,
    classify_h1,
    choice_output,
    holm_step_down,
    make_analysis_input_manifest,
    required_calibration_cell_ids,
    required_full_cell_ids,
    validate_analysis_input_manifest,
    validate_analysis_report,
)
from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    DIRECT_PROBE_SCORE_SOURCE,
    FULL_LM_EVAL_SCORE_SOURCE,
    FULL_IDENTITY_NAMESPACE,
    FIXED_HORIZON_CONTRAST_IDS,
    PHASE2_ANALYSIS_SCHEMA_VERSION,
    PRIMARY_CONTRAST_IDS,
    SchemaError,
    atomic_write_new_json,
    b2_logical_cells,
    make_calibration_label_sidecar,
    make_identity_manifest,
    make_source_provenance,
    make_hashed_manifest,
    protocol_cell_id,
    make_unseal_authorization,
    make_unseal_receipt,
    validate_calibration_label_sidecar,
    validate_unseal_authorization,
    validate_unseal_receipt,
)
from tflt.loopscope.phase2_trajectory import (
    TrajectoryError,
    build_b1_admission_proof,
    run_phase2_trajectory_probe,
    validate_b1_admission_proof,
    validate_probe_aggregate,
)
from tflt.loopscope.schema import attach_manifest_sha256
from tflt.loopscope.schema import canonical_json_bytes, manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json"


def identity(index):
    return {"task": "mmlu_subject", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}


def source_ref(name, digest="a" * 64):
    return {"path": "%s.json" % name, "sha256": digest}


def cell_ref(cell_id, index):
    return {"cell_id": cell_id, "path": "%s.json" % cell_id, "sha256": "%064x" % (index + 1)}


def analysis_sources(card):
    calibration = [cell_ref(cell_id, index) for index, cell_id in enumerate(required_calibration_cell_ids(card))]
    full = [cell_ref(cell_id, index + 100) for index, cell_id in enumerate(required_full_cell_ids(card))]
    return {
        "card": source_ref("card", card["manifest_sha256"]),
        "calibration_identity_manifest": source_ref("calibration_identity"),
        "full_identity_manifest": source_ref("full_identity", "b" * 64),
        "calibration_baseline": source_ref("calibration_baseline", "c" * 64),
        "calibration_aggregate": source_ref("calibration_aggregate", "7" * 64),
        "calibration_cells": calibration,
        "full_final_output_cells": full,
        "calibration_labels": source_ref("calibration_labels", "d" * 64),
        "unseal_authorization": source_ref("unseal_authorization", "e" * 64),
        "unseal_receipt": source_ref("unseal_receipt", "f" * 64),
    }


def provenance():
    return {
        "authorization_manifest_sha256": "1" * 64,
        "attempt_manifest_sha256": "2" * 64,
        "receipt_manifest_sha256": "3" * 64,
        "executor_thread_id": "019f5c1d-7355-7130-81e2-125b4f1e21c0",
        "command_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
    }


def analysis_route_provenance():
    return {
        "authorization_manifest": {"path": "analysis_authorization.json", "sha256": "1" * 64},
        "attempt_manifest": {"path": "analysis_attempt.json", "sha256": "2" * 64},
        "receipt_manifest": {"path": "analysis_receipt.json", "sha256": "3" * 64},
        "executor_thread_id": "019f5c1d-7355-7130-81e2-125b4f1e21c0",
    }


def source_provenance(namespace):
    if namespace == CALIBRATION_IDENTITY_NAMESPACE:
        path = (
            "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/inputs/"
            "loopscope-qwen17-mmlu-phase1-20260711-043615/probe_pool_manifest.json"
        )
    else:
        path = (
            "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/"
            "loopscope-qwen17-mmlu-phase1-20260711-053022/control/phase1_run_manifest.json"
        )
    identity_artifacts = [] if namespace == CALIBRATION_IDENTITY_NAMESPACE else [{
        "path": (
            "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/"
            "loopscope-qwen17-mmlu-phase1-20260711-053022/gate-e-full/"
            "baseline-full/results.json"
        ),
        "sha256": "c" * 64,
    }]
    return make_source_provenance(
        identity_namespace=namespace,
        verification_status="live_verified",
        source_manifest_path=path,
        source_manifest_sha256="a" * 64,
        renderer_subset_sha256="b" * 64,
        identity_artifacts=identity_artifacts,
    )


def probe_pool(identities, *, source_count=512):
    records = [
        {"id": row["doc_id"], "prompt_sha256": row["doc_hash"]}
        for row in identities
    ]
    renderer_base = {
        "renderer_entrypoint": "lm_eval.api.task.ConfigurableTask.fewshot_context",
        "lm_eval_version": "0.4.11",
        "renderer_source_sha256": "1" * 64,
        "source_files_sha256": "1" * 64,
        "template_sha256": "2" * 64,
        "task_configs_sha256": "2" * 64,
        "render_contract_sha256": "3" * 64,
        "dataset_revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "dataset_fingerprint_sha256": "4" * 64,
        "source_projection_sha256": "5" * 64,
        "render_sha256": identities[0]["doc_hash"],
        "renderer_manifest_sha256": "7" * 64,
    }
    rendering = []
    for row in identities:
        renderer = dict(renderer_base)
        renderer["render_sha256"] = row["doc_hash"]
        demos = [
            {
                "id": "subject-demo-%d" % index,
                "doc_index": index,
                "source": "cais/mmlu",
                "split": "dev",
                "subject": "subject",
                "doc_sha256": "%064x" % (100 + index),
                "rendered_sha256": "%064x" % (200 + index),
                "gold_sha256": "%064x" % (300 + index),
            }
            for index in range(5)
        ]
        rendering.append(
            {
                "id": row["doc_id"],
                "target": {
                    "source": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
                    "split": "validation",
                    "subject": "subject",
                },
                "task_group": "mmlu",
                "task_name": row["task"],
                "target_doc_id": row["doc_id"],
                "target_doc_index": int(row["doc_id"]),
                "target_doc_sha256": row["doc_hash"],
                "dataset_fingerprint": "a" * 64,
                "num_fewshot": 5,
                "uses_target_gold_labels": False,
                "fewshot_answers_present": True,
                "renderer": renderer,
                "fewshot_sample_ids": [demo["id"] for demo in demos],
                "demonstrations": demos,
                "prompt_sha256": row["doc_hash"],
            }
        )
    render_hash = hashlib.sha256(canonical_json_bytes(rendering)).hexdigest()
    renderer = dict(renderer_base)
    selected = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
        "split": "validation",
        "count": len(identities),
        "seed": 20260710,
        "sample_ids": [row["doc_id"] for row in identities],
        "records": records,
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": renderer,
        "render_contract_sha256": "3" * 64,
        "rendering_records": rendering,
        "render_contract_subset_sha256": render_hash,
        "source_manifest_sha256": "8" * 64,
    }
    selected_hash = manifest_sha256(selected)
    return {
        "source": selected["source"],
        "split": "validation",
        "count": len(identities),
        "seed": 20260710,
        "manifest_sha256": selected_hash,
        "source_manifest_sha256": "8" * 64,
        "selected_subset_sha256": selected_hash,
        "source_manifest_count": source_count,
        "sample_ids": selected["sample_ids"],
        "records": records,
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": renderer,
        "render_contract_sha256": "3" * 64,
        "rendering_records": rendering,
        "source_render_contract_subset_sha256": "9" * 64,
        "selected_render_contract_subset_sha256": render_hash,
    }


def producer(kind="gate_b_remote_b1_smoke_probe"):
    return {
        "producer_kind": kind,
        "attempt_manifest_sha256": "1" * 64,
        "receipt_manifest_sha256": "2" * 64,
        "command_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
        "revision_report_sha256": "5" * 64,
    }


def producer_evidence(value, suffix="fixture"):
    root = (
        "/hpc2hdd/home/xhuang225/workspaces/"
        "training_free_looped_transformers_loopscope/staging/%s" % suffix
    )
    return {
        "attempt_manifest": {"path": root + "/attempt.json", "sha256": value["attempt_manifest_sha256"]},
        "receipt_manifest": {"path": root + "/receipt.json", "sha256": value["receipt_manifest_sha256"]},
        "input_manifest": {"path": root + "/probe_pool_manifest.json", "sha256": "8" * 64},
        "command_args": {"path": root + "/command_args.json", "sha256": value["command_sha256"]},
        "environment": {"path": root + "/env.json", "sha256": value["environment_sha256"]},
        "revision_report": {"path": root + "/revision_evidence.json", "sha256": value["revision_report_sha256"]},
    }


def resource_usage(card, *, include_k1):
    metric = lambda cell_id: {
        "cell_id": cell_id,
        "wall_clock_seconds": 1.0,
        "peak_gpu_memory_bytes": 1024,
    }
    return {
        "schema_version": "loopscope.phase2-probe-resource-usage.v1",
        "baseline": metric(protocol_cell_id("baseline_no_loop", "none", 1, 1.0)),
        "loop_cells": [metric(cell["cell_id"]) for cell in b2_logical_cells(card)],
        "k1_cells": (
            [metric(protocol_cell_id("fixed_horizon", window, 1, 1.0)) for window in card["science"]["windows"]]
            if include_k1 else []
        ),
    }


def restore_proof(card, *, include_k1):
    row = lambda cell_id: {
        "cell_id": cell_id,
        "restore_called": True,
        "restore_completed": True,
    }
    return {
        "schema_version": "loopscope.phase2-wrapper-restore-proof.v1",
        "loop_cells": [row(cell["cell_id"]) for cell in b2_logical_cells(card)],
        "k1_cells": (
            [row(protocol_cell_id("fixed_horizon", window, 1, 1.0)) for window in card["science"]["windows"]]
            if include_k1 else []
        ),
    }


def choice_row(sample_identity, scores, gold=0):
    top = max(range(4), key=lambda index: scores[index])
    return {
        "doc_id": sample_identity["doc_id"],
        "doc_hash": sample_identity["doc_hash"],
        "filtered_resps": list(scores),
        "target": gold,
        "metrics": {"acc,none": int(top == gold)},
    }


def baseline_probe_row(sample_identity):
    boundaries = []
    for window in ("11:14", "12:15", "13:16"):
        start, end = (int(value) for value in window.split(":"))
        boundaries.append(
            {
                "window": window,
                "inclusive_boundary": "B_%d_to_B_%d" % (start, end + 1),
                "native_boundary_formula": "B_N-B_%d" % (end + 1),
                "baseline_update_norm": 1.0,
                "native_continuation_norm": 1.0,
                "baseline_nca": {
                    "value": 1.0,
                    "residual_norm": 1.0,
                    "native_continuation_norm": 1.0,
                    "valid": True,
                    "invalid_reason": None,
                    "dtype": "float32",
                    "provenance": "baseline_single_forward_B_(b+1)-B_a",
                    "body_call_t": None,
                },
            }
        )
    return {
        "sample_identity": sample_identity,
        "answer_position": 7,
        "final_output": choice_output(
            [4.0, 3.0, 2.0, 1.0],
            identity=sample_identity,
            score_source=DIRECT_PROBE_SCORE_SOURCE,
        ),
        "window_boundaries": boundaries,
    }


def trajectory_probe_row(sample_identity, cell):
    steps = []
    for body_call_t in range(cell["k"]):
        steps.append(
            {
                "body_call_t": body_call_t,
                "residual_norm": 1.0,
                "state_norm": 2.0,
                "relative_activity": 0.5,
                "residual_ratio_to_previous": None if body_call_t == 0 else 1.0,
                "adjacent_residual_cosine": None if body_call_t == 0 else 1.0,
                "adjacent_residual_cosine_valid": None if body_call_t == 0 else True,
                "nca": None if body_call_t == 0 else {
                    "value": 1.0,
                    "residual_norm": 1.0,
                    "native_continuation_norm": 1.0,
                    "valid": True,
                    "invalid_reason": None,
                    "dtype": "float32",
                    "provenance": "actual_loop_repeated_step",
                    "body_call_t": body_call_t,
                },
                "valid": True,
            }
        )
    return {
        "sample_identity": sample_identity,
        "answer_position": 7,
        "position_rule": "final_pre_answer_prompt_token",
        "boundary_identity": "native_continuation=B_N-B_(b+1)",
        "body_call_indexing": "zero_based_t=0..K-1",
        "steps": steps,
        "repeated_step_nca_all_valid": True,
        "valid": True,
        "errors": [],
        "event_counts": {
            "wrapper_forward": 1,
            "bypass_false": 1,
            "body_call": cell["k"],
            "operator_body_calls": cell["k"],
            "g_minus_x": cell["k"],
            "looped_hidden_vs_input": 1,
            "stash_pass": 1,
            "identity_forward": 3,
        },
        "final_output": choice_output(
            [4.0, 3.0, 2.0, 1.0],
            identity=sample_identity,
            score_source=DIRECT_PROBE_SCORE_SOURCE,
        ),
    }


class Phase2PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    def _rehash(self, payload):
        payload.pop("manifest_sha256", None)
        attach_manifest_sha256(payload)
        return payload

    def test_large_discordance_exact_mcnemar_is_stable(self):
        for left, right in ((0, 1024), (500, 600), (7000, 7042), (7042, 7000)):
            value = exact_mcnemar_p(left, right)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(exact_mcnemar_p(7000, 7042), exact_mcnemar_p(7042, 7000))

    def test_analysis_input_is_source_only_and_closes_cell_sets_and_statistics(self):
        payload = make_analysis_input_manifest(
            self.card, analysis_sources(self.card), analysis_route_provenance()
        )
        validate_analysis_input_manifest(payload, self.card)

        forged = copy.deepcopy(payload)
        forged["primary_contrasts"] = {"forged": {"mcnemar_raw_p": 0.0}}
        missing = copy.deepcopy(payload)
        missing["sources"]["full_final_output_cells"].pop()
        rogue = copy.deepcopy(payload)
        rogue["sources"]["calibration_cells"].append(cell_ref("rogue_cell", 999))
        wrong_seed = copy.deepcopy(payload)
        wrong_seed["statistics_contract"]["nca_bootstrap"]["seed"] = 1
        wrong_replicates = copy.deepcopy(payload)
        wrong_replicates["statistics_contract"]["full_paired_bootstrap"]["replicates"] = 1999
        for candidate in (forged, missing, rogue, wrong_seed, wrong_replicates):
            with self.assertRaises(SchemaError):
                validate_analysis_input_manifest(self._rehash(candidate), self.card)

    def test_analysis_route_reloads_real_control_packets_and_rejects_forged_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = analysis_sources(self.card)
            sources_sha = hashlib.sha256(canonical_json_bytes(sources)).hexdigest()
            common = {
                "card_manifest_sha256": self.card["manifest_sha256"],
                "analysis_sources_sha256": sources_sha,
                "output_root": str((root / "analysis").resolve()),
                "executor_thread_id": "executor-test",
            }
            authorization = make_hashed_manifest({
                "schema_version": "loopscope.phase2-analysis-authorization.v1",
                "artifact_kind": "phase2_h1_analysis_authorization",
                **common,
                "created_at_utc": "2026-07-14T03:00:00Z",
            })
            attempt = make_hashed_manifest({
                "schema_version": "loopscope.phase2-analysis-attempt.v1",
                "artifact_kind": "phase2_h1_analysis_attempt",
                **common,
                "authorization_manifest_sha256": authorization["manifest_sha256"],
                "created_at_utc": "2026-07-14T03:01:00Z",
            })
            receipt = make_hashed_manifest({
                "schema_version": "loopscope.phase2-analysis-receipt.v1",
                "artifact_kind": "phase2_h1_analysis_execution_receipt",
                **common,
                "authorization_manifest_sha256": authorization["manifest_sha256"],
                "attempt_manifest_sha256": attempt["manifest_sha256"],
                "created_at_utc": "2026-07-14T03:02:00Z",
            })
            for name, packet in (
                ("authorization.json", authorization),
                ("attempt.json", attempt),
                ("receipt.json", receipt),
            ):
                atomic_write_new_json(root / name, packet)
            route = {
                "authorization_manifest": {"path": "authorization.json", "sha256": authorization["manifest_sha256"]},
                "attempt_manifest": {"path": "attempt.json", "sha256": attempt["manifest_sha256"]},
                "receipt_manifest": {"path": "receipt.json", "sha256": receipt["manifest_sha256"]},
                "executor_thread_id": "executor-test",
            }
            request = make_analysis_input_manifest(self.card, sources, route)
            with mock.patch.object(
                p2a,
                "validate_phase2_workspace_output_path",
                side_effect=lambda value, card, context: Path(value).resolve(),
            ):
                loaded = p2a._load_analysis_control_manifests(request, root, self.card)
            self.assertEqual(loaded["attempt"], attempt)
            output_root = Path(common["output_root"])
            output_root.mkdir()
            request_path = root / "analysis_input.json"
            atomic_write_new_json(request_path, request)
            atomic_write_new_json(
                output_root / "command_args.json",
                p2a.analysis_command_record(request_path, output_root),
            )
            atomic_write_new_json(
                output_root / "env.json", p2a.analysis_environment_record()
            )
            evidence = {
                "analysis_control": loaded,
                "card": self.card,
                "request_path": request_path.resolve(),
            }
            with mock.patch.object(
                p2a,
                "validate_phase2_workspace_output_path",
                side_effect=lambda value, card, context: Path(value).resolve(),
            ):
                actual = p2a.derive_analysis_execution_provenance(
                    evidence, output_root
                )
            self.assertEqual(
                actual["attempt_manifest_sha256"], attempt["manifest_sha256"]
            )
            forged_command = p2a.analysis_command_record(request_path, output_root)
            forged_command["input"] = str((root / "other-input.json").resolve())
            (output_root / "command_args.json").unlink()
            atomic_write_new_json(output_root / "command_args.json", forged_command)
            with mock.patch.object(
                p2a,
                "validate_phase2_workspace_output_path",
                side_effect=lambda value, card, context: Path(value).resolve(),
            ), self.assertRaises(SchemaError):
                p2a.derive_analysis_execution_provenance(evidence, output_root)

            forged = copy.deepcopy(attempt)
            forged["analysis_sources_sha256"] = "f" * 64
            self._rehash(forged)
            atomic_write_new_json(root / "forged_attempt.json", forged)
            forged_request = copy.deepcopy(request)
            forged_request["analysis_provenance"]["attempt_manifest"] = {
                "path": "forged_attempt.json",
                "sha256": forged["manifest_sha256"],
            }
            self._rehash(forged_request)
            with mock.patch.object(
                p2a,
                "validate_phase2_workspace_output_path",
                side_effect=lambda value, card, context: Path(value).resolve(),
            ), self.assertRaises(SchemaError):
                p2a._load_analysis_control_manifests(
                    forged_request, root, self.card
                )

    def test_unseal_requires_card_sealed_hashes_authorization_and_receipt(self):
        identities = [identity(index) for index in range(512)]
        manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=identities,
            source_provenance=source_provenance(CALIBRATION_IDENTITY_NAMESPACE),
        )
        sealed_hashes = ["%064x" % (1000 + index) for index in range(16)]
        authorization = make_unseal_authorization(
            self.card,
            manifest,
            sealed_artifact_sha256=sealed_hashes,
            authorization_id="phase2-h1-v2-calibration-unseal-1",
            authorized_by_thread="019f5149-35f6-7383-b3ec-3376ba413d2b",
            issued_at_utc="2026-07-14T00:00:00Z",
        )
        validate_unseal_authorization(authorization, self.card, manifest, sealed_hashes)
        labels = make_calibration_label_sidecar(
            self.card,
            manifest,
            authorization,
            gold_indices=[index % 4 for index in range(512)],
            producer={
                "producer_kind": "authorized_calibration_label_unseal",
                "attempt_manifest_sha256": "1" * 64,
                "receipt_manifest_sha256": "2" * 64,
                "command_sha256": "3" * 64,
                "environment_sha256": "4" * 64,
                "revision_report_sha256": "5" * 64,
            },
        )
        validate_calibration_label_sidecar(labels, self.card, manifest, authorization)
        receipt = make_unseal_receipt(
            self.card,
            manifest,
            authorization,
            labels,
            sealed_artifact_sha256=sealed_hashes,
            accessed_by_thread="019f5132-cd06-7281-be3b-a01a06f7a757",
            accessed_at_utc="2026-07-14T00:01:00Z",
        )
        validate_unseal_receipt(receipt, self.card, manifest, authorization, labels, sealed_hashes)

        changed = list(sealed_hashes)
        changed[-1] = "9" * 64
        with self.assertRaises(SchemaError):
            validate_unseal_authorization(authorization, self.card, manifest, changed)
        with self.assertRaises(SchemaError):
            make_unseal_authorization(
                self.card,
                manifest,
                sealed_artifact_sha256=sealed_hashes,
                authorization_id="invalid-calendar",
                authorized_by_thread="planning-test",
                issued_at_utc="2026-99-14T00:00:00Z",
            )
        forged_receipt = copy.deepcopy(receipt)
        forged_receipt["label_sidecar_sha256"] = "8" * 64
        with self.assertRaises(SchemaError):
            validate_unseal_receipt(
                self._rehash(forged_receipt), self.card, manifest, authorization, labels, sealed_hashes
            )

    def test_b1_producer_path_emits_k1_and_prefix_proof_without_vectors(self):
        identities = [identity(index) for index in range(4)]
        baseline = [
            choice_output(
                [4.0, 3.0, 2.0, 1.0],
                identity=item,
                score_source=DIRECT_PROBE_SCORE_SOURCE,
            )
            for item in identities
        ]
        k1 = {window: copy.deepcopy(baseline) for window in ("11:14", "12:15", "13:16")}
        captures = {}
        for window in ("11:14", "12:15", "13:16"):
            for protocol, k, alpha in (
                ("shared_k2_anchor", 2, 1.0),
                ("fixed_step", 3, 1.5),
                ("fixed_step", 4, 2.0),
            ):
                cell_id = "%s_%s_k%d_a%s" % (
                    protocol, window.replace(":", "_"), k, str(alpha).replace(".", "p").replace("p0", "")
                )
                captures[cell_id] = [
                    {
                        "sample_identity": item,
                        "states": [[1.0, 2.0], [2.0, 3.0]][:k],
                        "residuals": [[0.1, 0.2], [0.2, 0.3]][:k],
                    }
                    for item in identities
                ]
        b1_producer = producer()
        proof = build_b1_admission_proof(
            self.card,
            baseline,
            k1,
            captures,
            probe_pool=probe_pool(identities),
            producer=b1_producer,
            producer_evidence=producer_evidence(b1_producer, "proof-unit"),
        )
        validate_b1_admission_proof(proof, self.card)
        self.assertTrue(proof["k1_equivalence"]["all_match"])
        self.assertTrue(proof["fixed_step_prefix_consistency"]["all_match"])
        serialized = json.dumps(proof, sort_keys=True)
        self.assertNotIn('"states"', serialized)
        self.assertNotIn('"residuals"', serialized)
        self.assertIn("_build_b1_admission_core(", inspect.getsource(run_phase2_trajectory_probe))

    def test_probe_producer_reloads_exact_attempt_receipt_and_actual_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_manifest = make_hashed_manifest({"schema_version": "fixture.pool.v1"})
            atomic_write_new_json(root / "probe_pool_manifest.json", input_manifest)
            input_ref = {
                "path": str((root / "probe_pool_manifest.json").resolve()),
                "sha256": input_manifest["manifest_sha256"],
            }
            common = {
                "card_manifest_sha256": self.card["manifest_sha256"],
                "probe_mode": "b1-smoke",
                "evidence_scale": "calibration_smoke_4",
                "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
                "revision": self.card["science"]["revision"],
                "input_manifest": input_ref,
                "output_root": str(root.resolve()),
                "executor_thread_id": "executor-test",
            }
            attempt = make_hashed_manifest({
                "schema_version": "loopscope.phase2-probe-attempt.v1",
                "artifact_kind": "phase2_b1_smoke_attempt",
                **common,
                "created_at_utc": "2026-07-14T02:00:00Z",
            })
            receipt = make_hashed_manifest({
                "schema_version": "loopscope.phase2-probe-receipt.v1",
                "artifact_kind": "phase2_b1_smoke_execution_receipt",
                **common,
                "attempt_manifest_sha256": attempt["manifest_sha256"],
                "created_at_utc": "2026-07-14T02:01:00Z",
            })
            atomic_write_new_json(root / "attempt.json", attempt)
            atomic_write_new_json(root / "receipt.json", receipt)
            command = {
                "probe_mode": "b1-smoke",
                "model": "qwen3-1.7b-base",
                "revision": self.card["science"]["revision"],
                "dtype": self.card["science"]["dtype"],
                "output_dir": str(root.resolve()),
                "input_manifest": input_ref["path"],
                "attempt_manifest": str((root / "attempt.json").resolve()),
                "receipt_manifest": str((root / "receipt.json").resolve()),
                "max_examples": 4,
                "b1_admission_proof": None,
            }
            atomic_write_new_json(root / "command_args.json", command)
            atomic_write_new_json(root / "env.json", {"VIRTUAL_ENV": "/fixture"})
            revision = make_hashed_manifest({
                "schema_version": "loopscope.phase2-probe-revision-evidence.v1",
                "card_manifest_sha256": self.card["manifest_sha256"],
                "manifest_revision": self.card["science"]["revision"],
                "revision_closure": {
                    "manifest_commit": self.card["science"]["revision"],
                    "model_commit": self.card["science"]["revision"],
                    "tokenizer_commit": self.card["science"]["revision"],
                    "match": True,
                },
            })
            atomic_write_new_json(root / "revision_evidence.json", revision)
            probe_producer = {
                "producer_kind": "gate_b_remote_b1_smoke_probe",
                "attempt_manifest_sha256": attempt["manifest_sha256"],
                "receipt_manifest_sha256": receipt["manifest_sha256"],
                "command_sha256": hashlib.sha256((root / "command_args.json").read_bytes()).hexdigest(),
                "environment_sha256": hashlib.sha256((root / "env.json").read_bytes()).hexdigest(),
                "revision_report_sha256": revision["manifest_sha256"],
            }
            evidence = {
                "attempt_manifest": {"path": str((root / "attempt.json").resolve()), "sha256": attempt["manifest_sha256"]},
                "receipt_manifest": {"path": str((root / "receipt.json").resolve()), "sha256": receipt["manifest_sha256"]},
                "input_manifest": input_ref,
                "command_args": {"path": str((root / "command_args.json").resolve()), "sha256": probe_producer["command_sha256"]},
                "environment": {"path": str((root / "env.json").resolve()), "sha256": probe_producer["environment_sha256"]},
                "revision_report": {"path": str((root / "revision_evidence.json").resolve()), "sha256": revision["manifest_sha256"]},
            }
            p2t._validate_probe_producer_evidence(
                evidence,
                probe_producer,
                card=self.card,
                probe_mode="b1-smoke",
                output_root=root,
                load_files=True,
            )
            forged = copy.deepcopy(receipt)
            forged["attempt_manifest_sha256"] = "f" * 64
            self._rehash(forged)
            atomic_write_new_json(root / "forged_receipt.json", forged)
            forged_producer = dict(probe_producer)
            forged_producer["receipt_manifest_sha256"] = forged["manifest_sha256"]
            forged_evidence = copy.deepcopy(evidence)
            forged_evidence["receipt_manifest"] = {
                "path": str((root / "forged_receipt.json").resolve()),
                "sha256": forged["manifest_sha256"],
            }
            with self.assertRaises(TrajectoryError):
                p2t._validate_probe_producer_evidence(
                    forged_evidence,
                    forged_producer,
                    card=self.card,
                    probe_mode="b1-smoke",
                    output_root=root,
                    load_files=True,
                )

    def test_b1_aggregate_closes_cell_order_and_proof_identity(self):
        identities = [identity(index) for index in range(4)]
        baseline_outputs = [baseline_probe_row(item)["final_output"] for item in identities]
        k1 = {window: copy.deepcopy(baseline_outputs) for window in ("11:14", "12:15", "13:16")}
        captures = {}
        for window in ("11:14", "12:15", "13:16"):
            for protocol, k, alpha in (
                ("shared_k2_anchor", 2, 1.0),
                ("fixed_step", 3, 1.5),
                ("fixed_step", 4, 2.0),
            ):
                cell_id = protocol_cell_id(protocol, window, k, alpha)
                captures[cell_id] = [
                    {
                        "sample_identity": item,
                        "states": [[1.0, 2.0], [2.0, 3.0]],
                        "residuals": [[0.1, 0.2], [0.2, 0.3]],
                    }
                    for item in identities
                ]
        b1_pool = probe_pool(identities)
        b1_producer = producer()
        b1_evidence = producer_evidence(b1_producer, "b1-aggregate")
        proof = build_b1_admission_proof(
            self.card,
            baseline_outputs,
            k1,
            captures,
            probe_pool=b1_pool,
            producer=b1_producer,
            producer_evidence=b1_evidence,
        )
        report = make_hashed_manifest(
            {
                "schema_version": "loopscope.phase2-b1-smoke-aggregate.v3",
                "artifact_kind": "b1_smoke_scalar_aggregate",
                "artifact_root": (
                    "/hpc2hdd/home/xhuang225/workspaces/"
                    "training_free_looped_transformers_loopscope/staging/b1-aggregate"
                ),
                "card_manifest_sha256": self.card["manifest_sha256"],
                "revision": self.card["science"]["revision"],
                "sample_count": 4,
                "probe_pool": b1_pool,
                "warnings": [],
                "session_contract": {
                    "model_load_count": 1,
                    "no_loop_boundary_passes_per_sample": 1,
                    "logical_loop_cell_count": 18,
                    "k1_admission_cell_count": 3,
                    "native_vectors_persisted": False,
                },
                "baseline_examples": [baseline_probe_row(item) for item in identities],
                "loop_cells": [
                    {
                        "cell": cell,
                        "sample_count": 4,
                        "trajectory_samples": [trajectory_probe_row(item, cell) for item in identities],
                    }
                    for cell in b2_logical_cells(self.card)
                ],
                "b1_admission_proof": {
                    "path": "b1_admission_proof.json",
                    "sha256": proof["manifest_sha256"],
                },
                "resource_usage": resource_usage(self.card, include_k1=True),
                "restore_proof": restore_proof(self.card, include_k1=True),
                "producer": b1_producer,
                "producer_evidence": b1_evidence,
                "vectors_persisted": False,
            }
        )
        validate_probe_aggregate(report, self.card, b1_proof=proof)

        reordered = copy.deepcopy(report)
        reordered["loop_cells"] = list(reversed(reordered["loop_cells"]))
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(self._rehash(reordered), self.card, b1_proof=proof)
        wrong_identity = copy.deepcopy(report)
        wrong_identity["loop_cells"][0]["trajectory_samples"][0]["sample_identity"] = identities[1]
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(self._rehash(wrong_identity), self.card, b1_proof=proof)

    def test_b2_aggregate_closes_pool_root_first_four_seal_resources_and_refs(self):
        full_identities = [identity(index) for index in range(512)]

        def proof_for(smoke_identities):
            baseline = [baseline_probe_row(item)["final_output"] for item in smoke_identities]
            k1 = {window: copy.deepcopy(baseline) for window in self.card["science"]["windows"]}
            captures = {}
            for window in self.card["science"]["windows"]:
                for protocol, k, alpha in (
                    ("shared_k2_anchor", 2, 1.0),
                    ("fixed_step", 3, 1.5),
                    ("fixed_step", 4, 2.0),
                ):
                    captures[protocol_cell_id(protocol, window, k, alpha)] = [
                        {
                            "sample_identity": item,
                            "states": [[1.0, 2.0], [2.0, 3.0]],
                            "residuals": [[0.1, 0.2], [0.2, 0.3]],
                        }
                        for item in smoke_identities
                    ]
            proof_producer = producer()
            return build_b1_admission_proof(
                self.card,
                baseline,
                k1,
                captures,
                probe_pool=probe_pool(smoke_identities),
                producer=proof_producer,
                producer_evidence=producer_evidence(proof_producer, "b1-for-b2-%s" % smoke_identities[0]["doc_id"]),
            )

        proof = proof_for(full_identities[:4])
        full_pool = probe_pool(full_identities)
        identity_manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=full_identities,
            source_provenance=source_provenance(CALIBRATION_IDENTITY_NAMESPACE),
        )
        b2_producer = producer("gate_b_remote_calibration_probe")
        b2_evidence = producer_evidence(b2_producer, "b2-aggregate")
        report = make_hashed_manifest(
            {
                "schema_version": "loopscope.phase2-calibration-aggregate.v3",
                "artifact_kind": "sealed_calibration_512_scalar_aggregate",
                "artifact_root": (
                    "/hpc2hdd/home/xhuang225/workspaces/"
                    "training_free_looped_transformers_loopscope/staging/b2-aggregate"
                ),
                "card_manifest_sha256": self.card["manifest_sha256"],
                "identity_manifest": {
                    "path": "calibration_identity_manifest.json",
                    "sha256": identity_manifest["manifest_sha256"],
                },
                "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
                "revision": self.card["science"]["revision"],
                "label_state": "sealed",
                "sample_count": 512,
                "probe_pool": full_pool,
                "warnings": [],
                "session_contract": {
                    "model_load_count": 1,
                    "no_loop_boundary_passes_per_sample": 1,
                    "logical_loop_cell_count": 15,
                    "k1_admission_cell_count": 0,
                    "native_vectors_persisted": False,
                },
                "baseline": {"path": "calibration_baseline.json", "sha256": "a" * 64},
                "cells": [
                    {
                        "cell_id": cell_id,
                        "path": "cells/%s.json" % cell_id,
                        "sha256": "%064x" % (1000 + index),
                    }
                    for index, cell_id in enumerate(required_calibration_cell_ids(self.card))
                ],
                "prior_b1_admission_proof": {
                    "path": (
                        "/hpc2hdd/home/xhuang225/workspaces/"
                        "training_free_looped_transformers_loopscope/staging/"
                        "b1-for-b2-0/b1_admission_proof.json"
                    ),
                    "sha256": proof["manifest_sha256"],
                },
                "resource_usage": resource_usage(self.card, include_k1=False),
                "restore_proof": restore_proof(self.card, include_k1=False),
                "producer": b2_producer,
                "producer_evidence": b2_evidence,
                "vectors_persisted": False,
            }
        )
        validate_probe_aggregate(
            report, self.card, b1_proof=proof, identity_manifest=identity_manifest
        )

        labelled = copy.deepcopy(report)
        labelled["probe_pool"]["gold_index"] = 0
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(
                self._rehash(labelled),
                self.card,
                b1_proof=proof,
                identity_manifest=identity_manifest,
            )
        labelled_target = copy.deepcopy(report)
        labelled_target["probe_pool"]["rendering_records"][0]["target"]["answer"] = "A"
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(
                self._rehash(labelled_target),
                self.card,
                b1_proof=proof,
                identity_manifest=identity_manifest,
            )
        unrelated = proof_for(full_identities[4:8])
        swapped = copy.deepcopy(report)
        swapped["prior_b1_admission_proof"]["sha256"] = unrelated["manifest_sha256"]
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(
                self._rehash(swapped),
                self.card,
                b1_proof=unrelated,
                identity_manifest=identity_manifest,
            )
        tampered_subset = copy.deepcopy(proof)
        tampered_subset["probe_pool_binding"]["selected_subset_sha256"] = "f" * 64
        self._rehash(tampered_subset)
        tampered_report = copy.deepcopy(report)
        tampered_report["prior_b1_admission_proof"]["sha256"] = tampered_subset["manifest_sha256"]
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(
                self._rehash(tampered_report),
                self.card,
                b1_proof=tampered_subset,
                identity_manifest=identity_manifest,
            )

    def test_logged_samples_join_is_order_independent_and_exact(self):
        first, second = identity(1), identity(2)
        result = {
            "samples": {
                "mmlu_subject": [
                    choice_row(second, [1.0, 4.0, 0.0, -1.0], gold=1),
                    choice_row(first, [4.0, 1.0, 0.0, -1.0], gold=0),
                ]
            }
        }
        rows = eval_runner._join_phase2_logged_samples(result, [first, second])
        self.assertEqual([row["sample_identity"] for row in rows], [first, second])
        self.assertTrue(all(row["choice_score_source"] == FULL_LM_EVAL_SCORE_SOURCE for row in rows))

        duplicate = copy.deepcopy(result)
        duplicate["samples"]["mmlu_subject"].append(choice_row(first, [4, 1, 0, -1], gold=0))
        with self.assertRaises(ValueError):
            eval_runner._join_phase2_logged_samples(duplicate, [first, second])
        missing_scores = copy.deepcopy(result)
        missing_scores["samples"]["mmlu_subject"][0].pop("filtered_resps")
        with self.assertRaises(ValueError):
            eval_runner._join_phase2_logged_samples(missing_scores, [first, second])

    def test_reversed_nca_interval_is_not_accepted_as_available(self):
        primary = {}
        for window in ("11:14", "12:15", "13:16"):
            for k in (2, 3, 4):
                primary["%s_k%d" % (window, k)] = {
                    "valid_fraction": 1.0,
                    "overall_interval": {"available": True, "interval": [0.2, 0.1]},
                }
        paired = {
            "k%d" % k: {
                "paired_valid_fraction": 1.0,
                "difference_interval": {"available": True, "interval": [0.1, 0.2]},
            }
            for k in (2, 3, 4)
        }
        with self.assertRaises(Phase2AnalysisError):
            classify_nca(primary, paired, {})

    def test_analysis_report_is_closed_world_and_recomputes_p_holm_and_labels(self):
        bootstrap = {"replicates": 2000, "seed": 20260710, "paired_index": True}

        def base_contrast():
            return {
                "sample_count": 14042,
                "delta_acc_pp": 0.0,
                "paired_ci_pp": [0.0, 0.0],
                "bootstrap": dict(bootstrap),
                "transitions": {
                    "right_to_right": 14042,
                    "wrong_to_right": 0,
                    "right_to_wrong": 0,
                    "wrong_to_wrong": 0,
                },
                "right_to_right": 14042,
                "wrong_to_right": 0,
                "right_to_wrong": 0,
                "wrong_to_wrong": 0,
                "mcnemar_raw_p": 1.0,
            }

        primary_specs = []
        for window, stem in (("11:14", "11_14"), ("12:15", "12_15"), ("13:16", "13_16")):
            k2 = protocol_cell_id("shared_k2_anchor", window, 2, 1.0)
            k3 = protocol_cell_id("fixed_step", window, 3, 1.5)
            k4 = protocol_cell_id("fixed_step", window, 4, 2.0)
            primary_specs.extend(
                (("fs_%s_k2_k3" % stem, k2, k3), ("fs_%s_k3_k4" % stem, k3, k4))
            )
        fixed_specs = [
            (
                "fh_12_15_k2_k3",
                protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0),
                protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
            ),
            (
                "fh_12_15_k3_k4",
                protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
                protocol_cell_id("fixed_horizon", "12:15", 4, 1.0),
            ),
        ]

        def family(specs, frozen_ids):
            raw = {identifier: 1.0 for identifier in frozen_ids}
            adjusted = holm_step_down(raw, frozen_ids)
            result = {}
            for identifier, left, right in specs:
                cell = base_contrast()
                cell.update(
                    {
                        "contrast_id": identifier,
                        "left_cell_id": left,
                        "right_cell_id": right,
                        "holm_adjusted_p": adjusted[identifier]["adjusted_p"],
                        "holm_significant": adjusted[identifier]["significant"],
                        "holm_rank": adjusted[identifier]["holm_rank"],
                    }
                )
                result[identifier] = cell
            return result

        primary = family(primary_specs, PRIMARY_CONTRAST_IDS)
        fixed = family(fixed_specs, FIXED_HORIZON_CONTRAST_IDS)
        mechanism = {}
        for identifier in list(PRIMARY_CONTRAST_IDS) + list(FIXED_HORIZON_CONTRAST_IDS):
            mechanism[identifier] = {
                "sample_count": 14042,
                "mean_js_nats": 0.0,
                "mean_entropy_delta_nats": 0.0,
                "mean_top_margin_delta_raw": 0.0,
                "mean_correct_margin_delta_raw": 0.0,
                "top1_retained_fraction": 1.0,
                "transitions": dict((primary.get(identifier) or fixed[identifier])["transitions"]),
                "wrong_overconfidence": {
                    "definition": "final_wrong_pairs_with_entropy_down_and_top_margin_up",
                    "denominator_scope": "wrong_to_wrong_plus_right_to_wrong",
                    "count": 0,
                    "denominator": 0,
                    "fraction": None,
                    "subgroups": {
                        "wrong_to_wrong": {"count": 0, "denominator": 0, "fraction": None},
                        "right_to_wrong": {"count": 0, "denominator": 0, "fraction": None},
                    },
                    "role": "mechanism_diagnostic_not_H1_decision_rule",
                },
            }

        median_interval = {
            "available": True,
            "median": 0.2,
            "interval": [0.1, 0.3],
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        paired_interval = {
            "available": True,
            "difference": 0.0,
            "interval": [-0.1, 0.1],
            "paired_index": True,
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        independent_interval = {
            "available": True,
            "difference": -0.1,
            "interval": [-0.2, 0.0],
            "stratified_within_frozen_subgroup": True,
            "left_count": 10,
            "right_count": 10,
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        nca_primary = {
            "%s_k%d" % (window, k): {
                "denominator": 512,
                "valid_count": 512,
                "valid_fraction": 1.0,
                "overall_interval": copy.deepcopy(median_interval),
            }
            for window in ("11:14", "12:15", "13:16")
            for k in (2, 3, 4)
        }
        nca_paired = {
            "k%d" % k: {
                "denominator": 512,
                "paired_valid_count": 512,
                "paired_valid_fraction": 1.0,
                "difference_interval": copy.deepcopy(paired_interval),
            }
            for k in (2, 3, 4)
        }
        nca_native = {
            "12:15_k%d" % k: {
                "valid_subgroup_counts": {
                    "right_to_right": 492,
                    "wrong_to_right": 10,
                    "right_to_wrong": 0,
                    "wrong_to_wrong": 10,
                },
                "right_to_right_interval": copy.deepcopy(median_interval),
                "wrong_to_right_minus_wrong_to_wrong_interval": copy.deepcopy(independent_interval),
                "resampling": "stratified_within_frozen_subgroup",
            }
            for k in (2, 3, 4)
        }
        nca_decision = classify_nca(nca_primary, nca_paired, nca_native)
        cumulative = {
            "phase1_k2_vs_baseline_12_15": base_contrast(),
            "fixed_step_k4_vs_k2_12_15": base_contrast(),
            "role": "secondary_sign_guard_and_context_only",
        }
        h1 = classify_h1(
            primary,
            d24_12_15_pp=0.0,
            d24_12_15_ci_pp=[0.0, 0.0],
            phase1_k2_vs_baseline_pp=0.0,
        )
        source_hashes = {
            "card": self.card["manifest_sha256"],
            "calibration_identity_manifest": "1" * 64,
            "full_identity_manifest": "2" * 64,
            "calibration_baseline": "3" * 64,
            "calibration_aggregate": "7" * 64,
            "calibration_cells": {
                cell_id: "%064x" % (200 + index)
                for index, cell_id in enumerate(required_calibration_cell_ids(self.card))
            },
            "full_final_output_cells": {
                cell_id: "%064x" % (400 + index)
                for index, cell_id in enumerate(required_full_cell_ids(self.card))
            },
            "calibration_labels": "4" * 64,
            "unseal_authorization": "5" * 64,
            "unseal_receipt": "6" * 64,
        }
        trajectory_summary = {}
        for cell in b2_logical_cells(self.card):
            steps = []
            for body_call_t in range(cell["k"]):
                steps.append(
                    {
                        "body_call_t": body_call_t,
                        "residual_norm_median": 1.0,
                        "state_norm_median": 2.0,
                        "relative_activity_median": 0.5,
                        "residual_ratio_to_previous_median": None if body_call_t == 0 else 1.0,
                        "adjacent_residual_cosine_median": None if body_call_t == 0 else 0.5,
                        "adjacent_residual_cosine_valid_count": 0 if body_call_t == 0 else 512,
                    }
                )
            trajectory_summary[cell["cell_id"]] = {
                "cell": dict(cell),
                "sample_count": 512,
                "valid_sample_count": 512,
                "valid_sample_fraction": 1.0,
                "steps": steps,
                "role": "calibration_512_mechanism_evidence_only",
            }
        report = make_hashed_manifest(
            {
                "schema_version": PHASE2_ANALYSIS_SCHEMA_VERSION,
                "card_id": self.card["card_id"],
                "card_manifest_sha256": self.card["manifest_sha256"],
                "analysis_input_manifest_sha256": "7" * 64,
                "analysis_input_file_sha256": "8" * 64,
                "source_artifacts": source_hashes,
                "identity_closure": {
                    "calibration": {
                        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
                        "identity_manifest_sha256": "9" * 64,
                        "ordered_identity_sha256": "a" * 64,
                        "sample_count": 512,
                        "natural_order": "canonical_manifest_list_order_zero_based",
                        "source_provenance": source_provenance(CALIBRATION_IDENTITY_NAMESPACE),
                    },
                    "full": {
                        "identity_namespace": "loopscope.phase2.full.mmlu14042.v1",
                        "identity_manifest_sha256": "b" * 64,
                        "ordered_identity_sha256": "c" * 64,
                        "sample_count": 14042,
                        "natural_order": "canonical_manifest_list_order_zero_based",
                        "source_provenance": source_provenance(FULL_IDENTITY_NAMESPACE),
                    },
                },
                "revision": self.card["science"]["revision"],
                "statistics_contract": copy.deepcopy(self.card["statistics"]),
                "primary_contrasts": primary,
                "fixed_horizon_contrasts": fixed,
                "cumulative_context": cumulative,
                "mechanism_diagnostics": mechanism,
                "calibration_trajectory_summary": trajectory_summary,
                "nca": {
                    "primary_cells": nca_primary,
                    "paired_12_15_minus_13_16": nca_paired,
                    "native_fidelity_cells": nca_native,
                    "decision": nca_decision,
                    "role": "independent_secondary_direction_diagnosis",
                },
                "h1": h1,
                "execution_provenance": provenance(),
                "independent_decisions": True,
            }
        )
        validate_analysis_report(report, self.card)
        validate_analysis_report(json.loads(json.dumps(report, sort_keys=True)), self.card)

        forged = copy.deepcopy(report)
        forged["primary_contrasts"]["fs_12_15_k2_k3"]["mcnemar_raw_p"] = 0.0
        reversed_interval = copy.deepcopy(report)
        reversed_interval["primary_contrasts"]["fs_12_15_k2_k3"]["paired_ci_pp"] = [1.0, -1.0]
        out_of_range_paired_interval = copy.deepcopy(report)
        out_of_range_paired_interval["primary_contrasts"]["fs_12_15_k2_k3"][
            "paired_ci_pp"
        ] = [-1000.0, 1000.0]
        unknown = copy.deepcopy(report)
        unknown["unregistered_result"] = True
        bool_p = copy.deepcopy(report)
        bool_p["primary_contrasts"]["fs_12_15_k2_k3"]["mcnemar_raw_p"] = True
        excessive_js = copy.deepcopy(report)
        excessive_js["mechanism_diagnostics"]["fs_12_15_k2_k3"]["mean_js_nats"] = math.log(2.0) + 0.01
        negative_subgroup = copy.deepcopy(report)
        negative_subgroup["mechanism_diagnostics"]["fs_12_15_k2_k3"]["wrong_overconfidence"]["subgroups"]["wrong_to_wrong"]["count"] = -1
        contradictory_nca = copy.deepcopy(report)
        contradictory_nca["nca"]["primary_cells"]["11:14_k2"]["valid_count"] = 0
        contradictory_nca["nca"]["primary_cells"]["11:14_k2"]["valid_fraction"] = 0.0
        bool_nca_seed = copy.deepcopy(report)
        bool_nca_seed["nca"]["primary_cells"]["11:14_k2"]["overall_interval"][
            "seed"
        ] = False
        native_count_mismatch = copy.deepcopy(report)
        native_count_mismatch["nca"]["native_fidelity_cells"]["12:15_k2"][
            "valid_subgroup_counts"
        ]["right_to_right"] = 491
        paired_exceeds_source = copy.deepcopy(report)
        paired_exceeds_source["nca"]["primary_cells"]["12:15_k2"][
            "valid_count"
        ] = 500
        paired_exceeds_source["nca"]["primary_cells"]["12:15_k2"][
            "valid_fraction"
        ] = 500 / 512.0
        paired_exceeds_source["nca"]["native_fidelity_cells"]["12:15_k2"][
            "valid_subgroup_counts"
        ]["right_to_right"] = 480
        bool_statistics = copy.deepcopy(report)
        bool_statistics["statistics_contract"]["nca_bootstrap"]["seed"] = False
        bool_bootstrap = copy.deepcopy(report)
        bool_bootstrap["primary_contrasts"]["fs_11_14_k2_k3"]["bootstrap"]["seed"] = False
        bool_body_call = copy.deepcopy(report)
        first_trajectory = next(iter(bool_body_call["calibration_trajectory_summary"].values()))
        first_trajectory["steps"][0]["body_call_t"] = False
        for candidate in (
            forged,
            reversed_interval,
            out_of_range_paired_interval,
            unknown,
            bool_p,
            excessive_js,
            negative_subgroup,
            contradictory_nca,
            bool_nca_seed,
            native_count_mismatch,
            paired_exceeds_source,
            bool_statistics,
            bool_bootstrap,
            bool_body_call,
        ):
            with self.assertRaises((SchemaError, Phase2AnalysisError)):
                validate_analysis_report(self._rehash(candidate), self.card)

    def test_atomic_analysis_writer_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            atomic_write_new_json(target, {"value": 1})
            self.assertEqual(json.loads(target.read_text()), {"value": 1})
            with self.assertRaises(FileExistsError):
                atomic_write_new_json(target, {"value": 2})


if __name__ == "__main__":
    unittest.main()
