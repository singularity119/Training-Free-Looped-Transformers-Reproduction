import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from tflt.loopscope.phase2_analysis import choice_output
from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    DIRECT_PROBE_SCORE_SOURCE,
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    SchemaError,
    make_calibration_cell_envelope,
    make_full_final_output_envelope,
    make_identity_manifest,
    make_source_provenance,
    make_hashed_manifest,
    validate_calibration_cell_envelope,
    validate_full_final_output_envelope,
    validate_identity_manifest,
    validate_phase2_card,
    verify_live_source_provenance,
    validate_phase2_workspace_output_path,
)
from tflt.loopscope.schema import attach_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json"


def identities(count):
    return [
        {"task": "mmlu_subject", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}
        for index in range(count)
    ]


def producer(kind):
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
    return payload


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


def direct_choice(identity):
    return choice_output(
        [4.0, 3.0, 2.0, 1.0],
        identity=identity,
        score_source=DIRECT_PROBE_SCORE_SOURCE,
    )


def trajectory_sample(identity):
    return {
        "sample_identity": dict(identity),
        "answer_position": 7,
        "position_rule": "final_pre_answer_prompt_token",
        "boundary_identity": "native_continuation=B_N-B_(b+1)",
        "body_call_indexing": "zero_based_t=0..K-1",
        "steps": [
            {
                "body_call_t": 0,
                "residual_norm": 1.0,
                "state_norm": 2.0,
                "relative_activity": 0.5,
                "residual_ratio_to_previous": None,
                "adjacent_residual_cosine": None,
                "adjacent_residual_cosine_valid": None,
                "nca": None,
                "valid": True,
            },
            {
                "body_call_t": 1,
                "residual_norm": 0.9,
                "state_norm": 2.1,
                "relative_activity": 0.9 / 2.1,
                "residual_ratio_to_previous": 0.9,
                "adjacent_residual_cosine": 0.8,
                "adjacent_residual_cosine_valid": True,
                "nca": {
                    "value": 0.4,
                    "residual_norm": 0.9,
                    "native_continuation_norm": 1.2,
                    "valid": True,
                    "invalid_reason": None,
                    "dtype": "float32",
                    "provenance": "actual_loop_repeated_step",
                    "body_call_t": 1,
                },
                "valid": True,
            },
        ],
        "repeated_step_nca_all_valid": True,
        "valid": True,
        "errors": [],
        "event_counts": {
            "wrapper_forward": 1,
            "bypass_false": 1,
            "body_call": 2,
            "operator_body_calls": 2,
            "g_minus_x": 2,
            "looped_hidden_vs_input": 1,
            "stash_pass": 1,
            "identity_forward": 3,
        },
        "final_output": direct_choice(identity),
    }


class Phase2ClosedWorldRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    def _rehash(self, payload):
        payload.pop("manifest_sha256", None)
        attach_manifest_sha256(payload)
        return payload

    def test_live_source_provenance_reads_and_verifies_phase1_manifest(self):
        source_identities = identities(512)
        source = make_hashed_manifest(
            {
                "schema_version": "loopscope.probe-pool-manifest.v1",
                "source": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
                "split": "validation",
                "count": 512,
                "seed": 20260710,
                "task_group": "mmlu",
                "num_fewshot": 5,
                "uses_target_gold_labels": False,
                "render_contract_subset_sha256": "b" * 64,
                "renderer": {
                    "dataset_revision": "c30699e8356da336a370243923dbaf21066bb9fe"
                },
                "rendering_records": [
                    {
                        "task_name": row["task"],
                        "target_doc_id": row["doc_id"],
                        "target_doc_sha256": row["doc_hash"],
                    }
                    for row in source_identities
                ],
            }
        )
        provenance = make_source_provenance(
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            verification_status="live_verified",
            source_manifest_path=(
                "/hpc2hdd/home/xhuang225/workspaces/"
                "training_free_looped_transformers/inputs/"
                "loopscope-qwen17-mmlu-phase1-20260711-043615/"
                "probe_pool_manifest.json"
            ),
            source_manifest_sha256=source["manifest_sha256"],
            renderer_subset_sha256="b" * 64,
        )
        with mock.patch.object(Path, "read_text", return_value=json.dumps(source)):
            self.assertEqual(
                verify_live_source_provenance(
                    provenance, CALIBRATION_IDENTITY_NAMESPACE, source_identities
                ),
                source,
            )
        forged_identities = copy.deepcopy(source_identities)
        forged_identities[0]["doc_hash"] = "f" * 64
        with mock.patch.object(Path, "read_text", return_value=json.dumps(source)):
            with self.assertRaises(SchemaError):
                verify_live_source_provenance(
                    provenance, CALIBRATION_IDENTITY_NAMESPACE, forged_identities
                )
        forged = copy.deepcopy(provenance)
        forged["renderer_subset_sha256"] = "c" * 64
        with mock.patch.object(Path, "read_text", return_value=json.dumps(source)):
            with self.assertRaises(SchemaError):
                verify_live_source_provenance(
                    self._rehash(forged), CALIBRATION_IDENTITY_NAMESPACE
                )

    def test_full_identity_is_derived_from_actual_phase1_baseline_results(self):
        expected = identities(14042)
        sample_bytes = json.dumps(
            {"samples": {"mmlu_subject": expected}}, sort_keys=True
        ).encode("utf-8")
        run_root = (
            "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/"
            "loopscope-qwen17-mmlu-phase1-20260711-053022"
        )
        source = make_hashed_manifest(
            {
                "schema_version": "loopscope.phase1-run.v1",
                "run_root": run_root,
                "frozen_recipe": {
                    "repo_id": "Qwen/Qwen3-1.7B-Base",
                    "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                    "task": "mmlu",
                    "num_fewshot": 5,
                    "dtype": "float16",
                },
                "inputs": {
                    "probe_pool": {
                        "count": 512,
                        "render_contract_subset_sha256": "b" * 64,
                    }
                },
                "stages": {
                    "gate-e-full": {
                        "jobs": [{
                            "job_id": "baseline-full",
                            "output_dir": run_root + "/gate-e-full/baseline-full",
                        }]
                    }
                },
            }
        )
        sample_path = run_root + "/gate-e-full/baseline-full/results.json"
        provenance = make_source_provenance(
            identity_namespace=FULL_IDENTITY_NAMESPACE,
            verification_status="live_verified",
            source_manifest_path=run_root + "/control/phase1_run_manifest.json",
            source_manifest_sha256=source["manifest_sha256"],
            renderer_subset_sha256="b" * 64,
            identity_artifacts=[{
                "path": sample_path,
                "sha256": __import__("hashlib").sha256(sample_bytes).hexdigest(),
            }],
        )
        with mock.patch.object(Path, "read_text", return_value=json.dumps(source)), mock.patch.object(
            Path, "read_bytes", return_value=sample_bytes
        ):
            verify_live_source_provenance(
                provenance, FULL_IDENTITY_NAMESPACE, expected
            )
            forged = copy.deepcopy(expected)
            forged[-1]["doc_hash"] = "f" * 64
            with self.assertRaises(SchemaError):
                verify_live_source_provenance(
                    provenance, FULL_IDENTITY_NAMESPACE, forged
                )

        traversal = run_root + "/nested/../../../../tmp/results.json"
        with self.assertRaises(SchemaError):
            make_source_provenance(
                identity_namespace=FULL_IDENTITY_NAMESPACE,
                verification_status="live_verified",
                source_manifest_path=run_root + "/control/phase1_run_manifest.json",
                source_manifest_sha256=source["manifest_sha256"],
                renderer_subset_sha256="b" * 64,
                identity_artifacts=[{
                    "path": traversal,
                    "sha256": "a" * 64,
                }],
            )

        real_resolve = Path.resolve

        def symlink_escape(path, *args, **kwargs):
            if path == Path(sample_path):
                return Path("/tmp/phase1-symlink-escape/results.json")
            return real_resolve(path, *args, **kwargs)

        with mock.patch.object(Path, "resolve", new=symlink_escape), mock.patch.object(
            Path, "read_text", return_value=json.dumps(source)
        ):
            with self.assertRaises(SchemaError):
                verify_live_source_provenance(
                    provenance, FULL_IDENTITY_NAMESPACE, expected
                )

    def test_full_identity_uses_exact_phase1_sidecar_fallback_when_needed(self):
        import hashlib

        expected = identities(14042)
        run_root = (
            "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/"
            "loopscope-qwen17-mmlu-phase1-20260711-053022"
        )
        output_root = run_root + "/gate-e-full/baseline-full"
        results_path = output_root + "/results.json"
        sidecar_path = output_root + "/samples_mmlu_subject_20260711.jsonl"
        results_bytes = json.dumps(
            {"results": {"mmlu_subject": {"acc,none": 0.5}}}, sort_keys=True
        ).encode("utf-8")
        sidecar_bytes = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
            for row in expected
        )
        source = make_hashed_manifest(
            {
                "schema_version": "loopscope.phase1-run.v1",
                "run_root": run_root,
                "frozen_recipe": {
                    "repo_id": "Qwen/Qwen3-1.7B-Base",
                    "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                    "task": "mmlu",
                    "num_fewshot": 5,
                    "dtype": "float16",
                },
                "inputs": {
                    "probe_pool": {
                        "count": 512,
                        "render_contract_subset_sha256": "b" * 64,
                    }
                },
                "stages": {
                    "gate-e-full": {
                        "jobs": [{
                            "job_id": "baseline-full",
                            "output_dir": output_root,
                        }]
                    }
                },
            }
        )
        provenance = make_source_provenance(
            identity_namespace=FULL_IDENTITY_NAMESPACE,
            verification_status="live_verified",
            source_manifest_path=run_root + "/control/phase1_run_manifest.json",
            source_manifest_sha256=source["manifest_sha256"],
            renderer_subset_sha256="b" * 64,
            identity_artifacts=[
                {
                    "path": results_path,
                    "sha256": hashlib.sha256(results_bytes).hexdigest(),
                },
                {
                    "path": sidecar_path,
                    "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                },
            ],
        )
        real_glob = Path.glob

        def exact_glob(path, pattern):
            if path == Path(output_root) and pattern == "samples_*.jsonl":
                return [Path(sidecar_path)]
            return real_glob(path, pattern)

        def artifact_bytes(path):
            if path == Path(results_path):
                return results_bytes
            if path == Path(sidecar_path):
                return sidecar_bytes
            raise AssertionError("unexpected artifact read: %s" % path)

        with mock.patch.object(Path, "read_text", return_value=json.dumps(source)), mock.patch.object(
            Path, "read_bytes", autospec=True, side_effect=artifact_bytes
        ), mock.patch.object(Path, "glob", new=exact_glob):
            verify_live_source_provenance(
                provenance, FULL_IDENTITY_NAMESPACE, expected
            )

        for malformed_samples in ({}, [], None):
            with self.subTest(malformed_samples=malformed_samples):
                malformed_results = json.dumps(
                    {
                        "results": {"mmlu_subject": {"acc,none": 0.5}},
                        "samples": malformed_samples,
                    },
                    sort_keys=True,
                ).encode("utf-8")
                malformed_provenance = make_source_provenance(
                    identity_namespace=FULL_IDENTITY_NAMESPACE,
                    verification_status="live_verified",
                    source_manifest_path=run_root + "/control/phase1_run_manifest.json",
                    source_manifest_sha256=source["manifest_sha256"],
                    renderer_subset_sha256="b" * 64,
                    identity_artifacts=[
                        {
                            "path": results_path,
                            "sha256": hashlib.sha256(malformed_results).hexdigest(),
                        },
                        {
                            "path": sidecar_path,
                            "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                        },
                    ],
                )

                def malformed_artifact_bytes(path):
                    if path == Path(results_path):
                        return malformed_results
                    if path == Path(sidecar_path):
                        return sidecar_bytes
                    raise AssertionError("unexpected artifact read: %s" % path)

                with mock.patch.object(
                    Path, "read_text", return_value=json.dumps(source)
                ), mock.patch.object(
                    Path,
                    "read_bytes",
                    autospec=True,
                    side_effect=malformed_artifact_bytes,
                ), mock.patch.object(Path, "glob", new=exact_glob):
                    with self.assertRaises(SchemaError):
                        verify_live_source_provenance(
                            malformed_provenance,
                            FULL_IDENTITY_NAMESPACE,
                            expected,
                        )

    def test_card_is_closed_world_and_keeps_seal_scale_and_statistics(self):
        validate_phase2_card(self.card)
        cases = []
        unknown = copy.deepcopy(self.card)
        unknown["unknown_science_escape"] = True
        cases.append(unknown)
        unsealed = copy.deepcopy(self.card)
        unsealed["science"]["calibration_gold_sealed"] = False
        cases.append(unsealed)
        wrong_count = copy.deepcopy(self.card)
        wrong_count["science"]["calibration_count"] = 511
        cases.append(wrong_count)
        wrong_seed = copy.deepcopy(self.card)
        wrong_seed["statistics"]["nca_bootstrap"]["seed"] = 1
        cases.append(wrong_seed)
        bool_seed = copy.deepcopy(self.card)
        bool_seed["statistics"]["nca_bootstrap"]["seed"] = False
        cases.append(bool_seed)
        bool_beta = copy.deepcopy(self.card)
        bool_beta["science"]["beta"] = False
        cases.append(bool_beta)
        bool_decision = copy.deepcopy(self.card)
        bool_decision["decision_rules"]["h1"]["perturbation"]["delta_threshold_pp"] = False
        cases.append(bool_decision)
        wrong_replicates = copy.deepcopy(self.card)
        wrong_replicates["statistics"]["full_paired_bootstrap"]["replicates"] = 1999
        cases.append(wrong_replicates)
        reversed_interval = copy.deepcopy(self.card)
        reversed_interval["statistics"]["nca_bootstrap"]["interval"] = [97.5, 2.5]
        cases.append(reversed_interval)
        nonpositive_k = copy.deepcopy(self.card)
        nonpositive_k["trajectories"]["fixed_step"][0]["k"] = 0
        cases.append(nonpositive_k)
        full_vectors = copy.deepcopy(self.card)
        full_vectors["write_once_contract"]["full_vectors_persisted"] = True
        cases.append(full_vectors)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(SchemaError):
                    validate_phase2_card(self._rehash(payload))

    def test_all_new_evidence_roots_are_inside_independent_loopscope_workspace(self):
        root = self.card["write_once_contract"]["workspace_root"]
        accepted = validate_phase2_workspace_output_path(
            root + "/runs/phase2-fixture", self.card, context="test output"
        )
        self.assertTrue(str(accepted).startswith(root + "/"))
        for rejected in (
            "/tmp/phase2-fixture",
            "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/legacy",
            root + "/../training_free_looped_transformers/runs/escape",
        ):
            with self.subTest(rejected=rejected), self.assertRaises(SchemaError):
                validate_phase2_workspace_output_path(
                    rejected, self.card, context="test output"
                )

    def test_identity_manifest_rejects_duplicates_and_sidecar_reordering(self):
        ordered = identities(512)
        manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=ordered,
            source_provenance=source_provenance(CALIBRATION_IDENTITY_NAMESPACE),
        )
        validate_identity_manifest(manifest, self.card)

        duplicate = copy.deepcopy(manifest)
        duplicate["ordered_sample_identity"][-1] = duplicate["ordered_sample_identity"][0]
        duplicate["ordered_identity_sha256"] = duplicate["ordered_identity_sha256"]
        with self.assertRaises(SchemaError):
            validate_identity_manifest(self._rehash(duplicate), self.card)

        cell = {
            "cell_id": "shared_k2_anchor_12_15_k2_a1",
            "protocol": "shared_k2_anchor",
            "window": "12:15",
            "k": 2,
            "alpha": 1.0,
            "step_size": 0.5,
        }
        samples = [trajectory_sample(identity) for identity in ordered]
        envelope = make_calibration_cell_envelope(
            self.card, manifest, cell=cell, samples=samples,
            producer=producer("gate_b_remote_calibration_probe"),
        )
        validate_calibration_cell_envelope(envelope, self.card, manifest)
        reordered = copy.deepcopy(envelope)
        reordered["samples"] = list(reversed(reordered["samples"]))
        with self.assertRaises(SchemaError):
            validate_calibration_cell_envelope(self._rehash(reordered), self.card, manifest)

    def test_sealed_calibration_rejects_gold_malformed_choice_and_out_of_range_nca(self):
        ordered = identities(512)
        manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=ordered,
            source_provenance=source_provenance(CALIBRATION_IDENTITY_NAMESPACE),
        )
        cell = {
            "cell_id": "shared_k2_anchor_12_15_k2_a1",
            "protocol": "shared_k2_anchor",
            "window": "12:15",
            "k": 2,
            "alpha": 1.0,
            "step_size": 0.5,
        }
        base = make_calibration_cell_envelope(
            self.card,
            manifest,
            cell=cell,
            samples=[trajectory_sample(identity) for identity in ordered],
            producer=producer("gate_b_remote_calibration_probe"),
        )

        gold = copy.deepcopy(base)
        gold["samples"][0]["final_output"]["correctness"] = True
        malformed = copy.deepcopy(base)
        malformed["samples"][0]["final_output"]["choice_probabilities"][0] = -0.1
        cosine = copy.deepcopy(base)
        cosine["samples"][0]["steps"][1]["nca"]["value"] = 1.01
        for payload in (gold, malformed, cosine):
            with self.assertRaises(SchemaError):
                validate_calibration_cell_envelope(self._rehash(payload), self.card, manifest)

    def test_full_sidecar_forbids_residual_and_cross_namespace_join(self):
        ordered = identities(14042)
        full_manifest = make_identity_manifest(
            self.card,
            identity_namespace=FULL_IDENTITY_NAMESPACE,
            split="mmlu_test_full",
            identities=ordered,
            source_provenance=source_provenance(FULL_IDENTITY_NAMESPACE),
        )
        cell = {
            "cell_id": "fixed_step_12_15_k3_a1p5",
            "protocol": "fixed_step",
            "window": "12:15",
            "k": 3,
            "alpha": 1.5,
            "step_size": 0.5,
        }
        samples = [
            choice_output(
                [4.0, 3.0, 2.0, 1.0],
                identity=identity,
                score_source=FULL_LM_EVAL_SCORE_SOURCE,
                gold_index=0,
                evaluator_acc=True,
            )
            for identity in ordered
        ]
        envelope = make_full_final_output_envelope(
            self.card,
            full_manifest,
            cell=cell,
            samples=samples,
            producer=producer("lm_eval_logged_samples_adapter"),
        )
        validate_full_final_output_envelope(envelope, self.card, full_manifest)

        residual = copy.deepcopy(envelope)
        residual["samples"][0]["residual_norm"] = 1.0
        with self.assertRaises(SchemaError):
            validate_full_final_output_envelope(self._rehash(residual), self.card, full_manifest)

        calibration_manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=identities(512),
            source_provenance=source_provenance(CALIBRATION_IDENTITY_NAMESPACE),
        )
        with self.assertRaises(SchemaError):
            validate_full_final_output_envelope(envelope, self.card, calibration_manifest)


if __name__ == "__main__":
    unittest.main()
