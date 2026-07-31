import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tflt.loopscope.phase6_acquisition import build_sanitized_trajectory_record
from tflt.loopscope.phase4_schema import file_sha256, semantic_sha256
from tflt.loopscope.phase6_runtime import semantic_sha256 as runtime_record_sha256
from tflt.loopscope.phase6_schema import ELIGIBILITY_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/loopscope/run_qwen4_phase6_gate_d.py"
SPEC = importlib.util.spec_from_file_location("phase6_gate_d", SCRIPT)
gate_d = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate_d)

SHA = "a" * 64


def _eligibility_record(identity: str, category: str, state: str):
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "canonical_identity": identity,
        "category": category,
        "eligibility_state": state,
        "generation_count": 1,
        "sealed_payload_sha256": SHA,
        "sealed_membership_ref": "sealed_baseline/%s.bin" % identity,
        "provenance": {
            "producer_version": "gate-d2-test",
            "card_sha256": SHA,
            "renderer_manifest_sha256": SHA,
            "answer_span_extractor_sha256": SHA,
        },
    }


def _record(identity: str, payload_sha256: str):
    logits = np.zeros((37, 5), dtype=np.float32)
    logits[:, 0] = np.arange(37, dtype=np.float32)
    normalized = np.stack(
        [
            np.asarray([1.0 + index / 10.0, 0.5, -0.25], dtype=np.float32)
            for index in range(37)
        ]
    )
    raw = np.stack(
        [
            np.asarray([1.0, index / 20.0, 0.25], dtype=np.float32)
            for index in range(37)
        ]
    )
    return build_sanitized_trajectory_record(
        model_repo="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        dataset_repo="TIGER-Lab/MMLU-Pro",
        dataset_revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        split="test",
        canonical_identity=identity,
        category="biology",
        prompt_sha256=SHA,
        generated_completion_sha256=payload_sha256,
        generation_length=8,
        replay_length=3,
        anchor_token_index=2,
        answer_span_start_offset=10,
        answer_span_end_offset=11,
        answer_match_count=2,
        selected_match_ordinal=0,
        answer_first_token_index=3,
        answer_span_extractor_sha256=SHA,
        generated_id_text_aligner_sha256=SHA,
        generation_prefix_ids=[11, 12, 13],
        replay_ids=[11, 12, 13],
        boundary_logits=logits,
        final_normalized_vectors=normalized,
        raw_boundaries=raw,
        provenance={
            "producer_version": "gate-d-test",
            "card_sha256": SHA,
            "renderer_manifest_sha256": SHA,
            "tokenizer_manifest_sha256": SHA,
            "boundary_capture": "raw_B0_through_B36_final_norm_pre_hook",
            "final_norm_path": "model.norm",
            "lm_head_path": "lm_head",
        },
    )


class GateDManifestTests(unittest.TestCase):
    def test_exact_executor_and_planning_bindings(self):
        self.assertEqual(
            gate_d.EXECUTOR_THREAD_ID,
            "019fb8f3-8cb7-7c93-a8a0-bae811735601",
        )
        self.assertEqual(
            gate_d.PLANNING_THREAD_ID,
            "019fb3de-2298-75f2-a083-0dca453ea79c",
        )
        self.assertEqual(
            gate_d.ORDERED_IDENTITY_SHA256,
            "ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5",
        )

    def test_shards_are_exact_ordinal_modulo_and_bounded(self):
        members = [
            {
                "ordinal": index,
                "canonical_identity": "identity-%05d" % index,
                "category": "biology",
                "question_id": index,
                "safe_content_sha256": SHA,
                "rendered_prefix_sha256": SHA,
                "rendered_token_ids_sha256": SHA,
                "sequence_length": 10,
            }
            for index in range(12032)
        ]
        manifest = gate_d._build_shard_manifest(
            members,
            run_root=Path("/tmp/gate-d-test"),
            shard_count=8,
            population_manifest_sha256=SHA,
            expected_commit="b" * 40,
        )
        self.assertEqual(manifest["shard_count"], 8)
        self.assertEqual(manifest["max_concurrent_gpus"], 8)
        self.assertEqual(
            [row["ordinal"] for row in manifest["shards"][3]["members"]],
            list(range(3, 12032, 8)),
        )
        flattened = [
            row["canonical_identity"]
            for shard in manifest["shards"]
            for row in shard["members"]
        ]
        self.assertEqual(len(flattened), 12032)
        self.assertEqual(len(set(flattened)), 12032)
        self.assertEqual(manifest["manifest_sha256"], gate_d._manifest_hash(manifest))
        for invalid in (7, 33):
            with self.assertRaisesRegex(gate_d.GateDError, r"\[8,32\]"):
                gate_d._build_shard_manifest(
                    members,
                    run_root=Path("/tmp/gate-d-test"),
                    shard_count=invalid,
                    population_manifest_sha256=SHA,
                    expected_commit="b" * 40,
                )

    def test_launcher_uses_same_acquire_and_fresh_process_verify_path(self):
        debug = gate_d._launcher_text(
            mode="debug",
            run_root=Path("/tmp/debug"),
            gate_b_manifest_path=Path("/tmp/gate_b.json"),
            expected_commit="b" * 40,
        )
        formal = gate_d._launcher_text(
            mode="formal",
            run_root=Path("/tmp/formal"),
            gate_b_manifest_path=Path("/tmp/gate_b.json"),
            expected_commit="b" * 40,
        )
        self.assertIn("acquire-debug", debug)
        self.assertIn("verify-debug", debug)
        self.assertIn("acquire-shard", formal)
        self.assertIn("verify-shard", formal)
        self.assertIn("${SLURM_ARRAY_TASK_ID}", formal)
        self.assertNotIn("selector", debug.lower())
        self.assertNotIn("selector", formal.lower())


class GateDArtifactTests(unittest.TestCase):
    def test_coverage_floors_are_inclusive_and_fail_closed(self):
        members = [
            {
                "ordinal": index,
                "canonical_identity": "identity-%04d" % index,
                "category": "biology",
            }
            for index in range(200)
        ]
        exact_floor = [
            _eligibility_record(
                member["canonical_identity"],
                member["category"],
                "ANCHOR_ELIGIBLE" if index < 199 else "ANCHOR_NOT_EXPRESSED",
            )
            for index, member in enumerate(members)
        ]
        summary = gate_d._coverage_summary(
            members=members,
            eligibility_records=exact_floor,
            enforce_floors=True,
        )
        self.assertEqual(summary["eligible_count"], 199)
        self.assertEqual(summary["not_expressed_count"], 1)
        self.assertEqual(summary["overall_coverage"], 0.995)

        below_overall = [dict(row) for row in exact_floor]
        below_overall[-2] = dict(below_overall[-2], eligibility_state="ANCHOR_NOT_EXPRESSED")
        with self.assertRaisesRegex(
            gate_d.GateDError, "BLOCK_ANCHOR_COVERAGE_BELOW_FLOOR"
        ):
            gate_d._coverage_summary(
                members=members,
                eligibility_records=below_overall,
                enforce_floors=True,
            )

    def test_per_category_floor_and_canonical_order_are_enforced(self):
        members = []
        for index in range(1000):
            members.append(
                {
                    "ordinal": index,
                    "canonical_identity": "identity-%04d" % index,
                    "category": "biology" if index < 100 else "law",
                }
            )
        records = [
            _eligibility_record(
                member["canonical_identity"],
                member["category"],
                (
                    "ANCHOR_NOT_EXPRESSED"
                    if member["category"] == "biology" and index in (98, 99)
                    else "ANCHOR_ELIGIBLE"
                ),
            )
            for index, member in enumerate(members)
        ]
        summary = gate_d._coverage_summary(
            members=members,
            eligibility_records=records,
            enforce_floors=True,
        )
        self.assertEqual(summary["categories"]["biology"]["coverage"], 0.98)

        below_category = [dict(row) for row in records]
        below_category[97] = dict(
            below_category[97], eligibility_state="ANCHOR_NOT_EXPRESSED"
        )
        with self.assertRaisesRegex(
            gate_d.GateDError, "BLOCK_ANCHOR_COVERAGE_BELOW_FLOOR"
        ):
            gate_d._coverage_summary(
                members=members,
                eligibility_records=below_category,
                enforce_floors=True,
            )

        with self.assertRaisesRegex(gate_d.GateDError, "canonical order"):
            gate_d._coverage_summary(
                members=members,
                eligibility_records=list(reversed(records)),
                enforce_floors=False,
            )

    def test_opaque_payload_is_hash_checked_without_json_or_text_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt-0001"
            output.mkdir()
            sealed_dir = output / "sealed_baseline"
            sealed_dir.mkdir(mode=0o700)
            payload = b"\xff\xfeopaque-not-utf8"
            payload_path = sealed_dir / "000000.bin"
            payload_sha = gate_d._write_new_bytes(payload_path, payload, mode=0o600)
            record = _record("identity-0", payload_sha)
            eligibility = _eligibility_record(
                "identity-0", "biology", "ANCHOR_ELIGIBLE"
            )
            eligibility["sealed_payload_sha256"] = payload_sha
            eligibility["sealed_membership_ref"] = "sealed_baseline/000000.bin"
            member = {
                "ordinal": 0,
                "canonical_identity": "identity-0",
                "category": "biology",
                "question_id": 0,
                "safe_content_sha256": SHA,
                "rendered_prefix_sha256": SHA,
                "rendered_token_ids_sha256": SHA,
                "sequence_length": 10,
            }
            membership_sha = semantic_sha256([member])
            closure = {
                "ordinal": 0,
                "canonical_identity": "identity-0",
                "eligibility_state": "ANCHOR_ELIGIBLE",
                "prompt_sha256": SHA,
                "generated_completion_sha256": payload_sha,
                "generation_ids_sha256": record["provenance"]["generation_ids_sha256"],
                "replay_ids_sha256": record["provenance"]["replay_ids_sha256"],
                "generation_length": 8,
                "replay_length": 3,
                "anchor_token_index": 2,
                "answer_span_start_offset": 10,
                "answer_span_end_offset": 11,
                "answer_match_count": 2,
                "selected_match_ordinal": 0,
                "answer_first_token_index": 3,
                "generation_count": 1,
                "replay_count": 1,
                "trajectory_count": 1,
                "loop_insertions": 0,
                "anchor_resolved": True,
                "unique_token_mapping": True,
                "record_semantic_sha256": runtime_record_sha256(record),
                "final_norm_pre_hook_count": 1,
                "raw_boundary_count": 37,
                "final_norm_postnorm_allclose": True,
                "final_norm_postnorm_max_abs": 0.0,
                "replay_next_token_closure": True,
                "replay_sequence_length": 3,
            }
            records_sha = gate_d._write_new_jsonl(
                output / gate_d.SANITIZED_NAME, [record]
            )
            eligibility_sha = gate_d._write_new_jsonl(
                output / gate_d.ELIGIBILITY_NAME, [eligibility]
            )
            closure_sha = gate_d._write_new_jsonl(
                output / gate_d.CLOSURE_NAME, [closure]
            )
            sealed = {
                "schema_version": "loopscope.phase6.gate-d2-sealed-membership.v1",
                "payload_format": "opaque_utf8_completion_bytes_not_parsed_before_gate_g",
                "record_count": 1,
                "membership_sha256": membership_sha,
                "members": [
                    {
                        "ordinal": 0,
                        "canonical_identity": "identity-0",
                        "payload_relative": "sealed_baseline/000000.bin",
                        "byte_count": len(payload),
                        "sha256": payload_sha,
                    }
                ],
            }
            sealed["manifest_sha256"] = gate_d._manifest_hash(sealed)
            sealed_sha = gate_d._write_new_json(
                output / gate_d.SEALED_MEMBERSHIP_NAME, sealed
            )
            receipt = {
                "schema_version": gate_d.SHARD_RECEIPT_SCHEMA,
                "status": "COMPLETED",
                "mode": "formal",
                "shard_id": 0,
                "attempt": 1,
                "git": {"commit": "b" * 40},
                "membership_sha256": membership_sha,
                "record_count": 1,
                "trajectory_count": 1,
                "runtime_closure_count": 1,
                "eligibility_count": 1,
                "coverage": gate_d._coverage_summary(
                    members=[member],
                    eligibility_records=[eligibility],
                    enforce_floors=False,
                ),
                "artifacts": {
                    gate_d.SANITIZED_NAME: records_sha,
                    gate_d.ELIGIBILITY_NAME: eligibility_sha,
                    gate_d.CLOSURE_NAME: closure_sha,
                    gate_d.SEALED_MEMBERSHIP_NAME: sealed_sha,
                },
                "information_barrier": {
                    "protected_target_fields_accessed": False,
                    "sealed_payload_content_inspected_after_write": False,
                    "selector_executed": False,
                    "outcomes_read": False,
                    "later_gate_entered": False,
                },
            }
            receipt["manifest_sha256"] = gate_d._manifest_hash(receipt)
            gate_d._write_new_json(output / gate_d.RECEIPT_NAME, receipt)
            verified = gate_d._validate_attempt(
                output=output,
                members=[member],
                membership_sha256=membership_sha,
                expected_commit="b" * 40,
                mode="formal",
                shard_id=0,
                attempt=1,
            )
            self.assertEqual(len(verified["records"]), 1)
            self.assertEqual(file_sha256(payload_path), payload_sha)

    def test_write_once_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            gate_d._write_new_bytes(path, b"first")
            with self.assertRaises(FileExistsError):
                gate_d._write_new_bytes(path, b"second")

    def test_not_expressed_has_no_persisted_runtime_closure(self):
        evidence = {
            "canonical_identity": "identity-not-expressed",
            "eligibility_state": "ANCHOR_NOT_EXPRESSED",
            "generation_count": 1,
            "replay_count": 0,
            "trajectory_count": 0,
            "loop_insertions": 0,
            "sealed_payload_sha256": SHA,
        }
        with self.assertRaisesRegex(
            gate_d.GateDError, "only eligible runtime closures"
        ):
            gate_d._safe_runtime_closure(0, evidence)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eligible-only.jsonl"
            gate_d._write_new_jsonl(path, [], allow_empty=True)
            self.assertEqual(gate_d._strict_jsonl(path, allow_empty=True), [])


class GateDParserTests(unittest.TestCase):
    def test_parser_exposes_only_gate_d_commands(self):
        parser = gate_d.build_parser()
        self.assertEqual(parser.parse_args(["dry-run"]).command, "dry-run")
        common = [
            "--run-root",
            "/tmp/fresh",
            "--gate-b-manifest",
            "/tmp/gate_b.json",
            "--expected-commit",
            "b" * 40,
        ]
        self.assertEqual(
            parser.parse_args(["freeze-formal", *common, "--shard-count", "8"]).shard_count,
            8,
        )
        shard = parser.parse_args(
            ["acquire-shard", *common, "--shard-id", "2", "--attempt", "1"]
        )
        self.assertEqual((shard.shard_id, shard.attempt), (2, 1))
        scheduler = parser.parse_args(
            [
                "record-scheduler",
                *common,
                "--job-id",
                "101",
                "--job-id",
                "102",
            ]
        )
        self.assertEqual(scheduler.job_ids, ["101", "102"])


if __name__ == "__main__":
    unittest.main()
