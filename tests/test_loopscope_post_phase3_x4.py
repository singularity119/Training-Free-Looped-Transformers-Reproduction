import copy
import unittest
from pathlib import Path

from tflt.loopscope.post_phase3_x4 import (
    AUTHORIZED_RUN_ROOT,
    IMPLEMENTATION_RELATIVE_PATHS,
    K_ORDER,
    X4Error,
    _contrast_order,
    _validate_audit_report,
    build_manifest,
    holm_step_down_three,
    joint_paired_bootstrap,
    load_x4_card,
    validate_manifest,
    validate_scheduler_rows,
)
from tflt.loopscope.phase3_schema import attach_manifest_sha256


def fixture_manifest():
    return build_manifest(
        run_root=AUTHORIZED_RUN_ROOT,
        expected_commit="f" * 40,
        partition="emergency_gpua40",
        qos=None,
        account=None,
        array_throttle=3,
        smoke_time_limit="00:40:00",
        full_time_limit="08:00:00",
        created_at_utc="fixture",
        hashes={path: "0" * 64 for path in IMPLEMENTATION_RELATIVE_PATHS},
        card_file_sha256="1" * 64,
    )


def scheduler_rows(job_id="123"):
    rows = []
    for index in range(3):
        rows.append(
            {
                "JobID": "%s_%d" % (job_id, index),
                "JobIDRaw": str(1000 + index),
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "Partition": "emergency_gpua40",
                "NodeList": "gpu%d" % index,
                "ElapsedRaw": "60",
                "AllocTRES": "gres/gpu=1",
                "Submit": "s",
                "Start": "s",
                "End": "e",
            }
        )
    return rows


def contrast_fixture():
    rows = {}
    for index, name in enumerate(_contrast_order()):
        rows[name] = [1 if index % 3 == 0 else 0, 0, -1, 1]
    return rows


def audit_fixture(k):
    prompt = {
        "decision": {
            "operator_body_calls": k,
            "restore_allclose": True,
            "code": "loop_effective_logits_changed",
            "max_final_logits_diff": 0.25,
        }
    }
    return {
        "loop_config": {
            "window": "15:15",
            "k": k,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
        },
        "patch_target": {
            "window": [15, 15],
            "classes_after_patch": {"15": "LoopedBlock"},
        },
        "overall_decision": {"code": "loop_effective_logits_changed"},
        "prompts": [copy.deepcopy(prompt), copy.deepcopy(prompt)],
    }


class PostPhase3X4Tests(unittest.TestCase):
    def test_card_materializes_exact_authority_and_science(self):
        root = Path(__file__).resolve().parents[1]
        card = load_x4_card(
            root / "configs/loopscope/post_phase3_x4_layer15_k234_card.json"
        )
        self.assertEqual(card["authorized_executor_thread"], "019f6bb3-8f65-7ea1-a2f9-425a7959680f")
        self.assertEqual(tuple(card["frozen_cells"]["ordered_k"]), K_ORDER)
        self.assertEqual(card["frozen_cells"]["window"], "15:15")
        self.assertFalse(card["comparators"]["rerun_comparators"])

    def test_manifest_freezes_three_k_cells_and_limit_boundary(self):
        manifest = fixture_manifest()
        validate_manifest(manifest)
        self.assertEqual(tuple(manifest["smoke"]["k_order"]), K_ORDER)
        self.assertEqual(tuple(manifest["full"]["k_order"]), K_ORDER)
        self.assertEqual(manifest["smoke"]["cell_count"], 3)
        self.assertEqual(manifest["full"]["cell_count"], 3)
        self.assertTrue(all("--limit" in cell["limit_argv"] for cell in manifest["smoke"]["cells"]))
        self.assertTrue(all("--limit" not in cell["argv"] for cell in manifest["full"]["cells"]))
        self.assertEqual(
            [cell["step_size"] for cell in manifest["full"]["cells"]],
            [0.5, 1.0 / 3.0, 0.25],
        )

        changed = copy.deepcopy(manifest)
        changed.pop("manifest_sha256")
        changed["full"]["cells"][0], changed["full"]["cells"][1] = (
            changed["full"]["cells"][1],
            changed["full"]["cells"][0],
        )
        attach_manifest_sha256(changed)
        with self.assertRaisesRegex(X4Error, "exact K2/K3/K4"):
            validate_manifest(changed)

    def test_scheduler_seal_requires_exact_three_completed_zero_exit_tasks(self):
        rows = scheduler_rows()
        accepted = validate_scheduler_rows(rows, "123")
        self.assertEqual([row["JobID"] for row in accepted], ["123_0", "123_1", "123_2"])
        with self.assertRaisesRegex(X4Error, "exact three"):
            validate_scheduler_rows(rows[:-1], "123")
        failed = copy.deepcopy(rows)
        failed[1]["State"] = "FAILED"
        failed[1]["ExitCode"] = "1:0"
        with self.assertRaisesRegex(X4Error, "task 1"):
            validate_scheduler_rows(failed, "123")

    def test_audit_requires_one_layer_calls_equal_k_effect_and_restore(self):
        for k in K_ORDER:
            result = _validate_audit_report(audit_fixture(k), k)
            self.assertEqual(result["single_decoder_layer_body"], 15)
            self.assertEqual(result["operator_body_calls_per_prompt"], [k, k])
            self.assertTrue(result["restore_allclose_all_prompts"])

        changed = audit_fixture(3)
        changed["patch_target"]["classes_after_patch"]["14"] = "LoopedBlock"
        with self.assertRaisesRegex(X4Error, "exactly decoder layer 15"):
            _validate_audit_report(changed, 3)

    def test_joint_bootstrap_and_three_test_holm_are_deterministic(self):
        contrasts = contrast_fixture()
        first = joint_paired_bootstrap(
            contrasts, replicates=101, seed=20260710, force_python=True
        )
        second = joint_paired_bootstrap(
            contrasts, replicates=101, seed=20260710, force_python=True
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["contrast_order"]), 9)
        self.assertTrue(first["joint_index_reuse_all_nine_contrasts"])
        self.assertEqual(len(first["index_stream_sha256"]), 64)

        holm = holm_step_down_three({"K2": 0.001, "K3": 0.02, "K4": 0.5})
        self.assertAlmostEqual(holm["K2"]["adjusted_p"], 0.003)
        self.assertAlmostEqual(holm["K3"]["adjusted_p"], 0.04)
        self.assertAlmostEqual(holm["K4"]["adjusted_p"], 0.5)


if __name__ == "__main__":
    unittest.main()
