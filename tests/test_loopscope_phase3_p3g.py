import copy
import unittest

from tflt.loopscope.phase3_p3g import (
    AUTHORIZED_RUN_ROOT,
    CANDIDATE_WINDOWS,
    P3GError,
    _smoke_identity,
    build_manifest,
    classify_targeted_label,
    holm_step_down_four,
    joint_paired_bootstrap,
    validate_manifest,
    validate_scheduler_rows,
)
from tflt.loopscope.phase3_schema import attach_manifest_sha256


IMPLEMENTATION_PATHS = (
    "src/tflt/loopscope/phase3_p3g.py",
    "scripts/loopscope/run_qwen17_phase3_p3g.py",
    "tests/test_loopscope_phase3_p3g.py",
)


def fixture_manifest():
    return build_manifest(
        run_root=AUTHORIZED_RUN_ROOT,
        expected_commit="f" * 40,
        partition="emergency_gpua40",
        qos=None,
        account=None,
        array_throttle=4,
        smoke_time_limit="00:40:00",
        full_time_limit="04:00:00",
        created_at_utc="fixture",
        hashes={path: "0" * 64 for path in IMPLEMENTATION_PATHS},
    )


def scheduler_rows(job_id="123"):
    rows = []
    for index in range(4):
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
    result = {}
    for index, window in enumerate(CANDIDATE_WINDOWS):
        result["%s_vs_baseline" % window] = [1 if index == 0 else 0, 0, -1, 1]
        result["%s_vs_12:15" % window] = [1 if index < 2 else 0, 0, -1, 0]
    return result


class Phase3P3GTests(unittest.TestCase):
    def test_manifest_freezes_exact_smoke_and_full_matrix(self):
        manifest = fixture_manifest()
        validate_manifest(manifest)
        self.assertEqual(tuple(manifest["smoke"]["window_order"]), CANDIDATE_WINDOWS)
        self.assertEqual(tuple(manifest["full"]["window_order"]), CANDIDATE_WINDOWS)
        self.assertEqual(manifest["smoke"]["cell_count"], 4)
        self.assertEqual(manifest["full"]["cell_count"], 4)
        self.assertTrue(
            all("--limit" in cell["limit_argv"] for cell in manifest["smoke"]["cells"])
        )
        self.assertTrue(
            all("--limit" not in cell["argv"] for cell in manifest["full"]["cells"])
        )
        self.assertEqual(
            manifest["required_summary_artifacts"],
            [
                "p3g_manifest.json",
                "p3g_targeted_full_analysis.json",
                "p3g_verifier_receipt.json",
            ],
        )

        changed = copy.deepcopy(manifest)
        changed.pop("manifest_sha256")
        changed["full"]["cells"][0], changed["full"]["cells"][1] = (
            changed["full"]["cells"][1],
            changed["full"]["cells"][0],
        )
        attach_manifest_sha256(changed)
        with self.assertRaisesRegex(P3GError, "exact four cells"):
            validate_manifest(changed)

    def test_scheduler_seal_requires_exact_four_completed_zero_exit_tasks(self):
        rows = scheduler_rows()
        accepted = validate_scheduler_rows(rows, "123")
        self.assertEqual(
            [row["JobID"] for row in accepted], ["123_%d" % index for index in range(4)]
        )
        with self.assertRaisesRegex(P3GError, "exact four"):
            validate_scheduler_rows(rows[:-1], "123")
        failed = copy.deepcopy(rows)
        failed[2]["State"] = "FAILED"
        failed[2]["ExitCode"] = "1:0"
        with self.assertRaisesRegex(P3GError, "task 2"):
            validate_scheduler_rows(failed, "123")

    def test_joint_bootstrap_reuses_one_stream_for_exact_eight_contrasts(self):
        contrasts = contrast_fixture()
        first = joint_paired_bootstrap(
            contrasts, replicates=101, seed=20260710, force_python=True
        )
        second = joint_paired_bootstrap(
            contrasts, replicates=101, seed=20260710, force_python=True
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["contrast_order"]), 8)
        self.assertTrue(first["joint_index_reuse_all_eight_contrasts"])
        self.assertEqual(len(first["index_stream_sha256"]), 64)
        self.assertEqual(
            first["contrasts"]["7:9_vs_baseline"]["point_delta_accuracy_fraction"],
            0.25,
        )

        reordered = dict(reversed(list(contrasts.items())))
        with self.assertRaisesRegex(P3GError, "contrast order"):
            joint_paired_bootstrap(
                reordered, replicates=5, seed=1, force_python=True
            )

    def test_four_test_holm_and_frozen_labels(self):
        raw = {
            "7:9": 0.001,
            "2:9": 0.02,
            "13:15": 0.5,
            "3:7": 1.0,
        }
        adjusted = holm_step_down_four(raw)
        self.assertAlmostEqual(adjusted["7:9"]["adjusted_p"], 0.004)
        self.assertTrue(adjusted["7:9"]["significant"])

        strong = {
            "versus_12_15": {
                "delta_accuracy_fraction": 0.01,
                "paired_ci95_fraction": [0.001, 0.02],
                "holm_adjusted_mcnemar_p": 0.01,
            }
        }
        label, checks = classify_targeted_label(strong)
        self.assertEqual(label, "TARGETED_VARIABLE_WIDTH_STRONG_IMPROVEMENT")
        self.assertTrue(all(checks.values()))

        point = copy.deepcopy(strong)
        point["versus_12_15"]["paired_ci95_fraction"][0] = -0.001
        label, _ = classify_targeted_label(point)
        self.assertEqual(label, "TARGETED_VARIABLE_WIDTH_POINT_IMPROVEMENT")

        none = copy.deepcopy(strong)
        none["versus_12_15"]["delta_accuracy_fraction"] = 0.0
        label, _ = classify_targeted_label(none)
        self.assertEqual(label, "TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT")

    def test_smoke_structure_counts_57_tasks_and_285_samples_without_metrics(self):
        samples = {}
        for task_index in range(57):
            task = "mmlu_subject_%02d" % task_index
            samples[task] = [
                {"doc_id": str(doc_id), "doc_hash": ("%064x" % (task_index * 5 + doc_id))}
                for doc_id in range(5)
            ]
        structure = _smoke_identity({"samples": samples, "results": {"acc,none": 0.0}})
        self.assertEqual(structure["task_count"], 57)
        self.assertEqual(structure["sample_count"], 285)
        self.assertEqual(len(structure["ordered_structural_identity_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
