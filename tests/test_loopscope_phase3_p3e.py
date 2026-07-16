import copy
import unittest

from tflt.loopscope.phase3_p3e import (
    ALL_WINDOWS,
    AUTHORIZED_RUN_ROOT,
    BLIND_BOTTOM_M,
    BLIND_TOP_M,
    BLIND_WINDOWS,
    P3EError,
    build_full_manifest,
    classify_width4,
    exact_mcnemar_p,
    holm_step_down_25,
    joint_paired_bootstrap,
    validate_full_manifest,
    validate_sealed_scheduler_rows,
)


def scheduler_rows(job_id="123"):
    rows = []
    for index in range(12):
        rows.append(
            {
                "JobIDRaw": "%s_%d" % (job_id, index),
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "Partition": "p",
                "NodeList": "n",
                "ElapsedRaw": "1",
                "AllocTRES": "gres/gpu=1",
                "Submit": "s",
                "Start": "s",
                "End": "e",
            }
        )
    return rows


def fixture_manifest():
    return build_full_manifest(
        run_root=AUTHORIZED_RUN_ROOT,
        expected_commit="f" * 40,
        partition="p",
        qos=None,
        account=None,
        array_throttle=4,
        time_limit="1-00:00:00",
        created_at_utc="fixture",
        hashes={path: "0" * 64 for path in (
            "src/tflt/loopscope/phase3_p3e.py",
            "scripts/loopscope/run_qwen17_phase3_p3e.py",
            "tests/test_loopscope_phase3_p3e.py",
        )},
    )


class Phase3P3ETests(unittest.TestCase):
    def test_exact_full_manifest_has_missing12_once_and_no_limit(self):
        manifest = fixture_manifest()
        validate_full_manifest(manifest)
        self.assertEqual(tuple(manifest["window_order"]), BLIND_WINDOWS)
        self.assertEqual(len({cell["window"] for cell in manifest["cells"]}), 12)
        self.assertTrue(all("--limit" not in cell["argv"] for cell in manifest["cells"]))
        self.assertTrue(all(cell["automatic_retry"] is False for cell in manifest["cells"]))
        changed = copy.deepcopy(manifest)
        changed.pop("manifest_sha256")
        changed["cells"] = changed["cells"][:-1]
        from tflt.loopscope.phase3_schema import attach_manifest_sha256

        attach_manifest_sha256(changed)
        with self.assertRaisesRegex(P3EError, "12 cells"):
            validate_full_manifest(changed)

    def test_partial_or_failed_12_cell_scheduler_set_cannot_unseal(self):
        rows = scheduler_rows()
        accepted = validate_sealed_scheduler_rows(rows, "123")
        self.assertEqual([row["JobIDRaw"] for row in accepted], ["123_%d" % i for i in range(12)])
        with self.assertRaisesRegex(P3EError, "exact 12"):
            validate_sealed_scheduler_rows(rows[:-1], "123")
        failed = copy.deepcopy(rows)
        failed[7]["State"] = "FAILED"
        failed[7]["ExitCode"] = "1:0"
        with self.assertRaisesRegex(P3EError, "task 7"):
            validate_sealed_scheduler_rows(failed, "123")

    def test_joint_bootstrap_reuses_one_stream_and_computes_g_blind(self):
        differences = {window: [0, 0, 0, 0] for window in ALL_WINDOWS}
        differences[BLIND_TOP_M[0]] = [1, 1, 0, 0]
        differences[BLIND_TOP_M[1]] = [1, 0, 1, 0]
        differences[BLIND_BOTTOM_M[0]] = [0, 0, 0, 0]
        differences[BLIND_BOTTOM_M[1]] = [0, 0, 0, 0]
        first = joint_paired_bootstrap(
            differences, replicates=101, seed=20260710, force_python=True
        )
        second = joint_paired_bootstrap(
            differences, replicates=101, seed=20260710, force_python=True
        )
        self.assertEqual(first, second)
        self.assertEqual(first["g_blind"]["point_fraction"], 0.5)
        self.assertTrue(first["joint_index_reuse_all_windows_and_g_blind"])
        self.assertEqual(len(first["index_stream_sha256"]), 64)

    def test_exact_identity_order_shape_is_required_by_bootstrap(self):
        differences = {window: [0, 0] for window in ALL_WINDOWS}
        differences["24:27"] = [0]
        with self.assertRaisesRegex(P3EError, "one positive length"):
            joint_paired_bootstrap(
                differences, replicates=5, seed=1, force_python=True
            )
        reordered = dict(reversed(list(differences.items())))
        reordered["24:27"] = [0, 0]
        with self.assertRaisesRegex(P3EError, "window order"):
            joint_paired_bootstrap(
                reordered, replicates=5, seed=1, force_python=True
            )

    def test_exact_mcnemar_and_all25_holm_path(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 5), 0.0625)
        raw = {window: 1.0 for window in ALL_WINDOWS}
        raw["12:15"] = 0.001
        adjusted = holm_step_down_25(raw)
        self.assertAlmostEqual(adjusted["12:15"]["adjusted_p"], 0.025)
        self.assertTrue(adjusted["12:15"]["significant"])
        with self.assertRaisesRegex(P3EError, "all-25"):
            holm_step_down_25(dict(list(raw.items())[:-1]))

    def test_first_match_width4_labels(self):
        selected = {"delta_accuracy_fraction": 0.01, "holm_significant": False}
        label, checks = classify_width4(
            evidence_complete=True,
            retrospective_recovered=True,
            selected_window_row=selected,
            oracle_windows=["12:15"],
            oracle_inside_support=True,
            deployment_top3=["12:15", "6:9", "7:10"],
            false_positives_excluded=True,
            blind_label="SUPPORTED",
        )
        self.assertEqual(label, "WIDTH4_WITHIN_CELL_FEASIBILITY_SUPPORTED")
        self.assertTrue(checks["frozen_top1_hits_oracle"])
        harmful = {"delta_accuracy_fraction": -0.01, "holm_significant": True}
        label, _ = classify_width4(
            evidence_complete=True,
            retrospective_recovered=True,
            selected_window_row=harmful,
            oracle_windows=["0:3"],
            oracle_inside_support=False,
            deployment_top3=["12:15", "6:9", "7:10"],
            false_positives_excluded=True,
            blind_label="REFUTED",
        )
        self.assertEqual(label, "WIDTH4_HARMFUL_SELECTION")
        label, _ = classify_width4(
            evidence_complete=True,
            retrospective_recovered=True,
            selected_window_row=selected,
            oracle_windows=["0:3"],
            oracle_inside_support=False,
            deployment_top3=["12:15", "6:9", "7:10"],
            false_positives_excluded=True,
            blind_label="INCONCLUSIVE",
        )
        self.assertEqual(label, "WIDTH4_COVERAGE_FAILURE")


if __name__ == "__main__":
    unittest.main()
