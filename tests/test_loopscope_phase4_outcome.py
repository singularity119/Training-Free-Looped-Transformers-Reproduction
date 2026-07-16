from __future__ import annotations

import copy
import unittest

from tflt.loopscope.phase4_outcome import (
    AUTHORIZED_RUN_ROOT,
    AUTHORIZED_BASE_COMMIT,
    BATCH_SIZE,
    CELLS,
    P4CError,
    SEEDS,
    SMOKE_IDENTITIES,
    build_launch_manifest,
    _submit_text,
    validate_launch_manifest,
)


class Phase4OutcomeContractTests(unittest.TestCase):
    def build(self, mode="smoke"):
        return build_launch_manifest(
            mode=mode,
            run_root=AUTHORIZED_RUN_ROOT,
            expected_commit=AUTHORIZED_BASE_COMMIT,
            partition="gpu-a40",
            gres="gpu:a40:1",
            qos=None,
            account=None,
            array_throttle=4,
            time_limit="01:00:00",
            created_at_utc="TEST",
            hashes={path: "1" * 64 for path in (
                "src/tflt/loopscope/phase4_outcome.py",
                "scripts/loopscope/run_qwen4_phase4_p4c.py",
                "tests/test_loopscope_phase4_outcome.py",
            )},
        )

    def test_exact_eight_cell_order_and_baseline(self):
        value = self.build()
        observed = [(row["cell_id"], row["role"], row["window"]) for row in value["cells"]]
        self.assertEqual(observed, list(CELLS))
        self.assertFalse(value["cells"][0]["loop_enabled"])
        self.assertTrue(all(row["loop_enabled"] for row in value["cells"][1:]))
        self.assertEqual(value["recipe"]["batch_size"], BATCH_SIZE)
        self.assertEqual(value["recipe"]["seeds"], SEEDS)

    def test_smoke_is_exact_five_and_full_has_no_limit(self):
        smoke = self.build("smoke")
        full = self.build("full")
        self.assertEqual(smoke["recipe"]["limit"], 5)
        self.assertEqual(smoke["recipe"]["task"], "mmlu_pro_business")
        self.assertEqual(smoke["population"]["smoke_identities"], list(SMOKE_IDENTITIES))
        self.assertIsNone(full["recipe"]["limit"])
        self.assertEqual(full["recipe"]["task"], "mmlu_pro")

    def assert_rejected(self, mutate):
        value = self.build()
        mutate(value)
        body = {key: item for key, item in value.items() if key != "manifest_sha256"}
        from tflt.loopscope.phase4_schema import semantic_sha256
        value["manifest_sha256"] = semantic_sha256(body)
        with self.assertRaises(P4CError):
            validate_launch_manifest(value, mode="smoke")

    def test_missing_extra_duplicate_reordered_cells_fail_closed(self):
        self.assert_rejected(lambda value: value["cells"].pop())
        self.assert_rejected(lambda value: value["cells"].append(copy.deepcopy(value["cells"][-1])))
        self.assert_rejected(lambda value: value["cells"].__setitem__(1, copy.deepcopy(value["cells"][2])))
        self.assert_rejected(lambda value: value["cells"].__setitem__(slice(1, 3), list(reversed(value["cells"][1:3]))))

    def test_nonpanel_variable_width_recipe_and_hash_drift_fail_closed(self):
        self.assert_rejected(lambda value: value["cells"][1].__setitem__("window", "15:19"))
        self.assert_rejected(lambda value: value["cells"][2].__setitem__("window", "7:10"))
        self.assert_rejected(lambda value: value["recipe"].__setitem__("k", 2))
        self.assert_rejected(lambda value: value["recipe"].__setitem__("batch_size", "auto"))
        self.assert_rejected(lambda value: value["frozen_inputs"].__setitem__("p4b_panel_file_sha256", "0" * 64))

    def test_unknown_mode_and_bad_throttle_rejected(self):
        with self.assertRaises(P4CError):
            self.build("variable-width")
        with self.assertRaises(P4CError):
            build_launch_manifest(
                mode="smoke", run_root=AUTHORIZED_RUN_ROOT,
                expected_commit=AUTHORIZED_BASE_COMMIT, partition="gpu-a40",
                gres="gpu:a40:1", qos=None, account=None, array_throttle=9,
                time_limit="01:00:00", created_at_utc="TEST", hashes={},
            )

    def test_submit_script_keeps_slurm_array_placeholders(self):
        text = _submit_text("smoke", self.build("smoke"))
        self.assertIn("p4c-smoke-%A_%a.out", text)
        self.assertIn("p4c-smoke-%A_%a.err", text)


if __name__ == "__main__":
    unittest.main()
