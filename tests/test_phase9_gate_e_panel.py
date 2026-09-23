import unittest
from copy import deepcopy

from tflt.loopscope.phase9_gate_e_accuracy import (
    EXPECTED_NEW_COUNT,
    EXPECTED_TEST_COUNT,
    load_config,
    new_cells,
    score_manifest,
    validate_score_manifest,
    validate_config,
)


class Phase9GateEPanelTests(unittest.TestCase):
    def test_extension_panel_is_only_three_windows_times_k3_k4_and_two_arms(self):
        config = load_config()
        cells = new_cells(config)
        self.assertEqual(len(cells), EXPECTED_NEW_COUNT)
        self.assertEqual({cell["k"] for cell in cells}, {3, 4})
        self.assertEqual({cell["arm"] for cell in cells}, {"Lag1-Online", "Lag1-Matched-norm"})
        self.assertTrue(all(cell["direction_policy"] == "lag1" for cell in cells))
        self.assertEqual({cell["intervention_mode"] for cell in cells}, {"spectral", "matched_norm"})
        self.assertEqual({(cell["model"], tuple(cell["window"])) for cell in cells}, {
            ("Qwen/Qwen3-4B-Base", (13, 16)),
            ("Qwen/Qwen3-4B-Base", (15, 18)),
            ("Qwen/Qwen3-1.7B-Base", (12, 15)),
        })
        self.assertTrue(all(cell["fit_semantics"] == "lag1_previous_round_native_full_prefix" for cell in cells))

    def test_score_manifest_is_gold_free_and_exactly_168504_records(self):
        config = load_config()
        value = score_manifest(config, pool_path="/frozen/test_pool.json", scope="FORMAL_TEST")
        cells = validate_score_manifest(value, config, pool_path="/frozen/test_pool.json", scope="FORMAL_TEST")
        self.assertEqual(len(cells), 12)
        self.assertEqual(value["sample_count_per_cell"], EXPECTED_TEST_COUNT)
        self.assertEqual(value["new_configuration_sample_records"], 168504)
        self.assertEqual(value["direction_policies"], ["lag1"])
        self.assertEqual(value["intervention_modes"], ["spectral", "matched_norm"])
        self.assertFalse(value["target_gold_loaded"])

    def test_config_requires_both_switches_and_lag1_formal_scope(self):
        config = load_config()
        for key in (
            "direction_policies",
            "intervention_modes",
            "formal_direction_policy",
            "direction_policy_rules",
            "direction_svd",
            "direction_strength",
            "intervention_rule",
        ):
            invalid = deepcopy(config)
            invalid.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_config_freezes_each_policy_and_intervention_semantic(self):
        config = load_config()
        mutations = (
            ("direction_policy_rules", "lag1", "used_at", "same round"),
            ("direction_policy_rules", "fixed_t0", "fit_at", "every round"),
            ("intervention_rule", "spectral", None, "identity"),
            ("intervention_rule", "matched_norm", None, "normalize input"),
        )
        for section, key, field, replacement in mutations:
            invalid = deepcopy(config)
            if field is None:
                invalid[section][key] = replacement
            else:
                invalid[section][key][field] = replacement
            with self.subTest(section=section, key=key, field=field), self.assertRaises(ValueError):
                validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
