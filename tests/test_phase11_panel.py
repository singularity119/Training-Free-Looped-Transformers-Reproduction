import copy
import unittest

from tflt.loopscope.phase11_panel import DATASETS, build_all_panels, build_panel, load_config, validate_config, validate_score_manifest, score_manifest


class PanelTests(unittest.TestCase):
    def test_54_unique_cells_and_k2_aliases(self):
        all_panels = build_all_panels()
        self.assertEqual(all_panels["independent_cell_count"], 54)
        self.assertEqual(all_panels["configuration_sample_records"], 245736)
        self.assertEqual(all_panels["generation_sample_records"], 224640)
        ids = []
        for dataset in DATASETS:
            panel = build_panel(dataset)
            self.assertEqual(len(panel["cells"]), 18)
            self.assertEqual(sum(c["arm"] == "Online" for c in panel["cells"]), 15)
            aliases = [c for c in panel["display_rows"] if c["alias_of"]]
            self.assertEqual(len(aliases), 3)
            self.assertTrue(all(c["k"] == 2 and c["direction_policy"] == "lag1" for c in aliases))
            self.assertTrue(all("fixed-t0" in c["result_cell_id"] for c in aliases))
            ids.extend(c["cell_id"] for c in panel["cells"])
        self.assertEqual(len(set(ids)), 54)

    def test_full_decode_and_frozen_bank_are_required(self):
        for key, value in (("decode_mode", "bypass"), ("cache_strategy", "last"),
                           ("direction_fit_scope", "rolling_decode"), ("max_new_tokens", 1024)):
            config = copy.deepcopy(load_config())
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_exact_gpqa_author_source_binding(self):
        config = copy.deepcopy(load_config())
        self.assertEqual(config["datasets"]["gpqa_main"]["source_kind"], "author_github_archive")
        self.assertEqual(config["datasets"]["gpqa_main"]["split"], "main_csv")
        for key, value in (("revision", "a" * 40), ("source_kind", "huggingface"), ("split", "train")):
            changed = copy.deepcopy(config)
            changed["datasets"]["gpqa_main"][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_manifest_refuses_added_or_missing_configuration(self):
        manifest = score_manifest("mmlu_pro")
        validate_score_manifest(manifest)
        manifest["cells"].pop()
        with self.assertRaises(ValueError):
            validate_score_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
