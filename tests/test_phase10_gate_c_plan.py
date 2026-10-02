"""Pure Python tests for frozen formal canaries and complete-population planning."""
import importlib.util
from pathlib import Path
import unittest

from tflt.loopscope.phase10_panel import load_config, score_manifest

_PATH = Path(__file__).resolve().parents[1] / "scripts/loopscope/prepare_phase10_gate_c.py"
_SPEC = importlib.util.spec_from_file_location("prepare_phase10_gate_c", _PATH)
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


class GateCPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.rows = [{"identity": f"canonical-test-{index}"} for index in range(14042)]
        cls.lengths = {spec["model"]: [{"context_length": 100 + (14041 - i) // divisor,
            "continuation_lengths": [1, 1, 1, 1], "prompt_token_length": 101 + (14041 - i) // divisor}
            for i in range(14042)] for divisor, spec in enumerate(cls.config["models"], start=2)}
        cls.plan, cls.artifacts = prepare.build_plan(
            cls.rows, cls.lengths, Path("/canonical/test_pool.json"), Path("/new/gate-c"), cls.config)

    def test_quantile_ranks_tie_break_and_canonical_order(self):
        items = [{"context_length": 100} for _ in range(14042)]
        selected, ranks = prepare.choose_indices(items)
        expected = [round(j * 14041 / 63) for j in range(64)]
        self.assertEqual(selected, expected)
        self.assertEqual(ranks, expected)
        items[123] = {"context_length": 999}
        selected, ranks = prepare.choose_indices(items)
        self.assertEqual(len(selected), 64)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertIn(123, selected)
        self.assertIn(0, selected)

    def test_selection_rejects_too_small_or_invalid_lengths(self):
        with self.assertRaises(ValueError):
            prepare.choose_indices([{"context_length": 1}] * 63)
        for value in (0, 1.5, True):
            items = [{"context_length": 1}] * 64
            items[0] = {"context_length": value}
            with self.assertRaises(ValueError):
                prepare.choose_indices(items)

    def test_complete_manifest_and_identity_population_remain_frozen(self):
        plan = self.plan
        self.assertEqual(self.artifacts["mmlu-manifest.json"], score_manifest(
            "mmlu", Path("/canonical/test_pool.json"), "FORMAL_TEST", self.config))
        self.assertEqual((plan["cell_count"], plan["new_cell_count"], plan["expected_reuse_cell_count"]),
                         (68, 59, 9))
        self.assertEqual(self.artifacts["full-identities.json"], [row["identity"] for row in self.rows])
        self.assertEqual(self.artifacts["full-indices.json"], list(range(14042)))
        self.assertEqual(len(plan["new_cells"]), 59)
        self.assertTrue(all(cell["full_count"] == 14042 and
            cell["full_identities_path"] == "/new/gate-c/full-identities.json" for cell in plan["new_cells"]))
        self.assertFalse(plan["target_gold_loaded"])
        self.assertFalse(plan["model_forward"])

    def test_canary_and_remainder_disjoint_cover_every_new_cell(self):
        self.assertEqual(len(self.plan["canaries"]), 7)
        manifest_cells = {cell["cell_id"]: cell for cell in self.artifacts["mmlu-manifest.json"]["cells"]}
        self.assertEqual(len({record["group_id"] for record in self.plan["canaries"]}), 7)
        for cell in self.plan["new_cells"]:
            remaining = self.artifacts[Path(cell["remaining_indices_path"]).name]
            self.assertEqual(len(remaining), cell["remaining_count"])
            if cell["canary_count"]:
                selected = self.artifacts[Path(cell["canary_indices_path"]).name]
                self.assertEqual(len(selected), 64)
                self.assertFalse(set(selected) & set(remaining))
                self.assertEqual(sorted(selected + remaining), list(range(14042)))
                self.assertEqual(manifest_cells[cell["cell_id"]]["strength"], 0.1)
                self.assertEqual(cell["canary_shard_id"], "formal-canary64")
            else:
                self.assertEqual(remaining, list(range(14042)))
        self.assertEqual(self.plan["canary_sample_records"], 448)
        self.assertEqual(self.plan["new_sample_records"], 828478)
        self.assertEqual(self.plan["remaining_sample_records"], 828030)
        self.assertEqual(self.plan["total_sample_records"], 954856)

    def test_all_groups_and_label_free_lengths_support_remaining_estimate(self):
        groups = self.plan["groups"]
        self.assertEqual(len(groups), 7)
        self.assertEqual(sum(group["new_cell_count"] for group in groups), 59)
        self.assertEqual(sum(group["unstarted_cell_count"] for group in groups), 52)
        self.assertEqual(sum(group["remaining_sample_records"] for group in groups), 828030)
        for model, entry in self.plan["models"].items():
            diagnostics = self.artifacts[Path(entry["lengths_path"]).name]
            self.assertEqual([row["canonical_index"] for row in diagnostics], list(range(14042)))
            self.assertEqual(sum(item["count"] for item in entry["distribution"]["context_token_counts"]), 14042)
            self.assertEqual(entry["distribution"]["native_context_differs_from_cached_prompt_count"], 14042)
            longest = max(self.lengths[model], key=lambda item: item["context_length"])["context_length"]
            self.assertEqual(max(diagnostics[i]["context_length"] for i in entry["canary_indices"]), longest)
            self.assertEqual(set(diagnostics[0]), {"canonical_index", "identity", "context_length",
                                                 "continuation_lengths", "prompt_token_length"})

    def test_partial_pool_and_duplicate_identity_cannot_be_frozen(self):
        with self.assertRaises(ValueError):
            prepare.build_plan(self.rows[:-1], self.lengths, Path("/pool"), Path("/output"), self.config)
        rows = list(self.rows)
        rows[0] = rows[1]
        with self.assertRaises(ValueError):
            prepare.build_plan(rows, self.lengths, Path("/pool"), Path("/output"), self.config)


if __name__ == "__main__":
    unittest.main()
