"""Pure Python checks for the separate Thinking-2507 scientific recipe."""
import copy
import importlib.util
from pathlib import Path
import unittest

from tflt.loopscope import phase11_thinking2507 as recipe

_SPEC = importlib.util.spec_from_file_location(
    "thinking2507_contract", Path(__file__).resolve().parents[1] /
    "scripts/loopscope/thinking2507_contract.py")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


class Tokenizer:
    pieces = {1: "<think>", 2: "</think>", 3: "Final answer: (B)\n",
              4: "reasoning\n", 5: "Final answer: (A)\n", 6: "<|im_end|>"}

    def convert_tokens_to_ids(self, text):
        return {"<think>": 1, "</think>": 2}[text]

    def encode(self, text, add_special_tokens=False):
        return [2] if text == "</think>" else [10, 11]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(i, "x") for i in ids
                       if not skip_special_tokens or i not in (1, 2, 6))

    def apply_chat_template(self, messages, tokenize, **options):
        return [10, 11] if tokenize else "native thinking template"


class Thinking2507Tests(unittest.TestCase):
    def span(self, ids):
        return recipe.answer_span(ids, [10, 1], Tokenizer(), [6])

    def test_cross_boundary_conflict_fails(self):
        parsed = self.span([1, 5, 4, 2, 3, 6])
        self.assertEqual(parsed["answer_mode"], "conflicting_multiple")
        self.assertIsNone(parsed["prediction_index"])
        self.assertEqual(parsed["final_text"], "Final answer: (B)\n")
        self.assertEqual(parsed["parser"], "D4-whole-output-GPQA-S2-v1")
        self.assertEqual(parsed["scoring_region"], "whole_generated_output")

    def test_repeated_same_letter_across_boundary_passes(self):
        parsed = self.span([1, 3, 4, 2, 3, 6])
        self.assertEqual(parsed["answer_mode"], "consistent_multiple")
        self.assertEqual(parsed["prediction_index"], 1)
        self.assertEqual(parsed["valid_final_line_count"], 2)

    def test_unclosed_thinking_answer_is_eligible(self):
        parsed = self.span([1, 4, 5, 6])
        self.assertTrue(parsed["unclosed_thinking"])
        self.assertTrue(parsed["final_empty"])
        self.assertEqual(parsed["final_token_count"], 0)
        self.assertEqual(parsed["predicted_label"], "A")

    def test_frozen_model_greedy_and_context(self):
        self.assertEqual(recipe.MODEL, "Qwen/Qwen3-4B-Thinking-2507")
        self.assertEqual(recipe.REVISION, "UNVERIFIED_OSS_COPY")
        self.assertEqual(recipe.MAX_NEW_TOKENS, 8192)
        self.assertEqual(recipe.MAX_LENGTH, 262144)
        self.assertEqual(recipe.SAMPLING,
                         {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0})
        cells = recipe.selected_cells()
        self.assertEqual(len(cells), 3)
        self.assertIsNone(cells["native"]["window"])
        self.assertEqual(cells["current_t_k2_l05"]["k"], 2)
        self.assertEqual(cells["current_t_k2_l05"]["strength"], .5)
        self.assertEqual(cells["fixed_t0_k3_l09"]["k"], 3)
        self.assertEqual(cells["fixed_t0_k3_l09"]["strength"], .9)
        for cell in cells.values():
            self.assertEqual(cell["model"], recipe.MODEL)
            self.assertEqual(cell["revision"], recipe.REVISION)
        self.assertEqual(recipe.actual_seed(447), 20261018+447)

    def test_prepare_pool_preserves_options_and_new_bindings(self):
        row = {"identity": "synthetic:0", "task": "gpqa_main", "question": "Synthetic?",
               "choices": ["a", "b", "c", "d"], "choice_labels": list("ABCD"),
               "subject": "synthetic", "fewshot_sample_ids": []}
        original = {"task": "gpqa_main", "target_gold_loaded": False,
                    "engineering_synthetic": True, "test_identities": [row["identity"]], "rows": [row]}
        before = copy.deepcopy(original)
        pool = recipe.prepare_pool(original, Tokenizer())
        self.assertEqual(before, original)
        self.assertEqual(pool["rows"][0]["choices"], row["choices"])
        self.assertEqual(pool["max_new_tokens"], 8192)
        self.assertEqual(pool["max_length"], 262144)
        self.assertEqual(pool["sampling"], recipe.SAMPLING)
        pool["sampling"] = {"do_sample": True}
        with self.assertRaises(ValueError):
            recipe.validate_pool(pool)

    def test_metadata_distinguishes_new_recipe(self):
        cell = recipe.selected_cells()["native"]
        job = dict(cell_key="native", cell_id=cell["cell_id"], seed=recipe.SEED,
                   source_commit="test", prompt_mode="thinking", damping_scope="none",
                   pool="pool", run_root="run", indices=[0], synthetic=True, engineering=True)
        meta = contract.metadata(job, cell, {"think_open_id": 1, "think_close_id": 2})
        self.assertEqual(meta["schema"], "loopscope.phase11.thinking2507.job.v1")
        self.assertEqual(meta["revision"], "UNVERIFIED_OSS_COPY")
        self.assertEqual(meta["max_new_tokens"], 8192)
        self.assertEqual(meta["sampling"], recipe.SAMPLING)
        self.assertEqual(meta["scoring_region"], "whole_generated_output")
        with self.assertRaises(ValueError):
            contract.validate_bundle(0, {}, {}, dict(meta, max_new_tokens=32768), {}, Tokenizer())


if __name__ == "__main__":
    unittest.main()
