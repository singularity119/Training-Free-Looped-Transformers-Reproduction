"""Pure Python checks for the separate Qwen3 nonthinking scientific recipe."""
import copy
import importlib.util
from pathlib import Path
import unittest

from tflt.loopscope import phase11_qwen3_nothink as recipe

_SPEC = importlib.util.spec_from_file_location(
    "qwen3_nothink_contract", Path(__file__).resolve().parents[1] /
    "scripts/loopscope/qwen3_nothink_contract.py")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


class Tokenizer:
    pieces = {1: "<think>", 2: "</think>", 3: "Final answer: (B)\n",
              4: "reasoning\n", 5: "Final answer: (A)\n", 6: "<|im_end|>", 7: "\n\n"}

    def convert_tokens_to_ids(self, text):
        return {"<think>": 1, "</think>": 2}[text]

    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [2]
        if text == "<think>":
            return [1]
        if text == recipe.EMPTY_THINKING_SPAN:
            return [1, 7, 2, 7]
        return [10, 1, 7, 2, 7]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(i, "x") for i in ids
                       if not skip_special_tokens or i not in (1, 2, 6))

    def apply_chat_template(self, messages, tokenize, **options):
        self.options = options
        return [10, 1, 7, 2, 7] if tokenize else "native template" + recipe.EMPTY_THINKING_SPAN


class Qwen3NothinkTests(unittest.TestCase):
    def span(self, ids):
        return recipe.answer_span(ids, [10, 1], Tokenizer(), [6])

    def test_native_cot_without_think_is_complete_final(self):
        parsed = self.span([4, 3, 6])
        self.assertFalse(parsed["unclosed_thinking"])
        self.assertFalse(parsed["final_empty"])
        self.assertEqual(parsed["thinking_token_count"], 0)
        self.assertEqual(parsed["final_token_range"], [0, 2])
        self.assertEqual(parsed["final_text"], "reasoning\nFinal answer: (B)\n")
        self.assertEqual(parsed["prediction_index"], 1)

    def test_cross_boundary_conflict_fails(self):
        parsed = self.span([1, 5, 4, 2, 3, 6])
        self.assertEqual(parsed["answer_mode"], "conflicting_multiple")
        self.assertIsNone(parsed["prediction_index"])
        self.assertEqual(parsed["final_text"], "Final answer: (A)\nreasoning\nFinal answer: (B)\n")
        self.assertEqual(parsed["parser"], "D12-nonthinking-whole-final-GPQA-S2-v1")
        self.assertEqual(parsed["scoring_region"], "whole_generated_output_as_final")

    def test_repeated_same_letter_across_boundary_passes(self):
        parsed = self.span([1, 3, 4, 2, 3, 6])
        self.assertEqual(parsed["answer_mode"], "consistent_multiple")
        self.assertEqual(parsed["prediction_index"], 1)
        self.assertEqual(parsed["valid_final_line_count"], 2)

    def test_unclosed_thinking_answer_is_eligible(self):
        parsed = self.span([1, 4, 5, 6])
        self.assertTrue(parsed["unclosed_thinking"])
        self.assertFalse(parsed["final_empty"])
        self.assertEqual(parsed["final_token_count"], 3)
        self.assertEqual(parsed["predicted_label"], "A")

    def test_frozen_model_greedy_and_context(self):
        self.assertEqual(recipe.MODEL, "Qwen/Qwen3-4B")
        self.assertEqual(recipe.REVISION, "UNVERIFIED_OSS_COPY")
        self.assertEqual(recipe.MAX_NEW_TOKENS, 32768)
        self.assertEqual(recipe.MAX_LENGTH, 40960)
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
        tokenizer = Tokenizer()
        pool = recipe.prepare_pool(original, tokenizer)
        self.assertIs(tokenizer.options["enable_thinking"], False)
        self.assertTrue(pool["rows"][0]["prompt"].endswith(recipe.EMPTY_THINKING_SPAN))
        self.assertEqual(pool["rows"][0]["input_ids"][-4:], [1, 7, 2, 7])
        self.assertEqual(pool["prompt_mode"], "cot")
        self.assertIs(pool["enable_thinking"], False)
        self.assertEqual(before, original)
        self.assertEqual(pool["rows"][0]["choices"], row["choices"])
        self.assertEqual(pool["max_new_tokens"], 32768)
        self.assertEqual(pool["max_length"], 40960)
        self.assertEqual(pool["sampling"], recipe.SAMPLING)
        pool["sampling"] = {"do_sample": True}
        with self.assertRaises(ValueError):
            recipe.validate_pool(pool)

    def test_prepare_rejects_nonempty_or_missing_closed_span(self):
        row = {"identity": "synthetic:0", "task": "gpqa_main", "question": "Synthetic?",
               "choices": ["a", "b", "c", "d"], "choice_labels": list("ABCD"),
               "subject": "synthetic", "fewshot_sample_ids": []}
        original = {"task": "gpqa_main", "target_gold_loaded": False,
                    "engineering_synthetic": True, "test_identities": [row["identity"]], "rows": [row]}
        class MissingSpan(Tokenizer):
            def apply_chat_template(self, messages, tokenize, **options):
                return [10, 1, 7, 2, 7] if tokenize else "native template"
        with self.assertRaises(ValueError):
            recipe.prepare_pool(original, MissingSpan())
        pool = recipe.prepare_pool(original, Tokenizer())
        pool["rows"][0]["input_ids"] = [10] * (recipe.MAX_LENGTH - recipe.MAX_NEW_TOKENS + 1)
        pool["rows"][0]["prompt_token_length"] = len(pool["rows"][0]["input_ids"])
        with self.assertRaises(ValueError):
            recipe.validate_pool(pool)

    def test_metadata_distinguishes_new_recipe(self):
        cell = recipe.selected_cells()["native"]
        job = dict(cell_key="native", cell_id=cell["cell_id"], seed=recipe.SEED,
                   source_commit="test", prompt_mode="cot", damping_scope="none",
                   pool="pool", run_root="run", indices=[0], synthetic=True, engineering=True)
        meta = contract.metadata(job, cell, {"think_open_id": 1, "think_close_id": 2})
        self.assertEqual(meta["schema"], "loopscope.phase11.qwen3_nothink.job.v1")
        self.assertEqual(meta["revision"], "UNVERIFIED_OSS_COPY")
        self.assertEqual(meta["max_new_tokens"], 32768)
        self.assertEqual(meta["sampling"], recipe.SAMPLING)
        self.assertEqual(meta["scoring_region"], "whole_generated_output_as_final")
        with self.assertRaises(ValueError):
            contract.validate_bundle(0, {}, {}, dict(meta, max_new_tokens=1), {}, Tokenizer())


if __name__ == "__main__":
    unittest.main()
