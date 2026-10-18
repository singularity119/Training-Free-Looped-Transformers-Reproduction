import copy
from contextlib import contextmanager
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tflt.loopscope.phase11_d8 import (MODEL, REVISION, SAMPLING, actual_seed,
    answer_span, selected_cells, prepare_pool, engineering_pool)


class Tokenizer:
    all_special_ids = [1, 2, 6]
    pad_token_id = 6
    pieces = {1: "<think>", 2: "</think>", 3: "Final answer: (B)\n", 4: "reasoning\n",
              5: "Final answer: (A)\n", 6: "<|im_end|>"}

    def convert_tokens_to_ids(self, text):
        return {"<think>": 1, "</think>": 2}[text]

    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [2]
        return [10, 11]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(t, "x") for t in ids if not skip_special_tokens or t not in (1, 2, 6))

    def apply_chat_template(self, messages, tokenize, **kwargs):
        self.messages = messages
        self.options = kwargs
        return [10, 11] if tokenize else "new native template"


class D8Tests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()

    def span(self, ids, prompt=()):
        return answer_span(ids, list(prompt), self.tok, [6])

    def test_thinking_answer_cannot_rescue_missing_close(self):
        s = self.span([1, 5, 4, 6])
        self.assertTrue(s["unclosed_thinking"])
        self.assertIsNone(s["prediction_index"])
        self.assertEqual(s["final_token_count"], 0)

    def test_first_close_excludes_temporary_answer_and_eos(self):
        s = self.span([1, 5, 4, 2, 3, 6])
        self.assertEqual(s["prediction_index"], 1)
        self.assertEqual(s["final_token_range"], [4, 5])
        self.assertEqual(s["thinking_token_range"], [1, 3])

    def test_prompt_opening_and_empty_final(self):
        s = self.span([4, 2, 6], prompt=[10, 1])
        self.assertEqual(s["prompt_think_open_count"], 1)
        self.assertTrue(s["final_empty"])
        self.assertIsNone(s["prediction_index"])

    def test_unchanged_s2_consistency_and_conflict(self):
        self.assertEqual(self.span([1, 4, 2, 3, 3, 6])["answer_mode"], "consistent_multiple")
        self.assertIsNone(self.span([1, 4, 2, 3, 5, 6])["prediction_index"])

    def test_multiple_boundaries_diagnosed_with_first_fixed(self):
        s = self.span([1, 4, 2, 3, 1, 2, 5, 6])
        self.assertTrue(s["anomalous_thinking_boundaries"])
        self.assertEqual(s["first_think_close_index"], 2)
        self.assertIsNone(s["prediction_index"])

    def test_cell_and_per_question_seed_independent_of_shards(self):
        cells = selected_cells()
        self.assertEqual(len(cells), 3)
        self.assertEqual(cells["native"]["window"], None)
        self.assertEqual(cells["current_t_k2_l05"]["k"], 2)
        self.assertEqual(cells["fixed_t0_k3_l09"]["strength"], .9)
        self.assertTrue(all(c["model"] == MODEL and c["revision"] == REVISION for c in cells.values()))
        self.assertEqual([actual_seed(i) for i in [8, 0, 7]], [20261026, 20261018, 20261025])
        self.assertEqual(actual_seed(8), actual_seed(list(range(9))[-1]))

    def test_prompt_rebuilt_from_question_and_frozen_options(self):
        row = {"identity": "synthetic:0", "task": "gpqa_main", "question": "Synthetic?",
            "choices": ["one", "two", "three", "four"], "choice_labels": list("ABCD"),
            "subject": "synthetic", "fewshot_sample_ids": [], "prompt": "old rendered prompt",
            "input_ids": [999], "prompt_token_length": 1}
        original = {"task": "gpqa_main", "target_gold_loaded": False, "engineering_synthetic": True,
                    "test_identities": [row["identity"]], "rows": [row]}
        before = copy.deepcopy(original)
        new = prepare_pool(original, self.tok)
        self.assertEqual(original, before)
        self.assertEqual(new["rows"][0]["choices"], row["choices"])
        self.assertNotIn("old rendered prompt", self.tok.messages[0]["content"])
        self.assertEqual([m["role"] for m in self.tok.messages], ["user"])
        self.assertTrue(self.tok.options["enable_thinking"])
        self.assertEqual(new["sampling"], SAMPLING)

    def test_legacy_two_option_engineering_fixture_gets_own_four_options(self):
        old = {"scope": "PREFLIGHT_ONLY", "engineering_synthetic": True,
               "rows": [{"choices": ["old one", "old two"], "choice_labels": ["A", "B"]}]}
        before = copy.deepcopy(old)
        new = engineering_pool(old)
        self.assertEqual(old, before)
        self.assertEqual(new["rows"][0]["choice_labels"], list("ABCD"))
        self.assertEqual(len(new["rows"][0]["choices"]), 4)
        with self.assertRaises(ValueError):
            engineering_pool({"scope": "FORMAL_TEST", "engineering_synthetic": False})

    def test_actual_generation_options_and_all_rng_resets(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/loopscope"))
        import phase11_d8 as driver
        import run_phase11_accuracy as runner
        fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        torch = SimpleNamespace(long="long", tensor=lambda *a, **k: object(), ones_like=lambda x: object(),
            manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock()))
        state = random.getstate()
        try:
            with patch.dict(sys.modules, {"numpy": fake_numpy}):
                driver.seed_compute(torch, actual_seed(17))
                first = random.random()
                driver.seed_compute(torch, actual_seed(17))
                self.assertEqual(first, random.random())
            fake_numpy.random.seed.assert_called_with(20261035)
            torch.manual_seed.assert_called_with(20261035)
            torch.cuda.manual_seed_all.assert_called_with(20261035)
        finally:
            random.setstate(state)

        @contextmanager
        def lifecycle(*args, **kwargs):
            yield {"prefill_count": 1, "decode_count": 4, "forwards": [{"cache_length_after": 6}]}

        model = SimpleNamespace(device="cuda", generate=Mock(return_value=[SimpleNamespace(tolist=lambda: [10, 11, 1, 4, 2, 3, 6])]))
        row = {"identity": "synthetic:0", "task": "gpqa_main", "subject": "synthetic",
               "choice_labels": list("ABCD"), "prompt": "new native template", "input_ids": [10, 11]}
        with patch("tflt.loopscope.phase11_runtime.make_runtime", return_value=None), \
             patch("tflt.loopscope.phase11_adapter.generation_forward_context", lifecycle):
            runner.acquire_row(torch, model, self.tok, None, [6], row, 0, selected_cells()["native"],
                {"max_length": 40960}, "PREFLIGHT_ONLY", 32768, None,
                generation_options=SAMPLING, preserve_special_tokens=True)
        options = model.generate.call_args.kwargs
        self.assertTrue(all(options[k] == v for k, v in SAMPLING.items()))
        self.assertEqual(options["max_new_tokens"], 32768)
        self.assertTrue(options["use_cache"])
        with patch("tflt.loopscope.phase11_runtime.make_runtime", return_value=None), \
             patch("tflt.loopscope.phase11_adapter.generation_forward_context", lifecycle):
            runner.acquire_row(torch, model, self.tok, None, [6], row, 0, selected_cells()["native"],
                {"max_length": 40960}, "PREFLIGHT_ONLY", 32768, None)
        old_options = model.generate.call_args.kwargs
        self.assertFalse(old_options["do_sample"])
        self.assertNotIn("temperature", old_options)


if __name__ == "__main__":
    unittest.main()
