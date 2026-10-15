"""D6 prompt-only derivation and scope-only intervention on the cached path."""
import copy
import importlib.util
from pathlib import Path
import unittest

from tflt.loopscope.phase11_adapter import generation_forward_context
from tflt.loopscope.phase11_data import (
    DIRECT_INSTRUCTION, INSTRUCTION, derive_gpqa_prompt_pool, prepare_gpqa_prompt_row,
    render_generation_prompt,
)
from tflt.loopscope.phase11_runtime import Phase11Runtime, make_runtime, validate_calls
from test_phase11_adapter import Ids, Model
from test_phase11_runtime import FakeRuntime


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("d6_accuracy_entry", REPO / "scripts/loopscope/run_phase11_accuracy.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("rendered chat must not receive extra special tokens")
        return [ord(char) for char in text]


class ScopedFakeRuntime(FakeRuntime):
    def __init__(self, policy, k, strength, damping_scope):
        Phase11Runtime.__init__(self, policy, k, strength, damping_scope=damping_scope)
        self.fits = []
        self.outputs = []

    def __call__(self, delta, t):
        result = super().__call__(delta, t)
        self.outputs.append(copy.deepcopy(result.data))
        return result


class PromptTests(unittest.TestCase):
    def row(self):
        content = render_generation_prompt({"question": "Synthetic question", "options": ["first", "second"],
                                            "category": "gpqa_main"})
        prompt = "<user>" + content + "</user><assistant>"
        ids = Tokenizer().encode(prompt, add_special_tokens=False)
        return {"identity": "synthetic:0", "sample_id": "synthetic:0", "task": "gpqa_main",
                "source_index": 0, "choices": ["first", "second"], "choice_labels": ["A", "B"],
                "fewshot_sample_ids": [], "question": "Synthetic question", "prompt": prompt,
                "input_ids": ids, "prompt_token_length": len(ids)}

    def test_cot_is_unchanged_and_direct_changes_only_exact_instruction_and_tokens(self):
        row = self.row()
        original = copy.deepcopy(row)
        cot = prepare_gpqa_prompt_row(row, Tokenizer())
        direct = prepare_gpqa_prompt_row(row, Tokenizer(), "direct")
        self.assertEqual(row, original)
        self.assertEqual(cot, original)
        self.assertEqual(direct["prompt"], original["prompt"].replace(INSTRUCTION, DIRECT_INSTRUCTION))
        self.assertEqual(direct["prompt"].replace(DIRECT_INSTRUCTION, INSTRUCTION), original["prompt"])
        self.assertIn("Answer: Let's think step by step.", direct["prompt"])
        self.assertTrue(direct["prompt"].startswith("<user>"))
        self.assertTrue(direct["prompt"].endswith("</user><assistant>"))
        token_fields = {"prompt", "input_ids", "prompt_token_length"}
        self.assertEqual({k: v for k, v in direct.items() if k not in token_fields},
                         {k: v for k, v in original.items() if k not in token_fields})
        self.assertEqual(direct["input_ids"], Tokenizer().encode(direct["prompt"], add_special_tokens=False))

    def test_pool_derivation_retains_identity_order_and_source_and_rejects_prompt_drift(self):
        row = self.row()
        pool = {"task": "gpqa_main", "target_gold_loaded": False, "rows": [row],
                "test_identities": [row["identity"]], "dataset_revision": "frozen-source"}
        direct = derive_gpqa_prompt_pool(pool, Tokenizer(), "direct")
        self.assertEqual(direct["test_identities"], pool["test_identities"])
        self.assertEqual(direct["dataset_revision"], pool["dataset_revision"])
        self.assertEqual(pool["rows"][0], row)
        for changed in ({"prompt": row["prompt"].replace(INSTRUCTION, "drift")}, {"input_ids": [1, 2]}):
            with self.assertRaisesRegex(ValueError, "instruction|tokenizer"):
                prepare_gpqa_prompt_row({**row, **changed}, Tokenizer(), "direct")
        with self.assertRaisesRegex(ValueError, "context"):
            prepare_gpqa_prompt_row(row, Tokenizer(), "direct", max_length=8193)


class ScopeTests(unittest.TestCase):
    def execute(self, policy, k, strength, scope):
        runtime = ScopedFakeRuntime(policy, k, strength, scope)
        model = Model(runtime)
        tokens = (0, 11, 12, 99)
        with generation_forward_context(model, runtime, "synthetic", tokens,
                                        special_ids=(99,), pad_token_id=0) as report:
            model.generate(Ids([list(tokens)]))
        return runtime, model, report, RUNNER.runtime_telemetry(runtime)

    def test_scope_preserves_prefill_k_loop_and_cache_but_disables_decode_damping(self):
        for policy, k, strength in (("current_t", 2, .5), ("fixed_t0", 3, .9)):
            full, full_model, full_report, full_stats = self.execute(policy, k, strength, "full")
            prefill, prefill_model, prefill_report, prefill_stats = self.execute(policy, k, strength, "prefill_only")
            self.assertEqual(full.outputs[:k], prefill.outputs[:k])
            self.assertEqual(full.fits, prefill.fits)
            self.assertEqual(full_report, prefill_report)
            self.assertEqual(full_model.seen, [4, 1, 1])
            self.assertEqual(prefill_model.seen, [4, 1, 1])
            self.assertEqual(validate_calls(prefill), 3)
            self.assertEqual([c["timesteps"] for c in prefill.context_summaries], [list(range(k))] * 3)
            self.assertTrue(all(not c["applied"] and c["same_object"] and
                                c["answer_max_abs_change"] == 0 and c["direction_fit_t"] is None
                                for c in prefill.calls[k:]))
            self.assertTrue(any(c["applied"] for c in full.calls[k:]))
            self.assertEqual(prefill.directions, {})
            for stats in (full_stats, prefill_stats):
                self.assertEqual(stats["prefill_callback_count"], k)
                self.assertEqual(stats["decode_callback_count"], 2 * k)
                self.assertEqual(stats["decode_fit_count"], 0)
                self.assertTrue(stats["directions_released"])
            self.assertEqual(full_stats["prefill_applied_by_t"], prefill_stats["prefill_applied_by_t"])
            self.assertEqual(prefill_stats["decode_applied_by_t"], {str(t): 0 for t in range(k)})
            self.assertEqual(prefill_stats["damping_scope"], "prefill_only")

    def test_default_runtime_full_and_loop_config_keep_full_decode_first_cache(self):
        cell = {"arm": "Online", "direction_policy": "current_t", "k": 2, "strength": .5,
                "cache_strategy": "first", "decode_mode": "full", "window": [15, 18],
                "model": "Qwen/Qwen3-4B-Instruct-2507"}
        self.assertEqual(make_runtime(cell).damping_scope, "full")
        scoped = make_runtime({**cell, "damping_scope": "prefill_only"})
        self.assertEqual(scoped.damping_scope, "prefill_only")
        config = RUNNER.make_loop_config(cell, scoped)
        self.assertEqual((config.k, config.cache_strategy, config.decode_mode), (2, "first", "full"))
        self.assertIs(config.residual_transform, scoped)


if __name__ == "__main__":
    unittest.main()
