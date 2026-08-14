import unittest
from types import SimpleNamespace

import torch

from tflt.loopscope.phase7_producer import (
    _check_single_batch,
    _native_final_hidden_and_logits,
    _project_causal_lm_logits,
    admit_choice_surfaces,
)
from tflt.loopscope.phase7_schema import CHOICE_SURFACES, Phase7ContractError
from tflt.loopscope.phase7_runtime import Phase7RuntimeError


class FakeTokenizer:
    def __init__(self, values):
        self.values = dict(values)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self.values[text]


class FakeInputIds:
    def __init__(self, shape):
        self.shape = tuple(shape)


class Phase7ProducerTests(unittest.TestCase):
    def test_choice_admission_keeps_ids_ephemeral_and_rejects_non_single_surfaces(self):
        tokenizer = FakeTokenizer({surface: [index + 1] for index, surface in enumerate(CHOICE_SURFACES)})
        summary, ephemeral_ids = admit_choice_surfaces(tokenizer)
        self.assertTrue(summary["all_single_token"])
        self.assertTrue(summary["all_distinct"])
        self.assertEqual(len(ephemeral_ids), 4)
        self.assertNotIn("token_ids", summary)

        multi = FakeTokenizer({surface: [1, 2] for surface in CHOICE_SURFACES})
        with self.assertRaisesRegex(Phase7ContractError, "BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN"):
            admit_choice_surfaces(multi)

    def test_renderer_batch_and_sequence_length_are_checked_before_forward(self):
        _check_single_batch({"input_ids": FakeInputIds((1, 8))}, 8)
        with self.assertRaises(Phase7RuntimeError):
            _check_single_batch({"input_ids": FakeInputIds((2, 8))}, 8)
        with self.assertRaises(Phase7RuntimeError):
            _check_single_batch({"input_ids": FakeInputIds((1, 9))}, 8)

    def test_causal_lm_closure_uses_hook_output_without_last_hidden_state(self):
        native_final = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        weight = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        class ShapeSensitiveHead:
            def __call__(self, hidden):
                projected = hidden @ weight
                if hidden.ndim == 2:
                    projected = projected + 1.0
                return projected

        head = ShapeSensitiveHead()
        native_logits = head(native_final)
        outputs = SimpleNamespace(logits=native_logits)
        model = SimpleNamespace(config=SimpleNamespace(final_logit_softcapping=None))

        final_hidden, final_logits = _native_final_hidden_and_logits(
            torch,
            model,
            head,
            outputs,
            {"normalized": native_final},
            1,
        )
        self.assertFalse(hasattr(outputs, "last_hidden_state"))
        self.assertTrue(torch.equal(final_hidden, native_final[:, 1, :]))
        self.assertTrue(torch.equal(final_logits, native_logits[:, 1, :]))

    def test_gemma2_softcap_is_part_of_native_logits_projection(self):
        hidden = torch.tensor([[3.0, -2.0]])
        head = lambda value: value
        model = SimpleNamespace(config=SimpleNamespace(final_logit_softcapping=2.0))
        expected = torch.tanh(hidden / 2.0) * 2.0
        self.assertTrue(torch.equal(_project_causal_lm_logits(torch, model, head, hidden), expected))


if __name__ == "__main__":
    unittest.main()
