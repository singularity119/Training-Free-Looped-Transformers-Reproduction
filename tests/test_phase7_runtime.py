import types
import unittest

from tflt.loopscope.phase7_runtime import (
    assemble_raw_boundaries,
    assert_no_loop_wrapper,
    module_path,
    register_raw_final_norm_hook,
    resolve_architecture,
    select_position_vector,
)
from tflt.loopscope.phase7_schema import Phase7ContractError


class FakeState:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        if self.ndim != 3 or not isinstance(key, tuple) or len(key) != 3:
            raise AssertionError("unexpected fake tensor slice")
        batch, position, hidden = key
        del batch, hidden
        if isinstance(position, int):
            return FakeState((self.shape[0], self.shape[2]))
        return FakeState((self.shape[0], 1, self.shape[2]))


class HookHandle:
    def __init__(self, callbacks, callback):
        self.callbacks = callbacks
        self.callback = callback

    def remove(self):
        if self.callback in self.callbacks:
            self.callbacks.remove(self.callback)


class FakeFinalNorm:
    def __init__(self):
        self.pre_hooks = []
        self.forward_hooks = []

    def register_forward_pre_hook(self, callback):
        self.pre_hooks.append(callback)
        return HookHandle(self.pre_hooks, callback)

    def register_forward_hook(self, callback):
        self.forward_hooks.append(callback)
        return HookHandle(self.forward_hooks, callback)

    def invoke(self, raw, normalized):
        for callback in tuple(self.pre_hooks):
            callback(self, (raw,))
        for callback in tuple(self.forward_hooks):
            callback(self, (raw,), normalized)


class FakeModule:
    pass


class FakeModel:
    def __init__(self, model_type, layer_count=4):
        self.config = types.SimpleNamespace(model_type=model_type, architectures=(), num_hidden_layers=layer_count)
        self.model = types.SimpleNamespace(
            layers=[FakeModule() for _ in range(layer_count)],
            embed_tokens=FakeModule(),
            norm=FakeFinalNorm(),
        )
        self.lm_head = FakeModule()

    def named_modules(self):
        yield "", self
        yield "model", self.model
        yield "model.norm", self.model.norm
        yield "lm_head", self.lm_head
        for index, layer in enumerate(self.model.layers):
            yield "model.layers.%d" % index, layer


class Phase7RuntimeTests(unittest.TestCase):
    def test_three_frozen_architecture_families_resolve_same_narrow_contract(self):
        for model_type, model_key in (
            ("qwen2", "qwen25_3b"),
            ("llama", "llama32_3b"),
            ("gemma2", "gemma2_2b"),
        ):
            model = FakeModel(model_type)
            adapter = resolve_architecture(model, model_key=model_key)
            self.assertEqual(adapter.decoder_blocks_path, "model.layers")
            self.assertEqual(adapter.embedding_path, "model.embed_tokens")
            self.assertEqual(adapter.final_norm_path, "model.norm")
            self.assertEqual(adapter.lm_head_path, "lm_head")
            self.assertEqual(module_path(model, model.model.norm), "model.norm")
            assert_no_loop_wrapper(model)

    def test_raw_boundaries_replace_post_norm_endpoint_and_probe_position_is_exact(self):
        hidden_states = [FakeState((1, 3, 8)) for _ in range(6)]
        captured = FakeState((1, 3, 8))
        boundaries = assemble_raw_boundaries(hidden_states, captured, layer_count=4, pre_hook_count=1)
        self.assertEqual(len(boundaries), 5)
        self.assertIs(boundaries[-1], captured)
        probe = select_position_vector(boundaries[0], 2)
        self.assertEqual(probe.shape, (1, 8))
        with self.assertRaises(Phase7ContractError):
            select_position_vector(boundaries[0], 3)

    def test_final_norm_capture_is_exactly_once_and_handle_is_removed(self):
        final_norm = FakeFinalNorm()
        capture, handle = register_raw_final_norm_hook(final_norm)
        raw = object()
        normalized = object()
        final_norm.invoke(raw, normalized)
        self.assertEqual(capture["count"], 1)
        self.assertIs(capture["raw"], raw)
        self.assertIs(capture["normalized"], normalized)
        handle.remove()
        self.assertEqual(final_norm.pre_hooks, [])
        self.assertEqual(final_norm.forward_hooks, [])


if __name__ == "__main__":
    unittest.main()
