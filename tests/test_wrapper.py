import unittest

from tflt.config import LoopConfig
from tflt.wrapper import apply_loop_wrapper, looped_model


class AddLayer:
    def __init__(self, delta):
        self.delta = delta
        self.calls = 0

    def __call__(self, hidden_states, *args, **kwargs):
        self.calls += 1
        return (hidden_states + self.delta,)


class DummyInner:
    def __init__(self):
        self.layers = [AddLayer(1.0), AddLayer(2.0), AddLayer(4.0)]


class DummyModel:
    def __init__(self):
        self.model = DummyInner()

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self, *args):
        return self.length

    def crop(self, length):
        self.length = length


class CacheLayer(AddLayer):
    def __call__(self, hidden_states, *args, **kwargs):
        cache = kwargs.get("past_key_value")
        if kwargs.get("use_cache") and cache is not None:
            cache.length += 1
        return super().__call__(hidden_states, *args, **kwargs)


class CacheModel:
    def __init__(self):
        self.layers = [CacheLayer(1.0)]


class WrapperTest(unittest.TestCase):
    def test_layer_mode_patch_and_restore(self):
        model = DummyModel()
        originals = list(model.model.layers)
        cfg = LoopConfig("dummy", (1, 1), k=2, iteration_mode="layer", strategy="naive")
        handle = apply_loop_wrapper(model, cfg)
        self.assertNotEqual(model.model.layers[1], originals[1])
        self.assertEqual(model.forward(0.0), 9.0)
        handle.restore()
        self.assertEqual(model.model.layers, originals)
        self.assertEqual(model.forward(0.0), 7.0)

    def test_block_mode_uses_identity_followers(self):
        model = DummyModel()
        cfg = LoopConfig("dummy", (0, 1), k=2, iteration_mode="block", strategy="naive")
        with looped_model(model, cfg):
            self.assertEqual(model.forward(0.0), 10.0)
        self.assertEqual(model.forward(0.0), 7.0)

    def test_decode_full_crops_body_cache_and_stashes_once(self):
        model = CacheModel()
        cache = FakeCache(5)
        cfg = LoopConfig(
            "dummy",
            (0, 0),
            k=3,
            iteration_mode="layer",
            strategy="naive",
            decode_mode="full",
        )
        with looped_model(model, cfg):
            out = model.layers[0](0.0, use_cache=True, past_key_value=cache)
        self.assertEqual(out[0], 3.0)
        self.assertEqual(cache.length, 6)


if __name__ == "__main__":
    unittest.main()
