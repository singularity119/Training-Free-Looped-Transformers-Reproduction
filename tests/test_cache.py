import unittest

from tflt.cache import crop_cache, snapshot_cache


class FakeTensor:
    def __init__(self, length):
        self.length = int(length)
        self.shape = (2, 4, self.length, 8)

    def __getitem__(self, key):
        seq = key[-2]
        if isinstance(seq, slice):
            stop = self.length if seq.stop is None else int(seq.stop)
            return FakeTensor(stop)
        return self


class LayeredCache:
    def __init__(self, lengths):
        self.key_cache = [FakeTensor(length) for length in lengths]
        self.value_cache = [FakeTensor(length) for length in lengths]

    def get_seq_length(self, *args):
        return self.key_cache[0].shape[-2]

    def crop(self, length):
        self.key_cache = [item[..., :length, :] for item in self.key_cache]
        self.value_cache = [item[..., :length, :] for item in self.value_cache]


class CacheTest(unittest.TestCase):
    def test_crop_cache_restores_layer_specific_lengths_after_global_crop(self):
        cache = LayeredCache([2861, 2860, 2860])
        snap = snapshot_cache({"past_key_value": cache})

        cache.key_cache[1] = FakeTensor(2861)
        cache.value_cache[1] = FakeTensor(2861)
        crop_cache(snap)

        self.assertEqual([item.shape[-2] for item in cache.key_cache], [2861, 2860, 2860])
        self.assertEqual([item.shape[-2] for item in cache.value_cache], [2861, 2860, 2860])


if __name__ == "__main__":
    unittest.main()
