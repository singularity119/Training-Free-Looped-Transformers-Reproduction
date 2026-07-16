from __future__ import annotations

import copy
import math
import unittest

from tflt.loopscope.phase4_scalars import (
    stable_log_softmax_float32,
    trajectory_record_from_logits,
)
from tflt.loopscope.phase4_schema import Phase4ContractError, validate_trajectory_record


class Phase4ScalarTests(unittest.TestCase):
    def test_float32_stable_full_vocabulary_scalars_and_d36(self) -> None:
        logits = [
            [math.sin((boundary + 1) * (token + 1) * 0.01) for token in range(128)]
            for boundary in range(37)
        ]
        record = trajectory_record_from_logits(
            canonical_identity="synthetic",
            category="synthetic",
            boundary_logits=logits,
            producer_provenance={"synthetic": True},
        )
        self.assertEqual(len(record["boundaries"]), 37)
        self.assertLessEqual(abs(record["boundaries"][36]["kl_to_final"]), 1e-6)
        serialized_keys = set(record)
        self.assertTrue(serialized_keys.isdisjoint({"logits", "probabilities", "hidden_states"}))

    def test_extreme_logits_are_stable(self) -> None:
        values = stable_log_softmax_float32([10000.0, -10000.0] + [0.0] * 126)
        self.assertTrue(all(math.isfinite(float(value)) for value in values))

    def test_non_finite_logits_fail_fast(self) -> None:
        logits = [[0.0] * 128 for _ in range(37)]
        logits[12][2] = float("nan")
        with self.assertRaisesRegex(Phase4ContractError, "non-finite"):
            trajectory_record_from_logits(
                canonical_identity="bad",
                category="synthetic",
                boundary_logits=logits,
                producer_provenance={"synthetic": True},
            )

    def test_persisted_tensor_field_is_rejected(self) -> None:
        logits = [[float(token % 7) for token in range(128)] for _ in range(37)]
        record = trajectory_record_from_logits(
            canonical_identity="synthetic",
            category="synthetic",
            boundary_logits=logits,
            producer_provenance={"synthetic": True},
        )
        tampered = copy.deepcopy(record)
        tampered["logits"] = [0.0]
        with self.assertRaisesRegex(Phase4ContractError, "forbidden"):
            validate_trajectory_record(tampered, d36_tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()
