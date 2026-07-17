from __future__ import annotations

import copy
import unittest

import numpy as np

from tflt.loopscope.phase4_hidden_geometry import (
    B2_RECORD_SCHEMA,
    COSINE_KEY,
    L2_KEY,
    Phase4HiddenGeometryError,
    analyze_hidden_geometry_records,
    geometry_scalars_from_numpy,
    strip_geometry_to_p4b,
    validate_hidden_geometry_record,
)
from tflt.loopscope.phase4_schema import validate_trajectory_record
from tflt.loopscope.phase4_verifier import _synthetic_scalar_record


def _geometry_record(index: int = 0):
    record = copy.deepcopy(_synthetic_scalar_record(index))
    vectors = np.zeros((37, 8), dtype=np.float32)
    for boundary in range(37):
        vectors[boundary, 0] = 1.0
        vectors[boundary, 1] = float(boundary + 1) / 37.0
        vectors[boundary, 2] = float((boundary + index) % 7) / 11.0
    geometry = geometry_scalars_from_numpy(vectors)
    record["schema_version"] = B2_RECORD_SCHEMA
    record["producer_provenance"].update(
        {
            "hidden_geometry_reduction_dtype": "float32",
            "final_norm_application": "exactly_once_per_boundary",
        }
    )
    for boundary, extension in zip(record["boundaries"], geometry):
        boundary.update(extension)
    validate_hidden_geometry_record(record)
    return record


class Phase4HiddenGeometryTests(unittest.TestCase):
    def test_exact_float32_geometry_and_endpoint(self) -> None:
        vectors = np.arange(37 * 4, dtype=np.float64).reshape(37, 4) + 1.0
        rows = geometry_scalars_from_numpy(vectors)
        self.assertEqual(len(rows), 37)
        self.assertLessEqual(abs(rows[36][L2_KEY]), 1e-7)
        self.assertLessEqual(abs(rows[36][COSINE_KEY] - 1.0), 1e-5)
        expected = np.sqrt(np.mean((vectors[0].astype(np.float32) - vectors[36].astype(np.float32)) ** 2, dtype=np.float32), dtype=np.float32)
        self.assertEqual(rows[0][L2_KEY], float(expected))

    def test_zero_norm_and_non_finite_fail_closed(self) -> None:
        zero = np.ones((37, 4), dtype=np.float32)
        zero[12] = 0.0
        with self.assertRaisesRegex(Phase4HiddenGeometryError, "zero"):
            geometry_scalars_from_numpy(zero)
        bad = np.ones((37, 4), dtype=np.float32)
        bad[5, 1] = np.nan
        with self.assertRaisesRegex(Phase4HiddenGeometryError, "non-finite"):
            geometry_scalars_from_numpy(bad)

    def test_b2_record_projects_to_unchanged_p4b_contract(self) -> None:
        record = _geometry_record()
        base = strip_geometry_to_p4b(record)
        validate_trajectory_record(base, d36_tolerance=1e-6)
        self.assertNotIn(L2_KEY, base["boundaries"][0])
        self.assertNotIn(COSINE_KEY, base["boundaries"][0])

    def test_forbidden_payloads_and_derived_distance_are_rejected(self) -> None:
        for key in (
            "hidden_states",
            "residual_tensors",
            "logits",
            "probabilities",
            "tensor_payload",
            "target_answer",
            "gold",
            "generated_tokens",
            "loop_outcome",
        ):
            with self.subTest(key=key):
                record = _geometry_record()
                record[key] = [0.0]
                with self.assertRaises(Exception):
                    validate_hidden_geometry_record(record)
        record = _geometry_record()
        record["boundaries"][0]["hidden_cosine_distance_to_final"] = 0.2
        with self.assertRaises(Phase4HiddenGeometryError):
            validate_hidden_geometry_record(record)

    def test_exit_minus_entry_additivity_and_correlations(self) -> None:
        report = analyze_hidden_geometry_records([_geometry_record(index) for index in range(4)])
        self.assertEqual(report["checks"]["window_count_width4"], 33)
        self.assertEqual(report["checks"]["window_count_width2_8"], 224)
        self.assertLessEqual(report["checks"]["maximum_additivity_abs_error"], 1e-12)
        self.assertLessEqual(
            report["checks"]["maximum_delta_cosine_distance_plus_delta_cosine_abs_error"],
            1e-12,
        )
        self.assertEqual(len(report["mean_curve_correlations"]), 3)
        self.assertEqual(len(report["width4_sample_delta_sign_agreement"]), 99)


if __name__ == "__main__":
    unittest.main()
