import copy
import hashlib
import math
import random
import unittest

from tflt.loopscope.phase5_schema import Phase5ContractError, canonical_json_bytes
from tflt.loopscope.phase5_variable_width import (
    CANDIDATE_STARTS,
    WIDTHS,
    analyze_boundary_samples,
    required_reference_starts,
    validate_extension_card,
    verify_width4_backward_compatibility,
)
from tflt.loopscope.phase5_variable_width import load_strict_json


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase5_multiwidth_mid40_card.json"


def _samples(count=12):
    rows = []
    for index in range(count):
        h_values = [
            2.0 - 0.015 * boundary + 0.0003 * index * math.sin(boundary + 1)
            for boundary in range(37)
        ]
        d_values = [
            0.8 - 0.009 * boundary + 0.0002 * index * math.cos(boundary + 1)
            for boundary in range(37)
        ]
        rows.append(
            {
                "identity": "id-%03d" % index,
                "subject": "subject-%d" % (index % 3),
                "H": h_values,
                "D": d_values,
            }
        )
    return rows


def _old_selector_fixture(rows):
    published = []
    for row in rows:
        if row["width"] != 4:
            continue
        contrasts = {}
        for new_name, old_name in (
            ("OH", "O_H"),
            ("OK", "O_K"),
            ("FH", "F_H"),
            ("FK", "F_K"),
        ):
            standard_error = row["contrast_standard_errors"][old_name]
            contrasts[old_name] = {
                "point": row[old_name],
                "standard_error": standard_error,
                "lower95": row["contrast_ci95"][old_name][0],
                "upper95": row["contrast_ci95"][old_name][1],
            }
            assert math.isclose(
                row["z_" + new_name],
                row[old_name] / max(standard_error, 1e-12),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        published.append(
            {
                "start": row["start"],
                "score": row["score"],
                "contrasts": contrasts,
                "point_eligible": row["point_eligible"],
                "failure_reasons": copy.deepcopy(row["eligibility_failures"]),
            }
        )
    return {"published_ranking": published}


class Phase5VariableWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_strict_json(CARD_PATH)

    def test_card_and_candidate_closure(self):
        validate_extension_card(self.card)
        self.assertEqual(
            {width: len(CANDIDATE_STARTS[width]) for width in WIDTHS},
            {3: 12, 4: 11, 5: 10, 6: 9},
        )
        self.assertEqual(sum(len(CANDIDATE_STARTS[width]) for width in WIDTHS), 42)
        for width in WIDTHS:
            for start in CANDIDATE_STARTS[width]:
                self.assertGreaterEqual(start, 11)
                self.assertLessEqual(start + width - 1, 24)
            self.assertTrue(set(CANDIDATE_STARTS[width]).issubset(required_reference_starts(width)))

    def test_formula_bootstrap_ranking_and_shared_draw_stream(self):
        samples = _samples()
        analysis = analyze_boundary_samples(
            samples, replicates=41, seed=20260722, enforce_population=False
        )
        reversed_analysis = analyze_boundary_samples(
            list(reversed(samples)), replicates=41, seed=20260722, enforce_population=False
        )
        self.assertEqual(analysis["candidate_count"], 42)
        self.assertEqual(
            analysis["bootstrap"]["draw_index_sha256"],
            reversed_analysis["bootstrap"]["draw_index_sha256"],
        )
        self.assertEqual(
            canonical_json_bytes(analysis["rows"]),
            canonical_json_bytes(reversed_analysis["rows"]),
        )
        self.assertEqual(
            [row["global_rank"] for row in analysis["rows"]], list(range(1, 43))
        )
        row = next(
            value for value in analysis["rows"] if value["width"] == 3 and value["start"] == 11
        )
        self.assertAlmostEqual(row["E_raw"], row["E_rate"] * 3, places=15)
        self.assertAlmostEqual(row["K_raw"], row["K_rate"] * 3, places=15)
        self.assertAlmostEqual(
            row["score"],
            min(row["z_OH"], row["z_OK"], row["z_FH"], row["z_FK"]),
            places=15,
        )
        self.assertEqual(
            row["comparison_starts"],
            {"s_minus_1": 10, "s_plus_1": 12, "s_minus_width": 8, "s_plus_width": 14},
        )

        normalized = sorted(samples, key=lambda value: (value["subject"], value["identity"]))
        groups = {}
        for index, value in enumerate(normalized):
            groups.setdefault(value["subject"], []).append(index)
        rng = random.Random(20260722)
        digest = hashlib.sha256()
        digest.update(b"[")
        for replicate in range(41):
            drawn = []
            for subject in sorted(groups):
                indices = groups[subject]
                drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
            if replicate:
                digest.update(b",")
            digest.update(canonical_json_bytes(drawn))
        digest.update(b"]")
        self.assertEqual(analysis["bootstrap"]["draw_index_sha256"], digest.hexdigest())

    def test_width4_exact_compatibility_and_material_mismatch(self):
        analysis = analyze_boundary_samples(
            _samples(), replicates=31, seed=7, enforce_population=False
        )
        selector = _old_selector_fixture(analysis["rows"])
        receipt = verify_width4_backward_compatibility(analysis["rows"], selector)
        self.assertEqual(receipt["row_count"], 11)
        self.assertEqual(receipt["max_absolute_difference"], 0.0)

        changed = copy.deepcopy(selector)
        changed["published_ranking"][0]["score"] += 1e-6
        with self.assertRaisesRegex(Phase5ContractError, "numeric mismatch"):
            verify_width4_backward_compatibility(analysis["rows"], changed)

        changed = copy.deepcopy(selector)
        changed["published_ranking"][0]["failure_reasons"].append("EXTRA")
        with self.assertRaisesRegex(Phase5ContractError, "failure reasons"):
            verify_width4_backward_compatibility(analysis["rows"], changed)

    def test_fail_closed_population_and_sample_schema(self):
        with self.assertRaisesRegex(Phase5ContractError, "1,531"):
            analyze_boundary_samples(_samples(), enforce_population=True)
        contaminated = _samples()
        contaminated[0]["outcome"] = 1
        with self.assertRaisesRegex(Phase5ContractError, "keys differ"):
            analyze_boundary_samples(contaminated, replicates=7, seed=1, enforce_population=False)


if __name__ == "__main__":
    unittest.main()
