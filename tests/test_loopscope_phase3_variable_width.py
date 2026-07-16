import copy
import hashlib
import math
import unittest
from pathlib import Path

from tflt.loopscope.phase3_variable_width import (
    EDGE_NAMES,
    STRICT_NAMES,
    analyze_boundary_metric_records,
    build_artifact_payloads,
    load_dual_card,
    verify_recomputed_artifacts,
    verify_width4_backward_compatibility,
)
from tflt.loopscope.schema import SchemaError


ROOT = Path(__file__).resolve().parents[1]


def metric_records():
    template = [1.0] * 29
    for boundary, value in {8: 0.0, 11: 0.0, 12: 2.0, 13: 0.0, 15: 1.0, 16: 0.0, 17: 1.0, 20: 1.0}.items():
        template[boundary] = value
    records = []
    for index in range(8):
        scale = 1.0 + 0.01 * index
        h_values = [scale * value for value in template]
        d_values = [4.0 - value for value in h_values]
        records.append(
            {
                "identity": {
                    "task": "mmlu_subject_%d" % (index % 2),
                    "doc_id": str(index),
                    "doc_hash": hashlib.sha256(("doc-%d" % index).encode()).hexdigest(),
                },
                "subject": "subject_%d" % (index % 2),
                "H": h_values,
                "D": d_values,
            }
        )
    return records


def selector_from_analysis(analysis):
    old_rows = []
    for row in analysis["strict_rows"]:
        if row["width"] != 4:
            continue
        contrasts = {}
        components = {}
        for name in STRICT_NAMES:
            contrasts[name] = {
                "point": row[name] * 4.0,
                "standard_error": row[name + "_se"] * 4.0,
                "percentile_95_ci": [value * 4.0 for value in row[name + "_ci95"]],
            }
            components[name] = row["z_" + name]
        old_rows.append(
            {
                "start": row["start"],
                "window": row["window"],
                "center": {
                    "entropy_drop": row["E_raw"],
                    "kl_rise": row["K_raw"],
                    "ce_drop": row["CEdrop_raw"],
                },
                "contrasts": contrasts,
                "score_components": components,
                "score": row["score"],
                "point_eligible": row["point_eligible"],
            }
        )
    return {
        "analysis": {
            "variants": {
                "CONSENSUS": {
                    "windows": old_rows,
                    "scopes": {
                        "deployment": {
                            "selection_frequency": analysis["width4_summary"]["selection_frequency"]
                        }
                    },
                }
            }
        }
    }


class VariableWidthAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_dual_card(
            ROOT / "configs/loopscope/phase3_variable_width_dual_card.json"
        )
        cls.records = metric_records()
        cls.analysis = analyze_boundary_metric_records(
            cls.records,
            cls.card,
            replicates=31,
            seed=20260716,
            enforce_population=False,
        )

    def test_strict_98_and_edge_168_counts_formulas_and_order(self):
        strict = self.analysis["strict_rows"]
        edge = self.analysis["edge_rows"]
        self.assertEqual(len(strict), 98)
        self.assertEqual(len(edge), 168)
        self.assertEqual(
            {width: sum(row["width"] == width for row in strict) for width in range(2, 9)},
            {2: 23, 3: 20, 4: 17, 5: 14, 6: 11, 7: 8, 8: 5},
        )
        first = next(row for row in strict if (row["width"], row["start"]) == (2, 2))
        by_pair = {(row["width"], row["start"]): row for row in edge}
        expected_oh = first["E_rate"] - 0.5 * (
            by_pair[(2, 1)]["E_rate"] + by_pair[(2, 3)]["E_rate"]
        )
        self.assertAlmostEqual(first["O_H"], expected_oh)
        self.assertTrue(all(math.isfinite(row["score"]) for row in strict + edge))
        for rows in (strict, edge):
            ranked = sorted(rows, key=lambda row: (-row["score"], row["width"], row["start"]))
            self.assertEqual([row["rank"] for row in ranked], list(range(1, len(rows) + 1)))

    def test_edge_reference_sets_are_exact_and_fully_cover_universe(self):
        edge = {(row["width"], row["start"]): row for row in self.analysis["edge_rows"]}
        self.assertEqual(edge[(2, 0)]["reference_starts"], [1, 2])
        self.assertEqual(edge[(2, 5)]["reference_starts"], [3, 4, 6, 7])
        self.assertEqual(edge[(8, 20)]["reference_starts"], [12, 19])
        counts = [row["reference_count"] for row in edge.values()]
        self.assertEqual(min(counts), 2)
        self.assertEqual(max(counts), 4)
        self.assertEqual(set(counts), {2, 3, 4})
        self.assertEqual(
            {(row["width"], row["start"]) for row in edge.values()},
            {(width, start) for width in range(2, 9) for start in range(29 - width)},
        )

    def test_joint_bootstrap_stream_and_width4_compatibility_fixture(self):
        self.assertEqual(
            self.analysis["bootstrap"]["index_stream_sha256"],
            "f0c9b32d8aeeaf1d33991936733b9694e842fcd7561f4d8acc2fa2a47499d1ad",
        )
        width4_eligible = [
            row["window"]
            for row in self.analysis["strict_rows"]
            if row["width"] == 4 and row["point_eligible"]
        ]
        self.assertEqual(width4_eligible, ["12:15"])
        self.assertEqual(self.analysis["width4_summary"]["selection_frequency"], 1.0)
        selector = selector_from_analysis(self.analysis)
        fixture_card = copy.deepcopy(self.card)
        row_12 = next(
            row for row in self.analysis["strict_rows"] if (row["width"], row["start"]) == (4, 12)
        )
        fixture_card["strict_consensus_98"]["backward_compatibility"][
            "reference_12_15_score"
        ] = row_12["score"]
        result = verify_width4_backward_compatibility(
            self.analysis["strict_rows"],
            self.analysis["width4_summary"],
            selector,
            fixture_card,
        )
        self.assertEqual(result["status"], "WIDTH4_EXACT_REPRODUCTION_PASS")

    def test_recompute_verifier_rejects_numeric_and_count_disagreement(self):
        selector = selector_from_analysis(self.analysis)
        fixture_card = copy.deepcopy(self.card)
        row_12 = next(
            row for row in self.analysis["strict_rows"] if (row["width"], row["start"]) == (4, 12)
        )
        fixture_card["strict_consensus_98"]["backward_compatibility"][
            "reference_12_15_score"
        ] = row_12["score"]
        compatibility = verify_width4_backward_compatibility(
            self.analysis["strict_rows"], self.analysis["width4_summary"], selector, fixture_card
        )
        artifacts = build_artifact_payloads(
            self.analysis,
            fixture_card,
            git_commit="a" * 40,
            input_provenance={"fixture": True},
            backward_compatibility=compatibility,
        )
        verify_recomputed_artifacts(
            self.analysis,
            fixture_card,
            selector,
            git_commit="a" * 40,
            input_provenance={"fixture": True},
            strict_payload=artifacts[0],
            strict_csv=artifacts[1],
            edge_payload=artifacts[2],
            edge_csv=artifacts[3],
            rankings_payload=artifacts[4],
        )
        numeric = copy.deepcopy(artifacts[0])
        numeric["rows"][0]["score"] += 1.0
        with self.assertRaisesRegex(SchemaError, "raw recomputation"):
            verify_recomputed_artifacts(
                self.analysis,
                fixture_card,
                selector,
                git_commit="a" * 40,
                input_provenance={"fixture": True},
                strict_payload=numeric,
                strict_csv=artifacts[1],
                edge_payload=artifacts[2],
                edge_csv=artifacts[3],
                rankings_payload=artifacts[4],
            )
        count = copy.deepcopy(artifacts[2])
        count["rows"].pop()
        count["row_count"] -= 1
        with self.assertRaisesRegex(SchemaError, "raw recomputation"):
            verify_recomputed_artifacts(
                self.analysis,
                fixture_card,
                selector,
                git_commit="a" * 40,
                input_provenance={"fixture": True},
                strict_payload=artifacts[0],
                strict_csv=artifacts[1],
                edge_payload=count,
                edge_csv=artifacts[3],
                rankings_payload=artifacts[4],
            )


if __name__ == "__main__":
    unittest.main()
