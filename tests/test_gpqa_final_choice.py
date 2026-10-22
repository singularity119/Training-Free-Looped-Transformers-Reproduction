import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope import gpqa_final_choice as workflow


class FinalChoiceWorkflowTest(unittest.TestCase):
    def test_surface_is_candidate_only(self):
        self.assertEqual(workflow.terminal_candidate("Final answer: B")["selected_label"], "B")
        self.assertIsNone(workflow.terminal_candidate("Maybe B or C"))
        self.assertIsNone(workflow.terminal_candidate("Final answer: A\nFinal answer: B"))
        self.assertIsNone(workflow.terminal_candidate("I choose B."))  # needs actual review

    def test_complete_blind_seal_score_and_reject_invalid_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = dict(rule_version=workflow.RULE, expected_n=448,
                            cells=[dict(cell=c, group="same-run", configuration={"arm": c})
                                   for c in ("native", "loop")],
                            contrasts=[dict(id="loop-native", reference="native", treatment="loop",
                                            family="primary")])
            rows = [dict(cell=c, identity=f"item-{i}", canonical_index=i,
                         output="I choose B.\u2028", options=dict(zip("ABCD", "abcd")))
                    for c in ("native", "loop") for i in range(448)]
            with self.assertRaises(ValueError):
                workflow.validate_source(rows[:-1], manifest)
            workflow.write_rows(base / "source.jsonl", rows)
            workflow.write_json(base / "manifest.json", manifest)
            root = base / "analysis"
            workflow.prepare(base / "source.jsonl", base / "manifest.json", root)
            blind = workflow.read_rows(root / "blind/all.jsonl")
            self.assertEqual(set(blind[0]), {"id", "output", "options"})
            decisions = [dict(id=r["id"], rule_version=workflow.RULE, selected_label="B",
                              type="CLEAR_NATURAL_COMMITMENT", evidence_start=0, evidence_end=11,
                              evidence="I choose B.", reason="Explicit commitment", method="manual",
                              reviewer="test-reviewer") for r in blind]
            workflow.write_rows(base / "decisions.jsonl", decisions)
            workflow.write_json(base / "disclosure.json", dict(gold_accessed_in_new_extraction=False,
                                historical_exposure="synthetic only", manual_scope="synthetic fixture"))
            with self.assertRaises(FileNotFoundError):
                workflow.score(root, base / "nonexistent-gold.jsonl")
            workflow.seal(root, base / "decisions.jsonl", base / "disclosure.json")
            with self.assertRaises(ValueError):
                workflow.seal(root, base / "decisions.jsonl", base / "disclosure.json")
            gold = [dict(cell=r["cell"], identity=r["identity"], gold_letter="B") for r in rows]
            workflow.write_rows(base / "gold.jsonl", gold)
            with patch.object(workflow, "paired_ci", return_value=[0, 0]):
                result = workflow.score(root, base / "gold.jsonl")
            self.assertEqual([c["correct"] for c in result["configurations"]], [448, 448])
            self.assertEqual(result["contrasts"][0]["holm_p"], 1)
            self.assertEqual(result["contrasts"][0]["delta_pp"], 0)
            first = json.loads((root / "private/first-gold-access.json").read_text())
            with patch.object(workflow, "paired_ci", return_value=[0, 0]):
                workflow.score(root, base / "gold.jsonl", output=base / "attempt2")
            self.assertEqual(first, json.loads((root / "private/first-gold-access.json").read_text()))


if __name__ == "__main__":
    unittest.main()
