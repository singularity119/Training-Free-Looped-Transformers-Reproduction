import json
import tempfile
import unittest
from pathlib import Path

from tflt.looppilot.controller import Action
from tflt.looppilot.offline import EvalSample, strict_join_samples, validate_doc_actions
from tflt.looppilot.revisions import build_revision_closure
from tflt.looppilot.schema import DocKey


def key(occurrence="0"):
    return DocKey("mmlu", "1", "a" * 64, "b" * 64, occurrence)


def sample(join_key=None, revision="rev-1", prediction=0, gold=0):
    return EvalSample(
        key=join_key or key(),
        revision_closure_id=revision,
        prediction=prediction,
        gold=gold,
        choice_loglikelihoods=(-1.0, -2.0, -3.0, -4.0),
        subject="abstract_algebra",
        prompt_length=32,
    )


class OfflineRevisionTest(unittest.TestCase):
    def test_strict_join_classifies_flips(self):
        pairs = strict_join_samples([sample(prediction=0)], [sample(prediction=1)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].outcome, "loop_hurts")
        self.assertTrue(pairs[0].baseline_correct)
        self.assertFalse(pairs[0].loop_correct)

    def test_join_missing_duplicate_and_revision_mismatch_fail_fast(self):
        with self.assertRaises(ValueError):
            strict_join_samples([sample()], [])
        with self.assertRaises(ValueError):
            strict_join_samples([sample(), sample()], [sample()])
        with self.assertRaises(ValueError):
            strict_join_samples([sample(revision="a")], [sample(revision="b")])

    def test_four_choice_requests_share_one_action(self):
        rows = [(key(), index, Action.LOOP_K2) for index in range(4)]
        validate_doc_actions(rows)
        rows[-1] = (key(), 3, Action.BASELINE)
        with self.assertRaises(ValueError):
            validate_doc_actions(rows)

    def test_revision_closure_requires_resolved_artifacts_and_hashes_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model_revision.json"
            tokenizer = root / "tokenizer_revision.json"
            renderer = root / "renderer.yaml"
            config = root / "phase1.json"
            model.write_text(json.dumps({"repo_id": "Qwen/Qwen3-1.7B-Base", "commit_hash": "1" * 40}), encoding="utf-8")
            tokenizer.write_text(json.dumps({"repo_id": "Qwen/Qwen3-1.7B-Base", "commit_hash": "1" * 40}), encoding="utf-8")
            renderer.write_text("task: mmlu\n", encoding="utf-8")
            config.write_text("{}\n", encoding="utf-8")
            closure = build_revision_closure(model, tokenizer, renderer, config, "0.4.11")
            self.assertEqual(closure.model_commit, "1" * 40)
            self.assertEqual(closure.tokenizer_commit, "1" * 40)
            self.assertEqual(len(closure.renderer_hash), 64)
            self.assertEqual(len(closure.closure_id), 64)

            model.write_text(json.dumps({"repo_id": "Qwen/Qwen3-1.7B-Base"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_revision_closure(model, tokenizer, renderer, config, "0.4.11")


if __name__ == "__main__":
    unittest.main()
