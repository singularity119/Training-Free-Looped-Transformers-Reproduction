"""Bounded validation selection and formal/preflight information boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tflt.loopscope.phase10_accuracy import load_pool, validate_mmlu_preflight_pool
from tflt.loopscope.phase10_panel import load_config
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION, SEED, identity

_PATH = Path(__file__).resolve().parents[1] / "scripts/loopscope/prepare_phase10_preflight.py"
_SPEC = importlib.util.spec_from_file_location("prepare_phase10_preflight", _PATH)
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


def validation_fixture():
    config = load_config()
    counts = {f"subject_{index:02d}": (27 if index < 49 else 26) for index in range(57)}
    rows = [{"identity": identity(subject, index), "subject": subject, "doc_index": index,
             "split": "validation", "question": "question", "choices": ["one", "two", "three", "four"],
             "prompt": "five native demonstrations\nAnswer:",
             "prompt_token_lengths": {model["model"]: 20 for model in config["models"]}}
            for subject in sorted(counts) for index in range(counts[subject])]
    return {"schema_version": "loopscope.phase10.mmlu_preflight_inputs.v1", "status": "PREFLIGHT_ONLY_GOLD_FREE",
            "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "split": "validation"}, "seed": SEED,
            "validation_identities": [row["identity"] for row in rows], "rows": rows,
            "source_evidence": {"subject_counts": counts, "tokenizers": [
                {"model": model["model"], "revision": model["revision"]} for model in config["models"]],
                "loaded_splits": ["validation", "dev"], "target_gold_loaded": False, "lm_eval_version": "0.4.11"}}


class PreflightInputsTests(unittest.TestCase):
    def test_validation_pool_is_preflight_only(self):
        bundle = validation_fixture()
        prepare.validate_mmlu_pool(bundle)
        validate_mmlu_preflight_pool(bundle)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            path.write_text(json.dumps(bundle))
            self.assertEqual(len(load_pool(path, "mmlu", "PREFLIGHT_ONLY")), 1531)
            with self.assertRaises(ValueError):
                load_pool(path, "mmlu", "FORMAL_TEST")

    def test_validation_pool_rejects_target_field_and_order_change(self):
        bundle = validation_fixture()
        bundle["rows"][0]["answer"] = 1
        with self.assertRaises(ValueError):
            prepare.validate_mmlu_pool(bundle)
        bundle = validation_fixture()
        bundle["rows"][0], bundle["rows"][1] = bundle["rows"][1], bundle["rows"][0]
        with self.assertRaises(ValueError):
            prepare.validate_mmlu_pool(bundle)

    def test_arc_selection_covers_extrema_and_format_in_canonical_order(self):
        rows = [{"choices": ["text"] * count, "choice_labels": labels} for count, labels in
                [(4, list("ABCD")), (3, list("ABC")), (5, list("ABCDE")), (4, list("1234")),
                 (4, list("ABCD")), (4, list("ABCD"))]]
        lengths = {"q4": [{"context_length": value, "continuation_lengths": [i + 1]}
                            for i, value in enumerate([4, 3, 2, 1, 100, 6])],
                   "q17": [{"context_length": value, "continuation_lengths": [20 if i == 3 else 1]}
                             for i, value in enumerate([4, 3, 90, 1, 5, 6])]}
        selected, reasons = prepare.choose_indices(rows, lengths, "arc_challenge")
        self.assertEqual(selected, list(range(6)))
        self.assertEqual(reasons["q4:longest_context"], 4)
        self.assertEqual(reasons["q4:longest_continuation"], 5)
        self.assertEqual(reasons["numeric_labels"], 3)
        with self.assertRaises(ValueError):
            prepare.choose_indices(rows, lengths, "arc_challenge", maximum=5)

    def test_tied_lengths_select_first_canonical_row(self):
        rows = [{"choices": list("ABCD")} for _ in range(10)]
        lengths = {"model": [{"context_length": 12, "continuation_lengths": [1] * 4} for _ in rows]}
        self.assertEqual(prepare.choose_indices(rows, lengths, "mmlu")[0], [0])


if __name__ == "__main__":
    unittest.main()
