import json
import random
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.phase10_data import (
    SEED, prepare_split, sample_identity, sanitize_target, summarize_lengths,
    train_answer_index, write_arc_inputs, describe_choices,
)


class Encoder:
    def _encode_pair(self, prompt, continuation):
        return [ord(c) for c in prompt], [ord(c) for c in continuation]


class Sampler:
    def __init__(self, rows):
        self.df = rows
        self.rnd = random.Random()
    def sample(self, n, eval_doc=None):
        assert eval_doc is None
        return self.rnd.sample(self.df, n)


class NativeTaskFake:
    def __init__(self, train):
        self.sampler = Sampler(train)
    def set_fewshot_seed(self, seed):
        self.sampler.rnd = random.Random(seed)
    def fewshot_context(self, doc, num_fewshot):
        demos = self.sampler.sample(num_fewshot)
        prefix = "".join("Question: " + d["question"] + "\nAnswer: "
                         + d["choices"]["text"][train_answer_index(d)] + "\n\n" for d in demos)
        return prefix + "Question: " + doc["question"] + "\nAnswer:"


def doc(index, labels=("A", "B")):
    return {"id": str(index), "question": f"question{index}",
            "choices": {"label": list(labels), "text": [f"answer{i}" for i in range(len(labels))]},
            "answerKey": labels[-1]}


class DataTests(unittest.TestCase):
    def test_variable_option_count_numeric_and_letter_labels(self):
        for labels in (("1", "2", "3"), ("A", "B", "C", "D", "E")):
            row = doc(3, labels)
            safe = sanitize_target(row)
            self.assertEqual(len(safe["choices"]["text"]), len(labels))
            self.assertEqual(train_answer_index(row), len(labels) - 1)
            self.assertNotIn("answerKey", safe)

    def test_target_gold_field_is_never_read(self):
        class GoldGuard(dict):
            def __getitem__(self, key):
                if key == "answerKey":
                    raise AssertionError("target gold accessed")
                return super().__getitem__(key)
        self.assertNotIn("answerKey", sanitize_target(GoldGuard(doc(4))))

    def test_seeded_native_draw_sequence_and_gold_free_output(self):
        train = [doc(i, ("1", "2", "3")) for i in range(30)]
        task = NativeTaskFake(train)
        targets = [doc(500), doc(501)]
        encoders = {"fake-model": {"encoder": Encoder(), "max_length": 9999}}
        rows = prepare_split(task, targets, "a" * 40, "test", encoders)
        rnd = random.Random(SEED)
        expected = [[sample_identity("a" * 40, "train", int(d["id"]), d["id"])
                     for d in rnd.sample(train, 25)] for _ in targets]
        self.assertEqual([r["fewshot_sample_ids"] for r in rows], expected)
        self.assertNotEqual(expected[0], expected[1])
        self.assertEqual(rows[0]["prompt"].split("\n\n")[-1], "Question: question500\nAnswer:")
        self.assertNotIn("answerKey", json.dumps(rows))
        self.assertEqual(prepare_split(task, targets, "a" * 40, "validation", encoders)[0]["fewshot_sample_ids"], expected[0])
        facts = summarize_lengths(rows, encoders)
        self.assertEqual(facts["fake-model"]["candidate_prefix_inconsistent_count"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fresh"
            bundle = {"schema_version": "synthetic", "rows_by_split": {"test": rows}}
            write_arc_inputs(bundle, root)
            text = (root / "ARC-test.jsonl").read_text()
            self.assertNotIn("answerKey", text)
            with self.assertRaises(FileExistsError):
                write_arc_inputs(bundle, root)

    def test_inconsistent_prefix_is_diagnostic_not_silent_repair(self):
        stats = describe_choices("abcdefghi", ["x", "long"], Encoder(), 10)
        self.assertFalse(stats["candidate_prefix_consistent"])
        self.assertEqual(stats["continuation_lengths"], [2, 5])
        self.assertEqual([c["left_truncated_tokens"] for c in stats["candidates"]], [0, 3])


if __name__ == "__main__":
    unittest.main()
