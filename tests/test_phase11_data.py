import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.phase11_data import (
    GPQA_ARCHIVE_MEMBER, GPQA_ARCHIVE_URL, GPQA_COLUMNS, GPQA_GITHUB_COMMIT, GPQA_GITHUB_REPO,
    GPQA_SEED, INSTRUCTION, chat_input, gpqa_source_provenance, mmlu_target, native_mmlu_indices, normalize_demo_cot,
    prepare_gpqa_inputs, prepare_mmlu_inputs, render_generation_prompt, render_question,
    read_gpqa_source, selected_mmlu_demonstrations, write_phase11_inputs,
)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        self_content = messages[0]["content"]
        return [1] + [ord(c) for c in self_content] + [2] if tokenize else "<user>" + self_content + "</user><assistant>"


def target(index=0, option_count=3):
    return {"question_id": index, "question": "synthetic target question", "options": [f"option {i}" for i in range(option_count)],
            "category": "physics", "src": "synthetic", "answer": "TARGET_GOLD", "cot_content": "TARGET_COT"}


def demo(index=0):
    return {"question_id": index, "question": f"demo {index}", "options": ["first", "second", "third"],
            "category": "physics", "src": "synthetic", "cot_content": "A: Let's think step by step.\nReasoning.\nThe answer is (B).",
            "answer": "B", "identity": f"validation:{index}"}


def gpqa(index=0):
    return {"Question": f"synthetic GPQA {index}", "Correct Answer": f"candidate {index}.0",
            **{f"Incorrect Answer {i}": f"candidate {index}.{i}" for i in range(1, 4)}}


class DataTests(unittest.TestCase):
    def test_target_label_and_cot_never_accessed(self):
        class GoldGuard(dict):
            def __getitem__(self, key):
                if key in ("answer", "answer_index", "cot_content"):
                    raise AssertionError("target gold/CoT accessed")
                return super().__getitem__(key)
        source = GoldGuard(target())
        safe = mmlu_target(source)
        rows = prepare_mmlu_inputs([source], {"physics": [demo(i) for i in range(5)]}, Tokenizer(), "a" * 40, "test", 10000)
        self.assertNotIn("answer", safe)
        serialized = json.dumps(rows)
        self.assertNotIn("TARGET_GOLD", serialized)
        self.assertNotIn("TARGET_COT", serialized)
        self.assertNotIn("cot_content", serialized)
        self.assertEqual(rows[0]["choice_labels"], ["A", "B", "C"])

    def test_demo_normalization_and_variable_options(self):
        self.assertEqual(render_question("synthetic question", [" first ", " second "]),
                         "Question:\nsynthetic question\nOptions:\nA. first\nB. second\n")
        for n in (2, 4, 7, 10):
            content = render_generation_prompt(mmlu_target(target(option_count=n)), [demo(i) for i in range(5)])
            self.assertTrue(content.endswith(INSTRUCTION))
            self.assertEqual(content.count("Final answer: (B)"), 5)
            self.assertNotIn("The answer is (B)", content)
            self.assertNotIn("A: Let's think step by step.", content)
            self.assertNotIn("Answer: Answer:", content)
            self.assertIn("Answer: Let's think step by step.", content)
            self.assertIn("C. third\nAnswer: Let's think step by step.", content)
            self.assertTrue(content.startswith("Question:\ndemo 0\nOptions:\n"))
            self.assertIn(f"{'ABCDEFGHIJ'[n-1]}. option {n-1}", content)
        cot = normalize_demo_cot("Reasoning.\nFinal answer: (B)", "B", ["one", "two"])
        self.assertEqual(cot, "Reasoning.\nFinal answer: (B)")
        with self.assertRaises(ValueError):
            normalize_demo_cot("Reason", "C", ["one", "two"])

    def test_chat_is_single_user_and_complete(self):
        tokenizer = Tokenizer()
        result = chat_input("synthetic", tokenizer, 4096)
        self.assertEqual(result["prompt_token_length"], len("synthetic") + 2)
        for messages, _, generation_prompt in tokenizer.calls:
            self.assertEqual(messages, [{"role": "user", "content": "synthetic"}])
            self.assertTrue(generation_prompt)
        with self.assertRaises(ValueError):
            chat_input("long", Tokenizer(), 2049)

    def test_gpqa_deterministic_permutation_and_separate_gold(self):
        sources = [gpqa(i) for i in range(8)]
        rows, sealed = prepare_gpqa_inputs(sources, Tokenizer(), "b" * 40, "train", 10000)
        repeat = prepare_gpqa_inputs(sources, Tokenizer(), "b" * 40, "train", 10000)
        self.assertEqual((rows, sealed), repeat)
        rng = random.Random(GPQA_SEED)
        for index, (row, gold) in enumerate(zip(rows, sealed)):
            expected = list(range(4))
            rng.shuffle(expected)
            self.assertEqual(gold["option_source_indices"], expected)
            self.assertEqual(row["choices"], [f"candidate {index}.{i}" for i in expected])
            self.assertEqual(row["choices"]["ABCD".index(gold["gold_letter"])], sources[index]["Correct Answer"])
        text = json.dumps(rows)
        for forbidden in ("Correct Answer", "Incorrect Answer", "gold_letter", "option_source_indices"):
            self.assertNotIn(forbidden, text)

    def test_explicit_demonstration_identity_order(self):
        validation = [demo(i) for i in range(5)]
        selected = selected_mmlu_demonstrations(validation, {"physics": [4, 1, 2, 3, 0]}, "a" * 40)
        self.assertEqual([d["question"] for d in selected["physics"]], ["demo 4", "demo 1", "demo 2", "demo 3", "demo 0"])
        self.assertTrue(selected["physics"][0]["identity"].endswith(":validation:4:4"))
        with self.assertRaises(ValueError):
            selected_mmlu_demonstrations(validation, {"physics": [0, 0, 1, 2, 3]}, "a" * 40)
        validation += [demo(5)]
        self.assertEqual(native_mmlu_indices(validation), {"physics": [0, 1, 2, 3, 4]})

    def test_existing_gpqa_csv_full_count_projection_and_source_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "synthetic.csv"
            columns = [*GPQA_COLUMNS, "Unneeded explanation"]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for i in range(448):
                    writer.writerow({**gpqa(i), "Unneeded explanation": "discard this field"})
            source_revision = "b" * 40
            source = {"source_kind": "huggingface", "revision": source_revision, "count": 448,
                      "format": "csv", "files": [str(csv_path)],
                      "source_urls": [f"https://huggingface.co/datasets/Idavidrein/gpqa/resolve/{source_revision}/gpqa_main.csv"]}
            rows = read_gpqa_source(source)
            self.assertEqual(len(rows), 448)
            self.assertEqual(rows[0], gpqa(0))
            self.assertEqual(rows[-1], gpqa(447))
            self.assertTrue(all(set(row) == set(GPQA_COLUMNS) for row in rows))
            alternative = {**source, "revision": "a" * 40,
                           "source_urls": [source["source_urls"][0].replace(source_revision, "a" * 40)]}
            self.assertEqual(read_gpqa_source(alternative), rows)
            for changes in ({"count": 447}, {"revision": "not-an-exact-commit"}, {"revision": "a" * 40}, {"format": "json"},
                            {"source_urls": [source["source_urls"][0].replace("gpqa_main.csv", "gpqa_diamond.csv")]}):
                with self.assertRaises(ValueError):
                    read_gpqa_source({**source, **changes})
            lines = csv_path.read_text().splitlines()
            csv_path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "population"):
                read_gpqa_source(source)

    def test_accepted_author_archive_source_and_github_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "gpqa_main.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[*GPQA_COLUMNS, "Explanation"])
                writer.writeheader()
                for i in range(448):
                    writer.writerow({**gpqa(i), "Explanation": "not a model input"})
            source = {"source_kind": "author_github_archive", "github_repo": GPQA_GITHUB_REPO,
                      "github_commit": GPQA_GITHUB_COMMIT, "archive_member": GPQA_ARCHIVE_MEMBER,
                      "revision": GPQA_GITHUB_COMMIT, "split": "main_csv", "format": "csv",
                      "name": "gpqa_main", "count": 448, "files": [str(csv_path)], "source_urls": [GPQA_ARCHIVE_URL]}
            raw = read_gpqa_source(source)
            self.assertEqual(len(raw), 448)
            self.assertTrue(all(set(row) == set(GPQA_COLUMNS) for row in raw))
            rows, sealed = prepare_gpqa_inputs(raw[:2], Tokenizer(), GPQA_GITHUB_COMMIT, "main_csv", 10000, source=source)
            expected = [f"github:{GPQA_GITHUB_REPO}@{GPQA_GITHUB_COMMIT}:{GPQA_ARCHIVE_MEMBER}:{i}" for i in range(2)]
            self.assertEqual([row["identity"] for row in rows], expected)
            self.assertEqual([row["identity"] for row in sealed], expected)
            self.assertTrue(all(not any(key in row for key in ("source_kind", "github_commit", "archive_member")) for row in rows))
            facts = {"revision": GPQA_GITHUB_COMMIT, "split": "main_csv", **gpqa_source_provenance(source)}
            bundle = {"schema_version": "synthetic", "model": "fake", "model_revision": "b" * 40,
                      "tokenizer_revision": "b" * 40, "rows_by_task": {"gpqa_main": rows},
                      "sealed_gold_by_task": {"gpqa_main": sealed}, "task_facts": {"gpqa_main": facts}}
            output = Path(temporary) / "fresh-author-input"
            write_phase11_inputs(bundle, output)
            pool = json.loads((output / "gpqa_main-pool.json").read_text())
            for key in ("source_kind", "github_repo", "github_commit", "archive_member"):
                self.assertEqual(pool[key], source[key])
            self.assertEqual(pool["dataset_revision"], GPQA_GITHUB_COMMIT)
            self.assertEqual(pool["split"], "main_csv")
            for changes in ({"archive_member": "dataset/gpqa_diamond.csv"},
                            {"source_urls": [GPQA_ARCHIVE_URL.replace("dataset.zip", "gpqa_main.csv")]}):
                with self.assertRaises(ValueError):
                    read_gpqa_source({**source, **changes})

    def test_fresh_output_and_sealed_mapping(self):
        rows, sealed = prepare_gpqa_inputs([gpqa()], Tokenizer(), "b" * 40, "train", 10000)
        bundle = {"schema_version": "synthetic", "model": "fake", "model_revision": "b" * 40,
                  "tokenizer_revision": "b" * 40,
                  "rows_by_task": {"gpqa_main": rows}, "sealed_gold_by_task": {"gpqa_main": sealed},
                  "task_facts": {"gpqa_main": {"revision": "b" * 40, "split": "train"}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fresh"
            write_phase11_inputs(bundle, root)
            self.assertNotIn("gold_letter", (root / "gpqa_main-pool.json").read_text())
            self.assertNotIn("candidate 0", (root / "input_facts.json").read_text())
            self.assertIn("gold_letter", (root / "sealed/gpqa_main-gold.jsonl").read_text())
            with self.assertRaises(FileExistsError):
                write_phase11_inputs(bundle, root)


if __name__ == "__main__":
    unittest.main()
