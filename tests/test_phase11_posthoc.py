import unittest

from tflt.loopscope.phase11_posthoc import extract_s1, score_s1


class PosthocTests(unittest.TestCase):
    def test_accepted_frozen_forms(self):
        for text in ("Final answer: A", "Final answer: (A)", " FINAL\tAnSwEr ： ( a )。 ",
                     "reasoning\nFinal answer: a.\n", "final answer:B！"):
            with self.subTest(text=text):
                self.assertEqual(extract_s1(text)["extraction_status"], "unique_valid")
        for mark in ".!;,．！。":
            self.assertEqual(extract_s1("Final answer: (C)" + mark)["predicted_label"], "C")

    def test_rejected_forms(self):
        for text in ("**Final answer: A**", "# Final answer: A", "The final answer: A",
                     "Final answer: A because ...", "\\boxed{A}", "Final answer: E",
                     "Final answer: A..", "Final answer: A，", "Final\nanswer: A",
                     "Final\vanswer: A", "Final answer: [A]"):
            with self.subTest(text=text):
                self.assertIsNone(extract_s1(text)["prediction_index"])

    def test_duplicate_and_conflicting_lines_fail(self):
        for last in ("A", "B"):
            result = extract_s1("Final answer: A\nFinal answer: (" + last + ")")
            self.assertEqual(result["valid_final_line_count"], 2)
            self.assertEqual(result["extraction_status"], "multiple_valid_final_lines")
            self.assertIsNone(result["predicted_label"])

    def test_failure_full_denominator_and_length_limit(self):
        rows = [dict(extract_s1(text), end_reason=reason) for text, reason in
                (("Final answer: A", "eos"), ("unfinished", "length_limit"),
                 ("Final answer: B", "length_limit"), ("Final answer: B", "eos"))]
        stats, correct = score_s1(rows, [0, 0, 1, 0])
        self.assertEqual(correct, [1, 0, 1, 0])
        self.assertEqual(stats["accuracy"], 0.5)
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["groups"]["length_limit"],
                         {"n": 2, "extraction_success_count": 1, "correct": 1})


if __name__ == "__main__":
    unittest.main()
