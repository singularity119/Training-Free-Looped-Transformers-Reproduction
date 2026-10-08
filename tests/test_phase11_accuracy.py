import copy
import unittest

from tflt.loopscope.phase11_accuracy import extract_final_answer, generation_record, termination, validate_cell_records


class AccuracyTests(unittest.TestCase):
    def row(self, index=0):
        return {"identity": "synthetic:" + str(index), "task": "gpqa_main", "subject": "synthetic",
                "choice_labels": list("ABCD")}

    def test_strict_valid_duplicate_and_invalid_answers(self):
        self.assertEqual(extract_final_answer("reasoning\n \tFinal answer: (B) \n", "ABCD")["prediction_index"], 1)
        for text in ("Final answer: (b)", "Final answer: (Z)", "Final answer:(A)", "Final answer: (A).",
                     "I think Final answer: (A)"):
            self.assertIsNone(extract_final_answer(text, "ABCD")["prediction_index"])
        for text in ("Final answer: (A)\nFinal answer: (A)", "Final answer: (A)\nFinal answer: (B)"):
            self.assertEqual(extract_final_answer(text, "ABCD")["extraction_status"], "multiple_valid_final_lines")

    def test_eos_wins_at_limit_and_truncated_valid_answer_is_retained(self):
        self.assertFalse(termination([1, 9], [9], 2)["truncated"])
        record = generation_record(self.row(), "Final answer: (A)", [1, 2], [9], 2, 0)
        self.assertTrue(record["truncated"])
        self.assertEqual(record["prediction_index"], 0)
        with self.assertRaises(ValueError):
            termination([1], [9], 2)

    def test_extraction_failure_preserves_complete_denominator(self):
        records = [generation_record(self.row(i), text, [9], [9], 2, i)
                   for i, text in enumerate(("Final answer: (A)", "No final answer", "Final answer: (C)"))]
        ordered = validate_cell_records("gpqa_main", records, ["synthetic:0", "synthetic:1", "synthetic:2"])
        self.assertEqual(len(ordered), 3)
        self.assertIsNone(ordered[1]["prediction_index"])
        with self.assertRaises(ValueError):
            validate_cell_records("gpqa_main", records[:2], ["synthetic:0", "synthetic:1", "synthetic:2"])

    def test_runtime_failures_and_duplicate_identity_do_not_close(self):
        record = generation_record(self.row(), "Final answer: (A)", [9], [9], 2, 0)
        failed = copy.deepcopy(record)
        failed["status"] = "RUNTIME_FAILED"
        with self.assertRaises(ValueError):
            validate_cell_records("gpqa_main", [failed], ["synthetic:0"])
        with self.assertRaises(ValueError):
            validate_cell_records("gpqa_main", [record, record], ["synthetic:0", "synthetic:1"])


if __name__ == "__main__":
    unittest.main()
