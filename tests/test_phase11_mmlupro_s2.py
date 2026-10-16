import unittest

from tflt.loopscope.phase11_mmlupro_s2 import extract_mmlupro_s2, PARSER_VERSION
from tflt.loopscope.phase11_s2 import extract_s2


class MMLUProS2Tests(unittest.TestCase):
    def test_actual_option_range_a_d_e_j(self):
        for label, count, index in (("A", 4, 0), ("D", 4, 3), ("E", 5, 4), ("J", 10, 9)):
            value = extract_mmlupro_s2(f"Final answer: ({label})", count)
            self.assertEqual((value["extraction_status"], value["prediction_index"]), ("ok", index))
            self.assertEqual(value["parser_version"], PARSER_VERSION)
        for label, count in (("D", 3), ("E", 4), ("J", 9), ("Z", 10)):
            value = extract_mmlupro_s2(f"Final answer: ({label})", count)
            self.assertEqual(value["extraction_status"], "out_of_range")
            self.assertIsNone(value["prediction_index"])

    def test_case_parentheses_whitespace_and_s1_punctuation(self):
        for punctuation in ("", ".", "!", ";", ",", "．", "！", "。"):
            for letter in ("e", "( e )"):
                self.assertEqual(extract_mmlupro_s2(f"  fInAl\tAnSwEr ： {letter} {punctuation} ", 5)["prediction_index"], 4)

    def test_consistent_repeats_conflicts_and_out_of_range_priority(self):
        value = extract_mmlupro_s2("Final answer: E\nFinal answer: (e).", 5)
        self.assertEqual((value["answer_mode"], value["prediction_index"]), ("consistent_multiple", 4))
        self.assertEqual(extract_mmlupro_s2("Final answer: A\nFinal answer: E", 5)["extraction_status"], "conflict")
        value = extract_mmlupro_s2("Final answer: A\nFinal answer: B\nFinal answer: J", 4)
        self.assertEqual(value["extraction_status"], "out_of_range")
        self.assertIsNone(value["prediction_index"])

    def test_no_prose_recovery_or_invalid_syntax(self):
        for text in ("I think the final answer: (E)", "Final answer: (E) because", "E", "Final answer: EE",
                     "Final answer: [E]", "Final answer: (E)..", "Final answer: K"):
            self.assertEqual(extract_mmlupro_s2(text, 10)["extraction_status"], "no_match")
        for count in (0, 27, True, 4.0):
            with self.assertRaises(ValueError):
                extract_mmlupro_s2("Final answer: A", count)

    def test_gpqa_default_unchanged(self):
        self.assertIsNone(extract_s2("Final answer: (E)")["prediction_index"])
        self.assertEqual(extract_s2("Final answer: (D)")["prediction_index"], 3)


if __name__ == "__main__":
    unittest.main()
