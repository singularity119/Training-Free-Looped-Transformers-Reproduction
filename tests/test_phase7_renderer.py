import unittest

from tflt.loopscope.phase7_renderer import (
    MANIFEST_KEYS,
    canonical_identity,
    select_smoke_manifest_rows,
)
from tflt.loopscope.phase7_schema import Phase7ContractError


def _rows():
    return [
        {
            "canonical_identity": canonical_identity("mmlu_first", 0),
            "subject": "mmlu_first",
            "task_name": "mmlu_mmlu_first",
            "validation_index": 0,
        },
        {
            "canonical_identity": canonical_identity("mmlu_first", 1),
            "subject": "mmlu_first",
            "task_name": "mmlu_mmlu_first",
            "validation_index": 1,
        },
        {
            "canonical_identity": canonical_identity("mmlu_second", 0),
            "subject": "mmlu_second",
            "task_name": "mmlu_mmlu_second",
            "validation_index": 0,
        },
        {
            "canonical_identity": canonical_identity("mmlu_second", 1),
            "subject": "mmlu_second",
            "task_name": "mmlu_mmlu_second",
            "validation_index": 1,
        },
    ]


class Phase7RendererTests(unittest.TestCase):
    def test_common_manifest_uses_first_two_subjects_and_two_rows_each(self):
        rows = select_smoke_manifest_rows(_rows())
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["subject"] for row in rows], ["mmlu_first"] * 2 + ["mmlu_second"] * 2)
        self.assertTrue(all(set(row) == set(MANIFEST_KEYS) for row in rows))

    def test_rows_after_first_two_subjects_are_not_selected(self):
        rows = _rows() + [
            {
                "canonical_identity": canonical_identity("mmlu_third", 0),
                "subject": "mmlu_third",
                "task_name": "mmlu_mmlu_third",
                "validation_index": 0,
            }
        ]
        self.assertEqual(len(select_smoke_manifest_rows(rows)), 4)

    def test_subject_with_one_row_is_skipped(self):
        rows = [
            {
                "canonical_identity": canonical_identity("mmlu_short", 0),
                "subject": "mmlu_short",
                "task_name": "mmlu_mmlu_short",
                "validation_index": 0,
            }
        ] + _rows()
        selected = select_smoke_manifest_rows(rows)
        self.assertEqual({row["subject"] for row in selected}, {"mmlu_first", "mmlu_second"})

    def test_identity_and_manifest_keys_are_closed(self):
        invalid = _rows()
        invalid[0]["canonical_identity"] = "wrong"
        with self.assertRaises(Phase7ContractError):
            select_smoke_manifest_rows(invalid)


if __name__ == "__main__":
    unittest.main()
