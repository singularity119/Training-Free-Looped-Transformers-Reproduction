import unittest

from tflt.loopscope.phase7_manifest import split_manifest_rows, validate_manifest_rows
from tflt.loopscope.phase7_schema import Phase7ContractError


def _rows(count=6):
    return [
        {
            "canonical_identity": "mmlu_subject_%d:validation:%d" % (index // 3, index % 3),
            "subject": "subject_%d" % (index // 3),
            "task_name": "mmlu_subject_%d" % (index // 3),
            "validation_index": index % 3,
        }
        for index in range(count)
    ]


class Phase7ManifestTests(unittest.TestCase):
    def test_validate_manifest_closes_identity_and_subject(self):
        rows = _rows()
        validated = validate_manifest_rows(rows, expected_count=6, expected_subjects=2)
        self.assertEqual(validated, rows)

    def test_split_is_contiguous_balanced_and_complete(self):
        rows = validate_manifest_rows(_rows())
        shards = split_manifest_rows(rows, 4)
        self.assertEqual([len(shard) for shard in shards], [1, 2, 1, 2])
        self.assertEqual(
            [row["canonical_identity"] for shard in shards for row in shard],
            [row["canonical_identity"] for row in rows],
        )

    def test_forbidden_or_duplicate_membership_fails(self):
        invalid = _rows()
        invalid[0]["prompt_text"] = "must not persist"
        with self.assertRaisesRegex(Phase7ContractError, "BLOCK_INFORMATION_BARRIER_VIOLATION"):
            validate_manifest_rows(invalid)
        duplicate = _rows()
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaisesRegex(Phase7ContractError, "duplicate identity"):
            validate_manifest_rows(duplicate)


if __name__ == "__main__":
    unittest.main()
