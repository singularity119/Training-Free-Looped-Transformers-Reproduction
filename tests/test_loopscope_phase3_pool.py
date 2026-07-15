import copy
import hashlib
import unittest
from pathlib import Path

from tflt.loopscope.schema import (
    SchemaError,
    attach_manifest_sha256,
    canonical_json_bytes,
)
from tflt.loopscope.phase3_pool import (
    make_pool_manifest,
    make_source_manifest,
    partition_starts,
    validate_input_artifact_kind,
    validate_pool_records,
    validate_source_manifest,
    verify_population_disjointness,
)
from tflt.loopscope.phase3_schema import (
    load_phase3_card,
    ordered_identity_sha256,
)
from tflt.loopscope.phase3_verifier import synthetic_pool_records


ROOT = Path(__file__).resolve().parents[1]


class Phase3PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_phase3_card(ROOT / "configs/loopscope/phase3_card.json")
        cls.records = synthetic_pool_records(cls.card)
        cls.source = make_source_manifest(cls.records, cls.card)
        cls.pool_manifest = make_pool_manifest(cls.records, cls.source, cls.card)

    def test_exact_1531_57_happy_path(self):
        normalized = validate_pool_records(self.records, self.source, self.card)
        self.assertEqual(len(normalized), 1531)
        self.assertEqual(len({row["subject"] for row in normalized}), 57)
        self.assertEqual(self.pool_manifest["record_count"], 1531)
        self.assertEqual(self.pool_manifest["subject_count"], 57)

    def test_1530_1532_duplicate_and_wrong_split_fail(self):
        with self.assertRaisesRegex(SchemaError, "exactly 1,531"):
            validate_pool_records(self.records[:-1], self.source, self.card)
        with self.assertRaisesRegex(SchemaError, "exactly 1,531"):
            validate_pool_records(self.records + [self.records[-1]], self.source, self.card)

        duplicate = list(self.records)
        duplicate[1] = copy.deepcopy(duplicate[1])
        duplicate[1]["identity"] = copy.deepcopy(duplicate[0]["identity"])
        duplicate[1]["identity"]["doc_hash"] = duplicate[1]["identity"]["doc_hash"].upper()
        with self.assertRaisesRegex(SchemaError, "duplicate canonical identities"):
            validate_pool_records(duplicate, self.source, self.card)

        wrong_split = list(self.records)
        wrong_split[0] = copy.deepcopy(wrong_split[0])
        wrong_split[0]["split"] = "test"
        with self.assertRaisesRegex(SchemaError, "validation split"):
            validate_pool_records(wrong_split, self.source, self.card)

    def test_missing_extra_same_count_and_source_metadata_drift_fail(self):
        changed = list(self.records)
        changed[-1] = copy.deepcopy(changed[-1])
        changed[-1]["rendered_prompt"] += " changed"
        prompt_hash = hashlib.sha256(changed[-1]["rendered_prompt"].encode("utf-8")).hexdigest()
        changed[-1]["prompt_sha256"] = prompt_hash
        changed[-1]["renderer_provenance"]["render_sha256"] = prompt_hash
        with self.assertRaisesRegex(SchemaError, "close against"):
            validate_pool_records(changed, self.source, self.card)

        fewer_subjects = copy.deepcopy(self.records)
        for record in fewer_subjects:
            if record["subject"] == "subject_56":
                record["subject"] = "subject_00"
        with self.assertRaisesRegex(SchemaError, "57 subjects"):
            make_source_manifest(fewer_subjects, self.card)

    def test_forbidden_and_unknown_fields_fail_closed(self):
        forbidden = list(self.records)
        forbidden[0] = copy.deepcopy(forbidden[0])
        forbidden[0]["gold_label"] = 0
        with self.assertRaisesRegex(SchemaError, "forbidden selector field"):
            validate_pool_records(forbidden, self.source, self.card)

        unknown = list(self.records)
        unknown[0] = copy.deepcopy(unknown[0])
        unknown[0]["mystery"] = "x"
        with self.assertRaisesRegex(SchemaError, "extra"):
            validate_pool_records(unknown, self.source, self.card)

    def test_source_manifest_reordering_is_not_silently_sorted(self):
        changed = copy.deepcopy(self.source)
        changed["records"][0], changed["records"][1] = (
            changed["records"][1],
            changed["records"][0],
        )
        changed["ordered_identity_sha256"] = ordered_identity_sha256(
            [row["identity"] for row in changed["records"]]
        )
        changed["ordered_source_record_sha256"] = hashlib.sha256(
            canonical_json_bytes(changed["records"])
        ).hexdigest()
        attach_manifest_sha256(changed)
        with self.assertRaisesRegex(SchemaError, "canonical lexicographic order"):
            validate_source_manifest(changed, self.card)

    def test_historical_blind_partition_is_exact_disjoint_closure(self):
        partition = partition_starts(self.card)
        historical = set(partition["historical_13"])
        blind = set(partition["blind_12"])
        self.assertEqual(len(historical), 13)
        self.assertEqual(len(blind), 12)
        self.assertFalse(historical & blind)
        self.assertEqual(historical | blind, set(range(25)))

    def test_identity_and_content_disjointness_are_independent(self):
        validation = [
            {
                "identity": {"task": "mmlu_a", "doc_id": "0", "doc_hash": "a" * 64},
                "split": "validation",
                "sanitized_content_sha256": "1" * 64,
            }
        ]
        test = [
            {
                "identity": {"task": "mmlu_b", "doc_id": "0", "doc_hash": "b" * 64},
                "split": "test",
                "sanitized_content_sha256": "2" * 64,
            }
        ]
        receipt = verify_population_disjointness(
            validation, test, self.card, enforce_frozen_counts=False
        )
        self.assertEqual(receipt["identity_intersection_count"], 0)
        self.assertEqual(receipt["sanitized_content_intersection_count"], 0)

        identity_overlap = copy.deepcopy(test)
        identity_overlap[0]["identity"] = copy.deepcopy(validation[0]["identity"])
        with self.assertRaisesRegex(SchemaError, "identity intersection"):
            verify_population_disjointness(
                validation, identity_overlap, self.card, enforce_frozen_counts=False
            )

        content_overlap = copy.deepcopy(test)
        content_overlap[0]["sanitized_content_sha256"] = validation[0][
            "sanitized_content_sha256"
        ]
        with self.assertRaisesRegex(SchemaError, "sanitized-content intersection"):
            verify_population_disjointness(
                validation, content_overlap, self.card, enforce_frozen_counts=False
            )

    def test_artifact_kind_allowlist_rejects_outcome_values(self):
        self.assertEqual(
            validate_input_artifact_kind("validation1531_pool"), "validation1531_pool"
        )
        with self.assertRaisesRegex(SchemaError, "allowlist"):
            validate_input_artifact_kind("historical_outcome_values")


if __name__ == "__main__":
    unittest.main()
