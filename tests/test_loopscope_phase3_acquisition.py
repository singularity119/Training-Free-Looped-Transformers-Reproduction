import copy
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase3_acquisition import (
    AUTHORIZED_RUN_ROOT,
    P3BAcquisitionError,
    _default_dataset_loader,
    build_safe_validation_pool,
    build_shard_manifest,
    build_smoke_manifest,
    parse_sacct_rows,
    pool_record_from_anchor_and_safe_row,
    safe_dataset_fields,
    safe_target_doc_sha256,
    trajectory_renderer_provenance,
    validate_membership_manifest,
    write_new_json,
    write_new_jsonl,
)
from tflt.loopscope.phase3_pool import make_pool_manifest, make_source_manifest
from tflt.loopscope.phase3_schema import (
    load_phase3_card,
    reject_forbidden_selector_fields,
    validate_pool_record,
    validate_trajectory_renderer,
)
from tflt.loopscope.phase3_verifier import synthetic_pool_records


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase3_card.json"


class GuardedRow(Mapping):
    def __init__(self, question, choices, subject):
        self._data = {
            "question": question,
            "choices": choices,
            "subject": subject,
            "answer": 2,
        }
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key == "answer":
            raise AssertionError("target value must never be accessed")
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def _git():
    return {
        "root": "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope",
        "branch": "loopscope",
        "commit": "3" * 40,
        "origin_loopscope": "3" * 40,
        "dirty": False,
    }


class Phase3AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_phase3_card(CARD_PATH)
        cls.synthetic_pool = synthetic_pool_records(cls.card)
        cls.source = make_source_manifest(cls.synthetic_pool, cls.card)
        cls.pool_manifest = make_pool_manifest(
            cls.synthetic_pool, cls.source, cls.card
        )

    def test_safe_dataset_projection_never_reads_target_value(self):
        row = GuardedRow("Question?", ["a", "b", "c", "d"], "math")
        safe = safe_dataset_fields(row, "math")
        self.assertEqual(set(row.accessed), {"question", "choices", "subject"})
        self.assertNotIn("answer", row.accessed)
        self.assertEqual(safe["choices"], ["a", "b", "c", "d"])

    def test_pool_record_from_anchor_is_closed_world_and_gold_free(self):
        safe = {
            "question": "Question?",
            "choices": ["a", "b", "c", "d"],
            "subject": "math",
        }
        prompt = "Five-shot fixture\nQuestion?\nA. a\nB. b\nC. c\nD. d\nAnswer:"
        anchor = {
            "task_name": "mmlu_math",
            "target_doc_id": "mmlu_math:validation:0",
            "target_doc_index": 0,
            "target_doc_sha256": safe_target_doc_sha256(safe),
            "subject": "math",
            "text": prompt,
        }
        record = pool_record_from_anchor_and_safe_row(anchor, safe, self.card)
        validate_pool_record(record, self.card)
        reject_forbidden_selector_fields(record, self.card)
        self.assertEqual(record["identity"]["doc_hash"], safe_target_doc_sha256(safe))
        self.assertEqual(
            set(record),
            {
                "schema_version",
                "identity",
                "subject",
                "split",
                "question",
                "ordered_choices",
                "rendered_prompt",
                "prompt_sha256",
                "sanitized_content_sha256",
                "renderer_provenance",
            },
        )

    def test_full_safe_join_is_1531_57_and_guarded(self):
        anchors = []
        task_evidence = []
        rows_by_task = {}
        guarded_rows = []
        for subject_index in range(57):
            subject = "subject_%02d" % subject_index
            task_name = "mmlu_%s" % subject
            count = 27 if subject_index < 49 else 26
            rows = []
            for index in range(count):
                row = GuardedRow(
                    "Safe question %s %d?" % (subject, index),
                    ["choice %s %d %s" % (subject, index, label) for label in "ABCD"],
                    subject,
                )
                safe = {
                    "question": row._data["question"],
                    "choices": list(row._data["choices"]),
                    "subject": subject,
                }
                prompt = "Fixture prompt %s %d\nAnswer:" % (subject, index)
                anchors.append(
                    {
                        "task_name": task_name,
                        "target_doc_id": "%s:validation:%d" % (task_name, index),
                        "target_doc_index": index,
                        "target_doc_sha256": safe_target_doc_sha256(safe),
                        "subject": subject,
                        "text": prompt,
                    }
                )
                rows.append(row)
                guarded_rows.append(row)
            rows_by_task[task_name] = rows
            task_evidence.append(
                {
                    "task_name": task_name,
                    "dataset_path": "cais/mmlu",
                    "dataset_name": subject,
                }
            )

        def loader(task, card):
            del card
            rows = rows_by_task[task["task_name"]]
            return rows, {
                "task_name": task["task_name"],
                "dataset_name": task["dataset_name"],
                "validation_fingerprint": hashlib.sha256(
                    task["task_name"].encode("utf-8")
                ).hexdigest(),
                "row_count": len(rows),
            }

        pool, evidence = build_safe_validation_pool(
            anchors,
            {"dataset": {"tasks": task_evidence}},
            self.card,
            dataset_loader=loader,
        )
        self.assertEqual(len(pool), 1531)
        self.assertEqual(len({record["subject"] for record in pool}), 57)
        self.assertEqual(len(evidence), 57)
        self.assertTrue(all("answer" not in row.accessed for row in guarded_rows))

    def test_default_loader_requests_validation_split_only(self):
        calls = {}

        class Rows(list):
            _fingerprint = "validation-fingerprint"

        fake_datasets = types.ModuleType("datasets")

        class DownloadMode:
            REUSE_DATASET_IF_EXISTS = "reuse"

        def load_dataset(**kwargs):
            calls.update(kwargs)
            return Rows([GuardedRow("Question?", ["a", "b", "c", "d"], "math")])

        fake_datasets.DownloadMode = DownloadMode
        fake_datasets.load_dataset = load_dataset
        task = {
            "task_name": "mmlu_math",
            "dataset_path": "cais/mmlu",
            "dataset_name": "math",
            "raw_split_fingerprints": {"validation": "validation-fingerprint"},
        }
        with patch.dict(sys.modules, {"datasets": fake_datasets}), patch.dict(
            os.environ, {"HF_DATASETS_CACHE": "/read-only/cache"}
        ):
            rows, evidence = _default_dataset_loader(task, self.card)
        self.assertEqual(calls["split"], "validation")
        self.assertEqual(calls["revision"], self.card["task"]["dataset_revision"])
        self.assertNotIn("test", calls)
        self.assertEqual(len(rows), 1)
        self.assertEqual(evidence["validation_fingerprint"], "validation-fingerprint")

    def test_smoke_and_shards_are_exact_frozen_memberships(self):
        smoke = build_smoke_manifest(
            self.synthetic_pool,
            self.source,
            self.pool_manifest,
            self.card,
            _git(),
            AUTHORIZED_RUN_ROOT,
            ["fixture", "freeze-smoke"],
        )
        validate_membership_manifest(
            smoke, self.synthetic_pool, self.source, self.pool_manifest, self.card
        )
        self.assertEqual([row["ordinal"] for row in smoke["records"]], [0, 1, 2, 3])

        shards = build_shard_manifest(
            self.synthetic_pool,
            self.source,
            self.pool_manifest,
            self.card,
            _git(),
            AUTHORIZED_RUN_ROOT,
            7,
            "4" * 64,
            ["fixture", "freeze-shards"],
        )
        validate_membership_manifest(
            shards, self.synthetic_pool, self.source, self.pool_manifest, self.card
        )
        ordinals = [
            row["ordinal"] for shard in shards["shards"] for row in shard["records"]
        ]
        self.assertEqual(sorted(ordinals), list(range(1531)))
        self.assertEqual(len(ordinals), len(set(ordinals)))

        changed = copy.deepcopy(shards)
        changed["shards"][0]["records"][0]["ordinal"] = 1
        changed["shards"][0]["membership_sha256"] = hashlib.sha256(
            json.dumps(
                changed["shards"][0]["records"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        from tflt.loopscope.schema import attach_manifest_sha256

        attach_manifest_sha256(changed)
        with self.assertRaises(P3BAcquisitionError):
            validate_membership_manifest(
                changed,
                self.synthetic_pool,
                self.source,
                self.pool_manifest,
                self.card,
            )

    def test_trajectory_renderer_is_exact_native_no_loop(self):
        renderer = trajectory_renderer_provenance(
            self.card, self.source, self.pool_manifest
        )
        validate_trajectory_renderer(renderer, self.card)
        self.assertEqual(renderer["formal_forward_count_per_identity"], 1)
        self.assertEqual(renderer["loop_insertions"], 0)

    def test_jsonl_writer_is_exclusive_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            digest = write_new_jsonl(path, [{"value": 1}, {"value": 2}])
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                write_new_jsonl(path, [{"value": 3}])

    def test_json_writer_is_exclusive_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            digest = write_new_json(path, {"value": 1})
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"value": 2})

    def test_sacct_parser_preserves_terminal_resource_fields(self):
        row = "|".join(
            [
                "123_0",
                "p3b",
                "gpu3",
                "normal",
                "acct",
                "COMPLETED",
                "0:0",
                "gpu3-1",
                "cpu=8,gres/gpu=1",
                "cpu=8,gres/gpu=1",
                "120",
                "2026-07-16T01:00:00",
                "2026-07-16T01:01:00",
                "2026-07-16T01:03:00",
            ]
        ) + "|\n"
        parsed = parse_sacct_rows(row)
        self.assertEqual(parsed[0]["State"], "COMPLETED")
        self.assertEqual(parsed[0]["ElapsedRaw"], "120")


if __name__ == "__main__":
    unittest.main()
