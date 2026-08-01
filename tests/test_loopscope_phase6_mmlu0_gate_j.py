import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope import phase6_mmlu0_gate_j as gate_j
from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    build_trajectory_record,
    choice_logits_to_probabilities,
    load_json,
    semantic_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase6_mmlu0_prefix_card.json"
EXPECTED_COMMIT = "1" * 40


def _record(card, ordinal, subject):
    probabilities = [
        choice_logits_to_probabilities([1.0 + 0.01 * index + ordinal * 1e-4, 0.2, -0.1, -0.4])
        for index in range(BOUNDARY_COUNT)
    ]
    hidden = [[1.0 + 0.01 * index, 0.2 + 0.005 * index, 0.5] for index in range(BOUNDARY_COUNT)]
    provenance = {
        "contract_version": card["renderer"]["contract_version"],
        "semantic_contract_sha256": card["renderer"]["semantic_contract_sha256"],
        "task_source_manifest_sha256": card["task"]["lm_eval_task_source"]["default_source_manifest_sha256"],
        "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
    }
    return build_trajectory_record(
        {"task": "mmlu", "doc_id": "mmlu_%s:validation:%d" % (subject, ordinal), "doc_hash": "%064x" % ordinal},
        subject,
        "%064x" % (ordinal + 1000),
        provenance,
        probabilities,
        hidden,
        card,
    )


class GateJMMLU0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_json(CARD_PATH)

    def test_dry_run_and_launcher_are_formal_only(self):
        result = gate_j.dry_run()
        self.assertEqual(result["status"], "DRY_RUN_VALID")
        self.assertEqual(result["gate"], "J")
        self.assertEqual(result["validation_record_count"], 1531)
        self.assertFalse(result["formal_loop_execution"])
        text = gate_j.build_sbatch_text(
            run_root=Path("/hpc2hdd/home/xhuang225/workspaces/gate-j-test"),
            expected_commit=EXPECTED_COMMIT,
            shard_count=8,
            partition="formal",
            gpu_type="A800",
            qos="normal",
            concurrency=8,
        )
        self.assertIn("#SBATCH --array=0-7%8", text)
        self.assertIn("#SBATCH --partition=formal", text)
        self.assertIn("#SBATCH --qos=normal", text)
        self.assertIn("#SBATCH --gres=gpu:A800:1", text)
        self.assertNotIn("--loop", text)

    def test_deterministic_sharding_covers_each_identity_once(self):
        rows = [
            {"ordinal": index, "identity": {"task": "mmlu", "doc_id": str(index)}, "subject": "s%d" % (index % 2)}
            for index in range(17)
        ]
        shards = gate_j._shard_rows(rows, 4)
        flattened = [row for shard in shards for row in shard]
        self.assertEqual(
            sorted(row["ordinal"] for row in flattened),
            [index for index in range(17)],
        )
        self.assertEqual(
            semantic_sha256([row["identity"] for row in sorted(flattened, key=lambda value: value["ordinal"])]),
            semantic_sha256([row["identity"] for row in rows]),
        )
        self.assertEqual(len({row["ordinal"] for row in flattened}), len(rows))

    def test_selector_freeze_requires_fresh_full_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest").mkdir()
            (root / "selector").mkdir()
            with self.assertRaises(gate_j.GateJMMLU0Error):
                gate_j.freeze_selector(run_root=root, expected_commit=EXPECTED_COMMIT)

    def test_no_forbidden_information_is_in_gate_j_launcher_or_dry_run(self):
        result = gate_j.dry_run()
        self.assertFalse(result["formal_loop_execution"])
        self.assertFalse(result["formal_outcome_read"])

    def test_record_fixture_is_selector_compatible(self):
        record = _record(self.card, 0, "subject0")
        projected = gate_j.selector_sample(record)
        self.assertEqual(set(projected), {"identity", "category", "H", "D"})


if __name__ == "__main__":
    unittest.main()
