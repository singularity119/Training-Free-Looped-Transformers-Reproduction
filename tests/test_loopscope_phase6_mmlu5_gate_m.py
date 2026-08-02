import unittest
from pathlib import Path

from tflt.loopscope import phase6_mmlu5_gate_m as gate_m
from tflt.loopscope.phase6_mmlu5_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    FORMAL_SEED,
    TASK_SOURCE_MANIFEST_SHA256,
    build_trajectory_record,
    choice_logits_to_probabilities,
    load_json,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase6_mmlu5_prefix_card.json"


def _synthetic_record(index: int):
    probabilities = [
        choice_logits_to_probabilities(
            [1.0 + 0.01 * layer + 0.001 * index, 0.2, -0.1, -0.4]
        )
        for layer in range(BOUNDARY_COUNT)
    ]
    hidden = [
        [1.0 + 0.01 * layer, 0.2 + 0.005 * layer, 0.5 + 0.001 * index]
        for layer in range(BOUNDARY_COUNT)
    ]
    card = load_json(CARD_PATH)
    return build_trajectory_record(
        {"task": "mmlu", "doc_id": "synthetic-%d" % index, "doc_hash": ("%064x" % index)},
        "subject-%d" % (index % 2),
        ("b" * 64),
        {
            "contract_version": card["renderer"]["contract_version"],
            "semantic_contract_sha256": card["renderer"]["semantic_contract_sha256"],
            "task_source_manifest_sha256": TASK_SOURCE_MANIFEST_SHA256,
            "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
        },
        probabilities,
        hidden,
        card,
    )


class GateMMMLU5Tests(unittest.TestCase):
    def test_dry_run_is_formal_gate_m_only(self):
        result = gate_m.dry_run()
        self.assertEqual(result["status"], "DRY_RUN_VALID")
        self.assertEqual(result["gate"], "M")
        self.assertEqual(result["candidate_count"], 42)
        self.assertFalse(result["formal_acquisition"])
        self.assertFalse(result["selector_executed"])
        self.assertFalse(result["outcome_read"])
        self.assertEqual(result["gate_m_formal_runtime_dry_plan"]["formal_gpu"], "A800")

    def test_launcher_has_eight_shards_and_bound_resources(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest").mkdir()
            (root / "manifest/formal_manifest.json").write_text(
                '{"scope":"debug_preflight"}\n', encoding="utf-8"
            )
            text = gate_m.build_sbatch_text(
                run_root=root,
                expected_commit=gate_m.AUTHORIZED_ADMISSION_BASE,
                partition="debug",
                gpu_type="1",
            )
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --array=0-7%8", text)
        self.assertIn("#SBATCH --time=00:29:00", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn("#SBATCH --gres=gpu:1:1", text)
        self.assertNotIn("selector", text.lower())
        self.assertNotIn("outcome", text.lower())

    def test_selector_is_mmlu5_schema_bound_and_outcome_free(self):
        records = [_synthetic_record(index) for index in range(8)]
        result = gate_m.analyze_selector(records=records, replicates=8, seed=FORMAL_SEED, formal=False)
        self.assertEqual(result["record_count"], 8)
        self.assertEqual(result["category_count"], 2)
        self.assertEqual(result["candidate_count"], 42)
        self.assertIn(result["selector_decision"], gate_m.SCIENTIFIC_STATES)
        self.assertEqual(len(result["rows"]), 42)
        self.assertFalse(any("hidden" in str(row).lower() for row in result["rows"]))


if __name__ == "__main__":
    unittest.main()
