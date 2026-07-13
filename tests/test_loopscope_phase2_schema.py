import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.phase2_analysis import choice_output
from tflt.loopscope.phase2_reuse import build_phase1_reuse_matrix, validate_reuse_matrix
from tflt.loopscope.phase2_schema import (
    b2_logical_cells,
    DIRECT_PROBE_SCORE_SOURCE,
    protocol_cell_id,
    validate_ordered_identity,
    validate_phase2_card,
    validate_scalar_sidecar,
    verify_phase2_implementation_hashes,
)
from tflt.loopscope.schema import SchemaError


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json"


class Phase2SchemaTests(unittest.TestCase):
    def test_card_hash_and_frozen_contract(self):
        payload = json.loads(CARD.read_text(encoding="utf-8"))
        validate_phase2_card(payload)
        verify_phase2_implementation_hashes(payload, ROOT)
        self.assertEqual(len(b2_logical_cells(payload)), 15)
        self.assertEqual(len({x["cell_id"] for x in b2_logical_cells(payload)}), 15)
        self.assertEqual(sum(x["protocol"] == "shared_k2_anchor" for x in b2_logical_cells(payload)), 3)

    def test_protocol_identity_keeps_alpha_and_protocol(self):
        self.assertNotEqual(
            protocol_cell_id("fixed_step", "12:15", 3, 1.5),
            protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
        )

    def test_ordered_identity_rejects_reordering(self):
        left = [
            {"task": "mmlu_a", "doc_id": "0", "doc_hash": "a" * 64},
            {"task": "mmlu_b", "doc_id": "0", "doc_hash": "b" * 64},
        ]
        with self.assertRaises(SchemaError):
            validate_ordered_identity(left, list(reversed(left)))

    def test_sidecar_rejects_vectors_and_nonfinite(self):
        base = choice_output(
            [4.0, 3.0, 2.0, 1.0],
            identity={"task": "mmlu", "doc_id": "1", "doc_hash": "a" * 64},
            score_source=DIRECT_PROBE_SCORE_SOURCE,
        )
        validate_scalar_sidecar(base)
        with self.assertRaises(SchemaError):
            validate_scalar_sidecar(dict(base, hidden_state=[1.0, 2.0]))
        with self.assertRaises(SchemaError):
            validate_scalar_sidecar(dict(base, metric=float("nan")))

    def test_reuse_matrix_only_has_gate_a_statuses(self):
        payload = build_phase1_reuse_matrix()
        validate_reuse_matrix(payload)
        statuses = {entry["status"] for entry in payload["entries"]}
        self.assertEqual(
            statuses,
            {"local_schema_confirmed", "historical_only", "requires_gate_b_live_check"},
        )
        raw = next(x for x in payload["entries"] if x["field"] == "raw_four_choice_loglikelihood")
        self.assertEqual(raw["status"], "requires_gate_b_live_check")


if __name__ == "__main__":
    unittest.main()
