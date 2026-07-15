import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.schema import SchemaError, verify_manifest_sha256
from tflt.loopscope.phase3_pool import validate_source_contract
from tflt.loopscope.phase3_schema import (
    PHASE3_CARD_BYTE_SHA256,
    file_sha256,
    load_phase3_card,
    pool_record_json_schema,
    trajectory_record_json_schema,
)
from tflt.loopscope.phase3_verifier import (
    build_verifier_receipt,
    verify_committed_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/phase3_card.json"
RECEIPT = ROOT / "configs/loopscope/phase3_card_verifier_receipt.json"
SOURCE_CONTRACT = ROOT / "configs/loopscope/phase3_source_manifest.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prepare_module():
    path = ROOT / "scripts/loopscope/prepare_qwen17_phase3.py"
    spec = importlib.util.spec_from_file_location("prepare_qwen17_phase3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase3VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_phase3_card(CARD)

    def test_source_contract_is_explicitly_not_live_data(self):
        contract = _load(SOURCE_CONTRACT)
        validate_source_contract(contract, self.card)
        self.assertEqual(contract["status"], "CONTRACT_ONLY_NO_LIVE_RECORDS")
        self.assertEqual(contract["live_source_bindings"]["status"],
                         "UNMATERIALIZED_UNTIL_SEPARATELY_AUTHORIZED_P3B")
        self.assertIsNone(contract["live_source_bindings"]["ordered_identity_sha256"])
        self.assertEqual(contract["external_actions_performed"], [])

    def test_committed_json_schemas_match_authoritative_builders(self):
        self.assertEqual(
            _load(ROOT / "configs/loopscope/phase3_pool_schema.json"),
            pool_record_json_schema(),
        )
        self.assertEqual(
            _load(ROOT / "configs/loopscope/phase3_trajectory_schema.json"),
            trajectory_record_json_schema(),
        )

    def test_receipt_recomputes_exactly_from_raw_contract(self):
        observed = verify_committed_receipt(CARD, RECEIPT)
        expected = build_verifier_receipt(CARD)
        self.assertEqual(observed, expected)
        verify_manifest_sha256(observed)
        fixture = observed["synthetic_fixture"]
        self.assertEqual(fixture["record_count"], 1531)
        self.assertEqual(fixture["subject_count"], 57)
        self.assertTrue(fixture["raw_boundary_recomputation"])
        self.assertTrue(fixture["boundary_metric_tamper_rejected"])
        self.assertTrue(fixture["analysis_score_tamper_rejected"])
        self.assertTrue(fixture["analysis_fixture_only_not_scientific_output"])
        self.assertEqual(observed["external_actions_performed"], [])

    def test_receipt_tamper_fails_read_only_compare(self):
        changed = copy.deepcopy(_load(RECEIPT))
        changed["synthetic_fixture"]["analysis_score_tamper_rejected"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(SchemaError):
                verify_committed_receipt(CARD, path)

    def test_prepare_script_verify_mode_is_read_only(self):
        module = _load_prepare_module()
        before = file_sha256(RECEIPT)
        receipt = module.verify_contract(CARD)
        self.assertEqual(receipt["manifest_sha256"], _load(RECEIPT)["manifest_sha256"])
        self.assertEqual(file_sha256(RECEIPT), before)

    def test_verifier_binds_only_authorized_phase3_implementation_paths(self):
        receipt = _load(RECEIPT)
        self.assertEqual(file_sha256(CARD), PHASE3_CARD_BYTE_SHA256)
        self.assertEqual(
            set(receipt["implementation_sha256"]),
            {
                "src/tflt/loopscope/phase3_schema.py",
                "src/tflt/loopscope/phase3_pool.py",
                "src/tflt/loopscope/phase3_analysis.py",
                "src/tflt/loopscope/phase3_verifier.py",
                "scripts/loopscope/prepare_qwen17_phase3.py",
            },
        )
        self.assertNotIn("src/tflt/cli.py", receipt["implementation_sha256"])
        self.assertNotIn("src/tflt/loopscope/__init__.py", receipt["implementation_sha256"])


if __name__ == "__main__":
    unittest.main()
