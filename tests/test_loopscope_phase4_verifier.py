from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tflt.loopscope.phase4_schema import (
    Phase4ContractError,
    load_json_object,
    validate_provenance_evidence,
)
from tflt.loopscope.phase4_verifier import build_verifier_receipt


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/phase4_pv_ek_trs_card.json"
EVIDENCE = ROOT / "configs/loopscope/phase4_provenance_non_reuse.json"


class Phase4VerifierTests(unittest.TestCase):
    def test_deterministic_receipt_recomputation(self) -> None:
        first = build_verifier_receipt(CARD, ROOT)
        second = build_verifier_receipt(CARD, ROOT)
        self.assertEqual(first, second)
        self.assertTrue(first["card"]["provenance_closed"])
        self.assertFalse(first["card"]["normal_verify_fails_closed"])
        self.assertEqual(
            first["checks"]["provenance_closure_mode"],
            "NOT_REUSABLE_FRESH_REQUIRED",
        )
        self.assertTrue(first["checks"]["p4c_fresh_baseline_and_15_18_required"])
        self.assertFalse(first["checks"]["historical_artifacts_reused"])
        self.assertTrue(first["checks"]["width4_compatibility_verified"])

    def test_non_reuse_evidence_requires_identity_registry_reasons_and_fresh_flags(self) -> None:
        card = load_json_object(CARD)
        evidence = load_json_object(EVIDENCE)
        validate_provenance_evidence(card, evidence)
        cases = []
        missing_identity = copy.deepcopy(evidence)
        del missing_identity["historical_run"]["historical_15_18_identity"]
        cases.append(missing_identity)
        empty_registry = copy.deepcopy(evidence)
        empty_registry["historical_run"]["known_outcome_registry"] = []
        cases.append(empty_registry)
        empty_reasons = copy.deepcopy(evidence)
        empty_reasons["historical_run"]["non_reuse_reasons"] = []
        cases.append(empty_reasons)
        missing_fresh_flag = copy.deepcopy(evidence)
        missing_fresh_flag["historical_run"]["p4c_fallback"]["fixed_15_18"] = "UNSET"
        cases.append(missing_fresh_flag)
        for invalid in cases:
            with self.subTest(case=cases.index(invalid)):
                with self.assertRaises(Phase4ContractError):
                    validate_provenance_evidence(card, invalid)

    def test_reuse_closed_requires_every_reuse_admission_fact(self) -> None:
        card = load_json_object(CARD)
        evidence = load_json_object(EVIDENCE)
        card["provenance_closure"]["closure_mode"] = "REUSE_CLOSED"
        evidence["closure_mode"] = "REUSE_CLOSED"
        evidence["historical_run"]["reuse_status"] = "REUSE_CLOSED"
        evidence["historical_run"]["non_reuse_reasons"] = []
        evidence["historical_run"]["p4c_fallback"] = {
            "baseline": "REUSE_AUTHORIZED",
            "fixed_15_18": "REUSE_AUTHORIZED",
        }
        with self.assertRaisesRegex(Phase4ContractError, "complete reuse evidence"):
            validate_provenance_evidence(card, evidence)


if __name__ == "__main__":
    unittest.main()
