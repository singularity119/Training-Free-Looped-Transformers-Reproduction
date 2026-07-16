from __future__ import annotations

import unittest
from pathlib import Path

from tflt.loopscope.phase4_verifier import build_verifier_receipt


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/phase4_pv_ek_trs_card.json"


class Phase4VerifierTests(unittest.TestCase):
    def test_deterministic_receipt_recomputation(self) -> None:
        first = build_verifier_receipt(CARD, ROOT)
        second = build_verifier_receipt(CARD, ROOT)
        self.assertEqual(first, second)
        self.assertFalse(first["card"]["provenance_closed"])
        self.assertTrue(first["card"]["normal_verify_fails_closed"])
        self.assertTrue(first["checks"]["width4_compatibility_verified"])


if __name__ == "__main__":
    unittest.main()
