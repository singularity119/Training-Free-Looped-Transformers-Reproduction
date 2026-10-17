import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/loopscope"))
import score_phase11_d7_interim as cli


class InterimTests(unittest.TestCase):
    def fixture(self):
        saved, data = [], {}
        for cell in cli.CELLS:
            raw, auxiliary = [], []
            for i in range(2):
                parsed = {"prediction_index": 0 if i == 0 else None,
                          "extraction_status": "ok" if i == 0 else "no_match"}
                row = {"cell_key": cell, "identity": str(i), "canonical_index": i,
                       "seed": 20261016, "generated_token_count": 2, "end_reason": "eos", **parsed}
                saved.append(row)
                raw.append(row)
                auxiliary.append(parsed)
            data[cell] = {"raw": raw, "auxiliary": auxiliary}
        return saved, data, ["0", "1"]

    @patch.object(cli, "COUNT", 2)
    def test_complete_saved_and_failed_extraction_full_denominator(self):
        saved, data, identities = self.fixture()
        cli.verify_saved(saved, data, identities)
        gold = [{"identity": "0", "label_index": 0}, {"identity": "1", "label_index": 1}]
        result = cli.score_saved(saved, gold, identities)
        self.assertEqual([r["accuracy_percent"] for r in result], [50, 50])
        self.assertEqual([r["format_failure_count"] for r in result], [1, 1])

    @patch.object(cli, "COUNT", 2)
    def test_incomplete_unknown_cell_and_changed_snapshot_rejected(self):
        saved, data, identities = self.fixture()
        for wrong in (saved[:-1], saved + [saved[0]]):
            with self.assertRaises(ValueError):
                cli.verify_saved(wrong, data, identities)
        wrong = copy.deepcopy(saved)
        wrong[0]["cell_key"] = "fixed_t0_k3_l09"
        with self.assertRaises(ValueError):
            cli.verify_saved(wrong, data, identities)
        wrong = copy.deepcopy(saved)
        wrong[0]["prediction_index"] = 1
        with self.assertRaises(ValueError):
            cli.verify_saved(wrong, data, identities)

    def test_incomplete_raw_prevents_gold_access(self):
        with patch.object(cli, "read", return_value={}), \
             patch.object(cli, "validate_cells", return_value=[{"cell_key": c} for c in cli.CELLS]), \
             patch.object(cli, "load_raw_cells", side_effect=ValueError("cell incomplete")), \
             patch.object(cli, "read_projection") as reader:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                cli.run(type("Args", (), {"plan": "unused", "output_dir": Path("unused")})())
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
