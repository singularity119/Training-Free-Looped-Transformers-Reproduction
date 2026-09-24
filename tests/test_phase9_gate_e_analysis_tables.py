import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "loopscope"))

from analyze_phase9_gate_e import PRIMARY_FAMILY, write_tables


class Phase9GateEAnalysisTablesTests(unittest.TestCase):
    def test_primary_contrast_csv_includes_bootstrap_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tables(Path(directory), {
                "cells": [],
                "contrasts": [{
                    "family": PRIMARY_FAMILY,
                    "reference": "old",
                    "treatment": "lag1",
                    "bootstrap": {"ci_low_pp": -0.25, "ci_high_pp": 0.5},
                }],
            })
            with (Path(directory) / "primary_contrasts.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["ci_low_pp"], "-0.25")
            self.assertEqual(row["ci_high_pp"], "0.5")


if __name__ == "__main__":
    unittest.main()
