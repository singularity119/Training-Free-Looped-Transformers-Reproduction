import json
from pathlib import Path
import tempfile
import unittest

from tflt.loopscope.phase9_accuracy import (
    ARMS,
    K_VALUES,
    load_config,
    logical_panel,
    manifest,
    validate_panel,
    write_manifest,
)


class Phase9PanelTests(unittest.TestCase):
    def test_frozen_panel_has_47_cells_and_three_retained_windows(self):
        config = load_config()
        cells = logical_panel(config)
        counts = validate_panel(cells, config)
        self.assertEqual(counts, {"cell_count": 47, "historical_count": 29,
                                  "new_count": 18, "native_count": 2})
        self.assertEqual(sum(cell["new_in_phase9"] for cell in cells), 18)
        self.assertEqual(sum(cell["arm"] == "Native" for cell in cells), 2)
        self.assertEqual({tuple(cell["window"]) for cell in cells if cell["window"]},
                         {(13, 16), (15, 18), (12, 15)})
        self.assertNotIn((12, 15), {
            tuple(cell["window"])
            for cell in cells
            if cell["model"] == "Qwen/Qwen3-4B-Base" and cell["window"]
        })
        self.assertNotIn((6, 9), {
            tuple(cell["window"])
            for cell in cells
            if cell["model"] == "Qwen/Qwen3-1.7B-Base" and cell["window"]
        })
        self.assertEqual({cell["k"] for cell in cells if cell["k"]}, set(K_VALUES))
        self.assertEqual(set(ARMS), {cell["arm"] for cell in cells if cell["window"]})

    def test_manifest_has_no_outcome_and_is_write_once(self):
        value = manifest(load_config())
        self.assertFalse(value["target_gold_loaded"])
        self.assertEqual(value["schema"], "loopscope.phase9.panel.v2")
        self.assertEqual(value["total_configuration_sample_records"], 659974)
        self.assertEqual(value["new_configuration_sample_records"], 252756)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, value)
            self.assertEqual(json.loads(path.read_text())["cell_count"], 47)
            with self.assertRaises(FileExistsError):
                write_manifest(path, value)


if __name__ == "__main__":
    unittest.main()
