"""Exercise approved mixed sources through the real raw closure and CLI paths."""
import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase11_accuracy import engineering_cell, generation_record
from tflt.loopscope.phase11_analysis import analyze_closed_panel, close_panel, verify_attempt
from tflt.loopscope.phase11_panel import MODEL, REVISION, score_manifest

C_SOURCE = "447f64be3172dae403f7cd0c07ae17399a6197b4"
D_SOURCE = "d" * 40  # Audited D revision is supplied at execution, never hard-coded.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "loopscope"


class SourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        rows = [{"identity": f"synthetic:{i}", "task": "gpqa_main", "split": "synthetic",
                 "subject": "physics", "choice_labels": ["A", "B", "C"],
                 "choices": ["one", "two", "three"], "prompt": "synthetic chat prompt",
                 "input_ids": [10, 11], "prompt_token_length": 2,
                 "fewshot_sample_ids": []} for i in range(4)]
        self.pool = {"schema_version": "loopscope.phase11.inputs.v1", "task": "gpqa_main",
                     "model": MODEL, "model_revision": REVISION, "tokenizer_revision": REVISION,
                     "target_gold_loaded": False, "scope": "PREFLIGHT_ONLY",
                     "engineering_synthetic": True, "rows": rows,
                     "test_identities": [r["identity"] for r in rows]}
        self.pool_path = self.directory / "pool.json"
        self.pool_path.write_text(json.dumps(self.pool))
        self.panel = score_manifest("gpqa_main", self.pool_path, "FORMAL_TEST")
        self.manifest = self.directory / "manifest.json"
        self.manifest.write_text(json.dumps(self.panel))
        self.roots = {}
        for cell in self.panel["cells"]:
            self.roots[cell["cell_id"]] = [self.write_attempt(cell, [0, 2], C_SOURCE),
                                           self.write_attempt(cell, [1, 3], D_SOURCE)]
        # Only population/source loading are fixtures; manifest, raw record,
        # index coverage, candidate and source checks use the production path.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("tflt.loopscope.phase11_analysis.COUNTS", {"gpqa_main": 4}))
        stack.enter_context(patch("tflt.loopscope.phase11_accuracy.load_pool", return_value=self.pool))

    def write_attempt(self, cell, indices, commit, scope="FORMAL_TEST"):
        root = self.directory / f"{cell['cell_id']}-{scope}-{indices[0]}"
        root.mkdir()
        metadata = {"cell": cell, "acquisition_cell": engineering_cell(cell, None, scope),
                    "engineering_check": None, "dataset": "gpqa_main", "scope": scope,
                    "indices": indices, "source_commit": commit, "pool": str(self.pool_path),
                    "manifest": str(self.manifest)}
        (root / "summary.json").write_text(json.dumps(
            {"status": "RAW_COMPLETE", "target_gold_loaded": False, "metadata": metadata}))
        (root / "command_args.json").write_text(json.dumps(metadata))
        records = [generation_record(self.pool["rows"][i], "Final answer: (A)",
                                     [12, 99], [99], 2048, i) for i in indices]
        (root / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        return str(root)

    def test_default_is_strict_and_single_source_metadata_unchanged(self):
        with self.assertRaisesRegex(ValueError, "one source commit"):
            close_panel(self.panel, self.pool, self.roots)
        for roots in self.roots.values():
            path = Path(roots[1]) / "summary.json"
            summary = json.loads(path.read_text())
            summary["metadata"]["source_commit"] = C_SOURCE
            path.write_text(json.dumps(summary))
        closure, _ = close_panel(self.panel, self.pool, self.roots)
        self.assertEqual(closure["source_commit"], C_SOURCE)
        self.assertNotIn("source_commits", closure)
        self.assertNotIn("approved_source_commits", closure)

    def test_explicit_approval_closes_noncontiguous_original_indices(self):
        closure, records = close_panel(self.panel, self.pool, self.roots,
                                       approved_source_commits=[C_SOURCE, D_SOURCE])
        self.assertIsNone(closure["source_commit"])
        self.assertEqual(closure["source_commits"], sorted([C_SOURCE, D_SOURCE]))
        self.assertEqual(closure["approved_source_commits"], sorted([C_SOURCE, D_SOURCE]))
        self.assertEqual(closure["record_counts"], {cell: 4 for cell in self.roots})
        for rows in records.values():
            self.assertEqual([r["canonical_index"] for r in rows], [0, 1, 2, 3])
        for approved in ([], [C_SOURCE], [D_SOURCE]):
            with self.assertRaises(ValueError):
                close_panel(self.panel, self.pool, self.roots, approved_source_commits=approved)

    def test_approval_preserves_science_and_overlap_checks(self):
        approved = [C_SOURCE, D_SOURCE]
        cell = self.panel["cells"][0]["cell_id"]
        bad_roots = dict(self.roots, **{cell: [self.roots[cell][0], self.roots[cell][0]]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            close_panel(self.panel, self.pool, bad_roots, approved_source_commits=approved)
        path = Path(self.roots[cell][0]) / "summary.json"
        summary = json.loads(path.read_text())
        summary["metadata"]["cell"]["cache_strategy"] = "last"
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "scientific cell"):
            close_panel(self.panel, self.pool, self.roots, approved_source_commits=approved)

    def test_unapproved_source_stops_before_gold_read(self):
        with self.assertRaisesRegex(ValueError, "approved source set"):
            analyze_closed_panel(self.panel, self.pool, self.roots,
                                 self.directory / "gold-must-stay-unread.json",
                                 approved_source_commits=[C_SOURCE])

    def test_preflight_noncontiguous_attempt_accepts_only_approved_source(self):
        panel = score_manifest("gpqa_main", self.pool_path, "PREFLIGHT_ONLY")
        self.manifest.write_text(json.dumps(panel))
        root = self.write_attempt(panel["cells"][0], [1, 3], D_SOURCE, "PREFLIGHT_ONLY")
        result = verify_attempt(panel, self.pool, root, approved_source_commits=[C_SOURCE, D_SOURCE])
        self.assertEqual(result["indices"], [1, 3])
        self.assertEqual(result["identities"], ["synthetic:1", "synthetic:3"])
        with self.assertRaisesRegex(ValueError, "approved source set"):
            verify_attempt(panel, self.pool, root, approved_source_commits=[C_SOURCE])

    def test_both_cli_entries_forward_repeated_approval_to_fresh_closure(self):
        roots_path = self.directory / "roots.json"
        roots_path.write_text(json.dumps(self.roots))
        gold = self.directory / "gold.json"
        gold.write_text(json.dumps([{"identity": row["identity"], "label_index": 0}
                                    for row in self.pool["rows"]]))
        common = ["--manifest", str(self.manifest), "--cell-roots", str(roots_path),
                  "--approved-source-commit", C_SOURCE, "--approved-source-commit", D_SOURCE]
        for script, extra, artifact in (("verify_phase11_scores.py", [], "closure.json"),
                                         ("analyze_phase11.py", ["--gold", str(gold)], "analysis.json")):
            output = self.directory / script
            args = [script, *common, *extra, "--output-dir", str(output)]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()), patch(
                    "tflt.loopscope.phase11_analysis.BOOTSTRAP_REPLICATES", 2):
                runpy.run_path(str(SCRIPTS / script), run_name="__main__")
            result = json.loads((output / artifact).read_text())
            closure = result if artifact == "closure.json" else result["closure"]
            self.assertEqual(closure["source_commits"], sorted([C_SOURCE, D_SOURCE]))
            self.assertEqual(closure["approved_source_commits"], sorted([C_SOURCE, D_SOURCE]))


if __name__ == "__main__":
    unittest.main()
