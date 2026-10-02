"""Focused canonical shard selection and merger tests; no model execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_phase10_accuracy import load_script, sample
from tflt.loopscope.phase10_accuracy import score_record, select_indices, write_json_once
from tflt.loopscope.phase10_panel import score_manifest


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pool = self.root / "pool.jsonl"
        self.rows = [sample(i) for i in range(4)]
        self.pool.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = score_manifest("arc_challenge", self.pool, "FORMAL_TEST")
        self.cells = [cell for cell in self.manifest["cells"] if cell["arm"] == "Native"]
        # Isolate merger semantics with four synthetic identities and two cells;
        # production validates the real full pool/manifest before reaching here.
        self.manifest["score_cells"] = self.cells
        write_json_once(self.manifest_path, self.manifest)
        self.verifier = load_script("verify_phase10_scores")
        for patch in (
                mock.patch.object(self.verifier, "load_pool", return_value=self.rows),
                mock.patch.object(self.verifier, "validate_score_manifest", return_value={c["cell_id"]: c for c in self.cells})):
            patch.start()
            self.addCleanup(patch.stop)

    def attempt(self, name, indices, cell=None, commit="source", scope="FORMAL_TEST", shard=True):
        cell = cell or self.cells[0]
        root = self.root / name
        root.mkdir()
        metadata = {"cell": cell, "dataset": "arc_challenge", "scope": scope,
            "pool": str(self.pool), "manifest": str(self.manifest_path), "indices": indices,
            "source_commit": commit, "target_gold_loaded": False}
        if shard:
            metadata["execution_selection"] = {"kind": "canonical_shard", "shard_id": name,
                "canonical_pool_count": len(self.rows), "indices": indices, "indices_path": "synthetic"}
        write_json_once(root / "command_args.json", metadata)
        write_json_once(root / "env.json", {"model_revision": cell["revision"],
            "model_dtype": "torch." + cell["dtype"], "dataset": "arc_challenge",
            "source_commit": commit, "max_length": 32768, "versions": {"lm_eval": "0.4.11"}})
        write_json_once(root / "summary.json", {"status": "SCORES_COMPLETE", "metadata": metadata,
            "count": len(indices), "target_gold_loaded": False, "positions_and_runtime": {
                "callback_by_t": {}, "applied_by_t": {}, "t0_mutation_count": 0,
                "nonanswer_mutation_count": 0}})
        records = [score_record(self.rows[i], "arc_challenge", [-i-1, -i-2, -i-3],
            [{"continuation_length": 1}] * 3, i if shard else None) for i in indices]
        (root / "scores.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
        return root

    def verify(self, roots, **kwargs):
        return self.verifier.verify(roots, self.manifest_path, self.pool, "FORMAL_TEST", **kwargs)

    def test_selection_preserves_canonical_domain_and_rejects_invalid_indices(self):
        self.assertEqual(select_indices(self.rows, self.cells[0], "FORMAL_TEST", shard=[0, 3]), [0, 3])
        self.assertEqual(select_indices(self.rows, self.cells[0], "FORMAL_TEST"), [0, 1, 2, 3])
        for indices in ([], [1, 1], [2, 0], [-1], [4], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                select_indices(self.rows, self.cells[0], "FORMAL_TEST", shard=indices)
        with self.assertRaises(ValueError):
            select_indices(self.rows, self.cells[0], "FORMAL_TEST", explicit=[0])

    def test_formal_runner_keeps_full_pool_and_manifest_for_subset(self):
        runner = load_script("run_phase10_accuracy")
        self.pool.write_text("\n".join(json.dumps(sample(i)) for i in range(1172)) + "\n")
        write_json_once(self.root / "formal-manifest.json",
            score_manifest("arc_challenge", self.pool, "FORMAL_TEST"))
        shard = self.root / "shard.json"
        write_json_once(shard, [0, 1171])
        args = runner.parser().parse_args(["--dataset", "arc_challenge", "--manifest",
            str(self.root / "formal-manifest.json"), "--pool", str(self.pool), "--cell-id",
            self.cells[0]["cell_id"], "--run-root", str(self.root / "unused"), "--commit", "synthetic",
            "--scope", "FORMAL_TEST", "--shard-indices", str(shard), "--shard-id", "tail-pair",
            "--dry-run"])
        manifest, _, rows, indices = runner.load_inputs(args)
        self.assertEqual(len(rows), 1172)
        self.assertEqual(len(manifest["cells"]), 68)
        self.assertEqual(indices, [0, 1171])
        self.assertFalse((self.root / "unused").exists())
        self.pool.write_text(json.dumps(sample(0)) + "\n")
        with self.assertRaises(ValueError):
            runner.load_inputs(args)

    def test_partial_union_and_canonical_order_equal_full_reference(self):
        first = self.attempt("first", [0, 3])
        second = self.attempt("second", [1, 2])
        closure, aligned = self.verify([first])
        self.assertEqual(closure["status"], "SCORE_ROOTS_CLOSED")
        self.assertEqual(closure["complete_cells"], [])
        self.assertEqual(aligned["indices_by_cell"][self.cells[0]["cell_id"]], [0, 3])
        with self.assertRaises(ValueError):
            self.verify([first], full_cell=True)
        merged, scores = self.verify([second, first], full_cell=True)
        self.assertEqual(merged["status"], "FULL_PHASE10_SCORE_CELLS_CLOSED")
        reference = self.attempt("reference", [0, 1, 2, 3], shard=False)
        _, expected = self.verify([reference], full_cell=True)
        self.assertEqual(scores["cells"], expected["cells"])

    def test_overlap_wrong_index_identity_and_mixed_source_rejected(self):
        first = self.attempt("first", [0, 3])
        overlap = self.attempt("overlap", [3])
        mixed = self.attempt("mixed", [1, 2], commit="other")
        for roots in ([first, overlap], [first, mixed]):
            with self.assertRaises(ValueError):
                self.verify(roots)
        path = first / "scores.jsonl"
        original = path.read_text()
        values = [json.loads(line) for line in original.splitlines()]
        for field, invalid in (("canonical_index", 1), ("identity", self.rows[1]["identity"])):
            changed = [{**values[0], field: invalid}, values[1]]
            path.write_text("\n".join(json.dumps(row) for row in changed) + "\n")
            with self.assertRaises(ValueError):
                self.verify([first])
        path.write_text(original)

    def test_full_panel_requires_all_cells_and_full_union(self):
        roots = [self.attempt("q4a", [0, 3]), self.attempt("q4b", [1, 2])]
        with self.assertRaises(ValueError):
            self.verify(roots, full_panel=True)
        roots.append(self.attempt("q17a", [0, 1], self.cells[1]))
        with self.assertRaises(ValueError):
            self.verify(roots, full_panel=True)
        roots.append(self.attempt("q17b", [2, 3], self.cells[1]))
        closure, aligned = self.verify(roots, full_panel=True)
        self.assertEqual(closure["status"], "FULL_PHASE10_SCORE_PANEL_CLOSED")
        self.assertEqual(closure["sample_count"], 8)
        self.assertEqual(aligned["identities"], [row["identity"] for row in self.rows])

    def test_preflight_shards_merge_only_reference_domain(self):
        reference = self.root / "preflight.json"
        write_json_once(reference, [0, 3])
        roots = [self.attempt("pre0", [0], scope="PREFLIGHT_ONLY"),
                 self.attempt("pre3", [3], scope="PREFLIGHT_ONLY")]
        closure, aligned = self.verifier.verify(roots, self.manifest_path, self.pool,
            "PREFLIGHT_ONLY", preflight_indices=reference, full_cell=True)
        self.assertEqual(closure["sample_count"], 2)
        self.assertEqual(aligned["indices_by_cell"][self.cells[0]["cell_id"]], [0, 3])
        with self.assertRaises(ValueError):
            select_indices(self.rows, self.cells[0], "PREFLIGHT_ONLY", explicit=[0, 3], shard=[1])


if __name__ == "__main__":
    unittest.main()
