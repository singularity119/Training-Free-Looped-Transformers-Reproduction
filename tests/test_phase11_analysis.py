import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase11_analysis import (
    BASE_FAMILY, FAMILY_COUNTS, ONLINE_FAMILY, STRATEGY_FAMILY,
    align_records, analyze_closed_panel, analyze_records, close_panel, frozen_contrasts,
    generation_prediction, gold_indices, score_cell, verify_attempt,
)
from tflt.loopscope.phase11_panel import MODEL, REVISION, build_panel, score_manifest
from tflt.loopscope.phase11_accuracy import engineering_cell, generation_record


def generated_record(index, text="Final answer: (A)", prediction=0, truncated=False):
    return {"identity": f"synthetic:{index}", "canonical_index": index, "category": "physics",
            "candidate_labels": ["A", "B", "C"], "status": "OK", "prediction_index": prediction,
            "generated_text": text, "generated_token_count": 7, "truncated": truncated,
            "end_reason": "length_limit" if truncated else "eos"}


def contrast_fake(reference, treatment, subjects, bootstrap_replicates, seed):
    return {"n": len(subjects), "mcnemar_exact_p": 0.5, "delta_pp": 100 * (sum(treatment)-sum(reference))/len(subjects),
            "bootstrap": {"replicates": bootstrap_replicates, "seed": seed}, "subjects": subjects}


def synthetic_pool(dataset="gpqa_main"):
    rows = [{"identity": f"synthetic:{i}", "task": dataset, "split": "synthetic", "subject": "physics",
             "choice_labels": ["A", "B", "C"], "choices": ["one", "two", "three"],
             "prompt": "synthetic chat prompt", "input_ids": [10, 11], "prompt_token_length": 2,
             "fewshot_sample_ids": []} for i in range(2)]
    return {"schema_version": "loopscope.phase11.inputs.v1", "task": dataset, "model": MODEL,
            "model_revision": REVISION, "tokenizer_revision": REVISION, "target_gold_loaded": False,
            "scope": "PREFLIGHT_ONLY", "engineering_synthetic": True,
            "rows": rows, "test_identities": [r["identity"] for r in rows]}


def write_attempt(root, panel, pool, manifest_path, cell, indices, check=None):
    root.mkdir()
    metadata = {"cell": cell, "acquisition_cell": engineering_cell(cell, check, panel["scope"]),
                "engineering_check": check, "dataset": panel["dataset"], "scope": panel["scope"],
                "indices": indices, "source_commit": "synthetic-commit", "pool": panel["pool"],
                "manifest": str(manifest_path)}
    (root / "summary.json").write_text(json.dumps({"status": "RAW_COMPLETE", "target_gold_loaded": False, "metadata": metadata}))
    (root / "command_args.json").write_text(json.dumps({"cell": cell}))
    records = [generation_record(pool["rows"][i], "Final answer: (A)", [12, 99], [99], 2048, i) for i in indices]
    (root / "records.jsonl").write_text("".join(json.dumps(record)+"\n" for record in records))


class AnalysisTests(unittest.TestCase):
    def test_frozen_44_comparisons_exclude_k2_aliases(self):
        panel = build_panel("mmlu_pro")
        contrasts = frozen_contrasts(panel)
        self.assertEqual(Counter(c["family"] for c in contrasts), Counter(FAMILY_COUNTS))
        self.assertEqual(len(contrasts), 44)
        aliases = {row["cell_id"] for row in panel["display_rows"] if row["alias_of"]}
        self.assertFalse(aliases.intersection(c[key] for c in contrasts for key in ("reference", "treatment")))

    def test_exact_identity_runtime_and_incomplete_rejection(self):
        records = [generated_record(i) for i in range(3)]
        ids = [r["identity"] for r in records]
        self.assertEqual(align_records("gpqa_main", list(reversed(records)), ids), records)
        for invalid in (records[:-1], records + [records[0]], [dict(records[0], status="RUNTIME_FAILED"), *records[1:]],
                        [dict(records[0], canonical_index=1), *records[1:]]):
            with self.assertRaises(ValueError):
                align_records("gpqa_main", invalid, ids)

    def test_full_denominator_failure_and_legal_truncation(self):
        records = [generated_record(0), generated_record(1, "reasoning without final line", None),
                   generated_record(2, truncated=True), generated_record(3, "Final answer: (A)\nFinal answer: (B)", None)]
        result, correct = score_cell("gpqa_main", records, [0, 0, 0, 0])
        self.assertEqual(correct, [1, 0, 1, 0])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["extraction_failure_rate"], 0.5)
        self.assertEqual(result["truncated_rate"], 0.25)
        with self.assertRaises(ValueError):
            generation_prediction(generated_record(0, "ordinary A", 0))

    def test_arc_acc_norm_and_descriptive_acc(self):
        record = {"status": "OK", "scores": [-2.0, -3.0], "choice_text_lengths": [1, 10]}
        result, correct = score_cell("arc_challenge", [record], [1])
        self.assertEqual(correct, [1])
        self.assertEqual(result["acc_norm"], 1.0)
        self.assertEqual(result["acc"], 0.0)

    def test_gold_exact_join_and_gpqa_sealed_letter(self):
        ids = ["synthetic:0", "synthetic:1"]
        labels = {identity: ["A", "B", "C"] for identity in ids}
        categories = {identity: "physics" for identity in ids}
        gold = [{"identity": ids[1], "gold_letter": "C", "option_source_indices": [2, 1, 0]},
                {"identity": ids[0], "label_index": 0, "category": "physics"}]
        self.assertEqual(gold_indices(gold, ids, labels, categories), [0, 2])
        for invalid in (gold[:1], gold + [gold[0]], [dict(gold[0], identity="unknown"), gold[1]]):
            with self.assertRaises(ValueError):
                gold_indices(invalid, ids, labels, categories)

    def test_analysis_subject_stratification_and_holm(self):
        for dataset in ("mmlu_pro", "gpqa_main"):
            panel = build_panel(dataset)
            records = [generated_record(i) for i in range(3)]
            records[1]["category"] = "chemistry"
            by_cell = {cell["cell_id"]: copy.deepcopy(records) for cell in panel["cells"]}
            gold = [{"identity": record["identity"], "label_index": 0} for record in records]
            with patch("tflt.loopscope.phase11_analysis.paired_contrast", side_effect=contrast_fake):
                result = analyze_records(panel, by_cell, gold)
            self.assertEqual(len(result["cells"]), 18)
            self.assertEqual(len(result["display_rows"]), 21)
            self.assertEqual(sum(row["alias_of"] is not None for row in result["display_rows"]), 3)
            self.assertEqual(len(result["contrasts"]), 44)
            for row in result["contrasts"]:
                self.assertEqual(row["holm_family_size"], FAMILY_COUNTS[row["family"]])
                self.assertEqual(row["bootstrap"], {"replicates": 10000, "seed": 20261008})
                self.assertEqual(row["subjects"], ["physics", "chemistry", "physics"] if dataset == "mmlu_pro" else ["all_items"] * 3)

    def test_gold_not_opened_before_fresh_raw_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_gold = Path(tmp) / "gold-does-not-exist.json"
            with patch("tflt.loopscope.phase11_analysis.close_panel", side_effect=ValueError("incomplete raw records")):
                with self.assertRaisesRegex(ValueError, "incomplete raw records"):
                    analyze_closed_panel({}, {}, {}, missing_gold)

    def test_single_preflight_attempt_uses_real_record_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pool = synthetic_pool()
            pool_path, manifest_path = directory / "pool.json", directory / "manifest.json"
            pool_path.write_text(json.dumps(pool))
            panel = score_manifest("gpqa_main", pool_path, "PREFLIGHT_ONLY")
            manifest_path.write_text(json.dumps(panel))
            cell = next(c for c in panel["cells"] if c["arm"] == "Online")
            root = directory / "attempt"
            write_attempt(root, panel, pool, manifest_path, cell, [0, 1], "zero-strength")
            verified = verify_attempt(panel, pool, root)
            self.assertEqual(verified["status"], "ATTEMPT_RAW_VERIFIED")
            self.assertFalse(verified["target_gold_loaded"])
            self.assertEqual(verified["records_count"], 2)
            self.assertNotIn("independent_cell_count", verified)
            raw = (root / "records.jsonl").read_text().splitlines()
            (root / "records.jsonl").write_text(raw[0]+"\n")
            with self.assertRaisesRegex(ValueError, "declared exact indices"):
                verify_attempt(panel, pool, root)

    def test_complete_shards_closure_and_overlap_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pool = synthetic_pool()
            pool_path, manifest_path = directory / "pool.json", directory / "manifest.json"
            pool_path.write_text(json.dumps(pool))
            panel = score_manifest("gpqa_main", pool_path, "FORMAL_TEST")
            manifest_path.write_text(json.dumps(panel))
            roots = {}
            for cell in panel["cells"]:
                roots[cell["cell_id"]] = []
                for i in (0, 1):
                    root = directory / f"{cell['cell_id']}-{i}"
                    write_attempt(root, panel, pool, manifest_path, cell, [i])
                    roots[cell["cell_id"]].append(str(root))
            # Only these test fixtures relax population size and source loading.
            # All shard metadata and raw validators follow the production path.
            with patch("tflt.loopscope.phase11_analysis.COUNTS", {"gpqa_main": 2}), patch(
                    "tflt.loopscope.phase11_accuracy.load_pool", return_value=pool):
                closure, records = close_panel(panel, pool, roots)
                self.assertEqual(closure["status"], "PANEL_CLOSED_GOLD_UNREAD")
                self.assertEqual(closure["sample_count"], 2)
                self.assertEqual(len(records), 18)
                broken = copy.deepcopy(roots)
                cell_id = panel["cells"][0]["cell_id"]
                broken[cell_id] = [roots[cell_id][0], roots[cell_id][0]]
                with self.assertRaisesRegex(ValueError, "overlap"):
                    close_panel(panel, pool, broken)
                broken[cell_id] = roots[cell_id][:1]
                with self.assertRaisesRegex(ValueError, "complete canonical"):
                    close_panel(panel, pool, broken)


if __name__ == "__main__":
    unittest.main()
