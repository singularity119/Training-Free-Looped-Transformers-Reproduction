import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_accuracy import generation_record
from tflt.loopscope.phase11_panel import MODEL, REVISION, build_panel, load_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/analyze_phase11_gpqa_8192.py"
SPEC = importlib.util.spec_from_file_location("gpqa_8192_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def fixture(directory):
    config = load_config()
    recipe = config["datasets"]["gpqa_main"]
    rows = [{"identity": f"synthetic-d4:{i}", "task": "gpqa_main", "split": recipe["split"],
             "subject": "physics", "choice_labels": list("ABCD"), "choices": ["one", "two", "three", "four"],
             "prompt": "synthetic-only prompt", "input_ids": [10, 11], "prompt_token_length": 2,
             "fewshot_sample_ids": []} for i in range(448)]
    pool = {"schema_version": "loopscope.phase11.inputs.v1", "task": "gpqa_main", "model": MODEL,
            "model_revision": REVISION, "tokenizer_revision": REVISION, "target_gold_loaded": False,
            "scope": "FORMAL_TEST", "engineering_synthetic": False, "dataset_revision": recipe["revision"],
            "split": recipe["split"], "rows": rows, "test_identities": [r["identity"] for r in rows],
            **{key: recipe[key] for key in ("source_kind", "github_repo", "github_commit", "archive_member")}}
    pool_path, member_path = directory / "pool.json", directory / "membership.json"
    write(pool_path, pool)
    cells = build_panel("gpqa_main")["cells"]
    membership = {"schema": analysis.SCHEMA, "population_count": 448, "record_count": 8064,
        "indices": list(range(448)), "max_new_tokens": 8192, "target_gold_loaded": False,
        "cells": [{"cell": c, "indices": list(range(448))} for c in cells], "pool": str(pool_path),
        "samples": [{"canonical_index": i, "identity": row["identity"], "prompt_token_length": 2}
                    for i, row in enumerate(rows)],
        "context_admission": {"all_population_inputs_checked": 448, "max_prompt_tokens": 2,
                              "required_capacity": 8194, "configured_native_capacity": config["max_length"]}}
    write(member_path, membership)
    roots = {}
    for cell in cells:
        root = directory / cell["cell_id"]
        root.mkdir()
        meta = {"schema": analysis.SCHEMA, "cell": cell, "indices": list(reversed(range(448))),
            "scope": "FORMAL_TEST", "max_new_tokens": 8192, "synthetic_min_new_tokens": None,
            "target_gold_loaded": False, "source_commit": "producer-commit", "pool": str(pool_path),
            "membership": str(member_path)}
        write(root / "command_args.json", meta)
        records, telemetry = [], []
        for i in meta["indices"]:
            text = {1: "Final answer: B", 2: "Final answer: A\nFinal answer: (A)",
                    3: "unfinished"}.get(i, "Final answer: (A)")
            ids = [12] * 8192 if i == 3 else [12, 99]
            record = generation_record(rows[i], text, ids, [99], 8192, i)
            records.append(record)
            fact = {"canonical_index": i, "identity": rows[i]["identity"], "prefill_count": 1,
                    "decode_count": len(ids) - 1, "cache_growth_closed": True}
            if cell["arm"] == "Online":
                fact["online"] = {"decode_fit_count": 0, "directions_released": True}
            telemetry.append(fact)
        for filename, values in (("records.jsonl", records), ("telemetry.jsonl", telemetry)):
            (root / filename).write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
        write(root / "summary.json", {"schema": analysis.SCHEMA, "status": "GPQA_8192_ATTEMPT_COMPLETE",
            "metadata": meta, "count": 448, "generated_token_count": sum(r["generated_token_count"] for r in records),
            "target_gold_loaded": False})
        roots[cell["cell_id"]] = [str(root)]
    roots_path, gold_path = directory / "roots.json", directory / "gold.jsonl"
    write(roots_path, roots)
    gold_path.write_text("".join(json.dumps({"identity": row["identity"], "gold_letter": "A"}) + "\n"
                                for row in reversed(rows)), encoding="utf-8")
    return member_path, roots, roots_path, gold_path


class GPQA8192AnalysisTests(unittest.TestCase):
    def test_full_population_barriers_before_gold_and_frozen_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            member, roots, roots_path, gold = fixture(directory)
            output = directory / "analysis"
            args = SimpleNamespace(membership=str(member), cell_roots=str(roots_path), gold=str(gold),
                                   output_dir=str(output), executor_commit="analysis-commit", approved_source_commit=None)
            original = analysis.read_rows
            gold_opens = []

            def checked_read(path):
                if Path(path) == gold:
                    closure = analysis.read_json(output / "raw_closure.json")
                    barrier = analysis.read_json(output / "extraction_barrier.json")
                    self.assertEqual(closure["status"], "PANEL_CLOSED_GOLD_UNREAD")
                    self.assertEqual(closure["record_count"], 8064)
                    self.assertFalse(closure["gold_read"])
                    self.assertFalse(barrier["gold_read"])
                    self.assertFalse(analysis.read_json(output / "s2_spec.json")["gold_read"])
                    self.assertEqual(sum(barrier["status_counts"]["S2"].values()), 8064)
                    self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 8064)
                    gold_opens.append(path)
                return original(path)

            stdout = io.StringIO()
            with patch.object(analysis, "read_rows", side_effect=checked_read), contextlib.redirect_stdout(stdout):
                report = analysis.run(args)
            self.assertEqual(len(gold_opens), 1)
            self.assertEqual(report["primary_parser"], "S2")
            self.assertEqual(report["bootstrap_replicates"], 10000)
            self.assertEqual(report["bootstrap_seed"], 20261008)
            for result in (report, report["s0"], report["s1"]):
                self.assertEqual(len(result["cells"]), 18)
                self.assertEqual(len(result["contrasts"]), 44)
                for cell in result["cells"]:
                    self.assertEqual(cell["n"], 448)
                    self.assertEqual(cell["correct"], 445 if result is report["s1"] else 446)
                    self.assertEqual(cell["truncated_count"], 1)
                    self.assertEqual(cell["generation_length"]["max"], 8192)
                    self.assertLessEqual(cell["accuracy_ci"]["ci_low"], cell["accuracy"])
                    self.assertGreaterEqual(cell["accuracy_ci"]["ci_high"], cell["accuracy"])
                for contrast in result["contrasts"]:
                    self.assertEqual(contrast["holm_family_size"], analysis.FAMILY_COUNTS[contrast["family"]])
                    self.assertEqual(contrast["bootstrap"]["seed"], 20261008)
                    self.assertEqual(contrast["bootstrap"]["replicates"], 10000)
            self.assertEqual(report["cells"][0]["extraction_success_count"], 447)
            self.assertEqual(report["cells"][0]["consistent_multiple_correct"], 1)
            self.assertEqual(report["s1"]["cells"][0]["extraction_success_count"], 446)
            self.assertEqual(report["s0"]["cells"][0]["extraction_success_count"], 446)
            self.assertNotIn("synthetic-d4", stdout.getvalue())
            self.assertNotIn("synthetic-only prompt", json.dumps(report))
            self.assertNotIn("prediction_index", json.dumps(report))
            self.assertNotIn("generated_text", json.dumps(report))
            with self.assertRaisesRegex(ValueError, "already exists"):
                analysis.run(args)

    def test_missing_overlap_drift_and_mechanism_failure_do_not_open_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            member, roots, roots_path, gold = fixture(directory)
            cid = next(iter(roots))
            incomplete = copy.deepcopy(roots)
            incomplete.pop(cid)
            overlap = copy.deepcopy(roots)
            overlap[cid] *= 2
            for invalid, message in ((incomplete, "exact 18"), (overlap, "overlap")):
                write(roots_path, invalid)
                args = SimpleNamespace(membership=str(member), cell_roots=str(roots_path),
                    gold=str(directory / "nonexistent-gold"), output_dir=str(directory / "rejected"),
                    executor_commit="analysis-commit", approved_source_commit=None)
                with self.assertRaisesRegex(ValueError, message):
                    analysis.run(args)
                self.assertFalse(Path(args.output_dir).exists())
            root = Path(roots[cid][0])
            raw = analysis.read_rows(root / "records.jsonl")
            original_record = dict(raw[0])
            raw[0]["identity"] = "other-identity"
            (root / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw))
            with self.assertRaisesRegex(ValueError, "raw identity"):
                analysis.close_raw_panel(member, roots)
            raw[0] = original_record
            (root / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw))
            online_cid = next(c for c in roots if "online" in c)
            path = Path(roots[online_cid][0]) / "telemetry.jsonl"
            telemetry = analysis.read_rows(path)
            telemetry[0]["online"]["decode_fit_count"] = 1
            path.write_text("".join(json.dumps(r)+"\n" for r in telemetry))
            with self.assertRaisesRegex(ValueError, "direction lifecycle"):
                analysis.close_raw_panel(member, roots)

    def test_source_set_and_failed_summary_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            member, roots, _, _ = fixture(Path(tmp))
            root = Path(next(iter(roots.values()))[0])
            summary = analysis.read_json(root / "summary.json")
            summary["metadata"]["source_commit"] = "compatible-other-commit"
            write(root / "summary.json", summary)
            write(root / "command_args.json", summary["metadata"])
            with self.assertRaisesRegex(ValueError, "mixed source"):
                analysis.close_raw_panel(member, roots)
            closure, _, _ = analysis.close_raw_panel(member, roots, ["producer-commit", "compatible-other-commit"])
            self.assertEqual(len(closure["source_commits"]), 2)
            summary["status"] = "GPQA_8192_ATTEMPT_FAILED"
            write(root / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "complete gold-unread"):
                analysis.close_raw_panel(member, roots, ["producer-commit", "compatible-other-commit"])


if __name__ == "__main__":
    unittest.main()
