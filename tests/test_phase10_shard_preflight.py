import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope.phase10_accuracy import score_record, write_json_once
from tflt.loopscope.phase10_panel import load_config, score_manifest
from tflt.loopscope.phase10_runtime import direction_schedule


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/loopscope/check_phase10_shard_preflight.py"
SPEC = importlib.util.spec_from_file_location("check_phase10_shard_preflight", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def validation_fixture():
    config = load_config()
    revision = config["datasets"]["mmlu"]["revision"]
    counts = {f"subject_{i:02d}": 27 if i < 49 else 26 for i in range(57)}
    rows = [{"identity": f"cais/mmlu@{revision}:validation:{subject}:{i}",
             "subject": subject, "doc_index": i, "split": "validation", "question": "question",
             "choices": ["one", "two", "three", "four"], "prompt": "demonstrations\nAnswer:",
             "prompt_token_lengths": {model["model"]: 20 for model in config["models"]}}
            for subject in sorted(counts) for i in range(counts[subject])]
    return {"schema_version": "loopscope.phase10.mmlu_preflight_inputs.v1",
            "status": "PREFLIGHT_ONLY_GOLD_FREE", "dataset": {"repo": "cais/mmlu", "revision": revision, "split": "validation"},
            "validation_identities": [row["identity"] for row in rows], "rows": rows,
            "source_evidence": {"subject_counts": counts, "target_gold_loaded": False,
                                "loaded_splits": ["validation", "dev"]}}


def inputs(root):
    pool = root / "pool.json"
    write_json_once(pool, validation_fixture())
    manifest = root / "manifest.json"
    write_json_once(manifest, score_manifest("mmlu", pool, "PREFLIGHT_ONLY"))
    return checker.parser().parse_args(["--manifest", str(manifest), "--pool", str(pool),
                                       "--run-root", str(root / "attempt"), "--commit", "synthetic"])


def write_records(root, records):
    root.mkdir()
    (root / "scores.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


class ShardPreflightTests(unittest.TestCase):
    def test_exact_four_candidate_scores_and_record_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"identity": f"identity-{i}", "scores": [-1., -2., -3., -4.],
                        "subject": "subject", "doc_index": i} for i in range(2)]
            write_records(root / "ref", records)
            write_records(root / "first", [{**records[0], "canonical_index": 0}])
            write_records(root / "second", [{**records[1], "canonical_index": 368}])
            result = checker.compare_roots(root / "ref", [root / "second", root / "first"])
            self.assertEqual(result, {"exact": True, "sample_count": 2, "candidate_count": 8, "max_abs_error": 0.0})
            changed = {**records[1], "scores": [-1., -2., -3., -4.0000001]}
            (root / "second/scores.jsonl").write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "differ from unsharded"):
                checker.compare_roots(root / "ref", [root / "first", root / "second"])

    def test_missing_overlap_and_nonfinite_scores_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"identity": "one", "scores": [-1., -2., -3., -4.]}
            write_records(root / "ref", [record, {**record, "identity": "two"}])
            write_records(root / "shard", [record])
            with self.assertRaisesRegex(ValueError, "identity union"):
                checker.compare_roots(root / "ref", [root / "shard"])
            with self.assertRaisesRegex(ValueError, "overlapping"):
                checker.compare_roots(root / "ref", [root / "shard", root / "shard"])
            (root / "shard/scores.jsonl").write_text(json.dumps({**record, "scores": [-1., -2., -3., float("nan")]}))
            with self.assertRaises(ValueError):
                checker.score_records(root / "shard")

    def test_bounded_validation_plan_and_no_site_packages_help_or_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = inputs(root)
            plan = checker.preflight_plan(args)
            self.assertEqual(plan["indices"], [0, 368])
            self.assertEqual(plan["cell_ids"], list(checker.CELL_IDS))
            self.assertEqual(plan["producer_process_count"], 9)
            env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = ["bash", str(REPO / "scripts/loopscope/phase10_shard_preflight.sbatch"),
                       str(REPO), sys.executable, str(args.manifest), str(args.pool), str(args.run_root), "synthetic"]
            result = subprocess.run(command, env={**env, "DRY_RUN": "1"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(args.run_root.exists())
            subset = root / "indices.json"
            write_json_once(subset, [0])
            args.preflight_indices = subset
            with self.assertRaisesRegex(ValueError, "exactly two"):
                checker.preflight_plan(args)

    def test_sequential_runner_uses_real_verifier_and_fresh_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = inputs(root)
            args.cell_id = checker.CELL_IDS[1]
            rows = validation_fixture()["rows"]
            cell = next(cell for cell in checker.read_json(args.manifest)["cells"] if cell["cell_id"] == args.cell_id)
            producer_calls = []
            execute = checker.execute

            def synthetic_producer(command, env, log_path):
                if command[0] != "bash":
                    return execute(command, env, log_path)
                producer_calls.append(command)
                attempt = Path(command[8])
                attempt.mkdir()
                shard_path = env.get("PHASE10_SHARD_INDICES")
                indices = checker.read_json(shard_path or env["PHASE10_PREFLIGHT_INDICES"])
                metadata = {"cell": cell, "acquisition_cell": cell, "engineering_check": None,
                            "dataset": "mmlu", "scope": "PREFLIGHT_ONLY", "pool": str(args.pool),
                            "manifest": str(args.manifest), "indices": indices, "source_commit": args.commit,
                            "target_gold_loaded": False,
                            "execution_selection": {"kind": "canonical_shard" if shard_path else "scope_selection",
                            "shard_id": env.get("PHASE10_SHARD_ID"), "indices": indices, "canonical_pool_count": len(rows)}}
                write_json_once(attempt / "command_args.json", metadata)
                write_json_once(attempt / "env.json", {"model_revision": cell["revision"],
                    "model_dtype": "torch." + cell["dtype"], "dataset": "mmlu", "source_commit": args.commit,
                    "versions": {"lm_eval": "0.4.11"}})
                k, count = cell["k"], len(indices)
                runtime = {"callback_by_t": {str(t): count for t in range(k)},
                           "applied_by_t": {str(t): count if t else 0 for t in range(k)},
                           "direction_fit_by_t": {str(t): count if not t else 0 for t in range(k)},
                           "direction_used_fit_by_t": {str(t): count if t else 0 for t in range(k)},
                           "direction_schedule": direction_schedule("fixed_t0", k), "model_context_count": count,
                           "t0_mutation_count": 0, "nonanswer_mutation_count": 0}
                write_json_once(attempt / "summary.json", {"status": "SCORES_COMPLETE", "metadata": metadata,
                    "count": count, "target_gold_loaded": False, "positions_and_runtime": runtime})
                records = [{**score_record(rows[i], "mmlu", [-1., -2., -3., -4.],
                    [{"continuation_length": 1}] * 4), "canonical_index": i} for i in indices]
                (attempt / "scores.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

            with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "debug"}), \
                    patch.object(checker, "execute", side_effect=synthetic_producer):
                self.assertEqual(checker.run(args), 0)
            self.assertEqual([Path(command[8]).name for command in producer_calls], ["reference", "shard-0", "shard-1"])
            result = checker.read_json(args.run_root / "SHARD_PREFLIGHT_VERIFIED.json")
            self.assertEqual(result["reports"][0]["max_abs_error"], 0)
            self.assertFalse(result["target_gold_loaded"])
            self.assertEqual(checker.read_json(Path(result["reports"][0]["closures"]["merged"]))["status"],
                             "FULL_PHASE10_SCORE_CELLS_CLOSED")


if __name__ == "__main__":
    unittest.main()
