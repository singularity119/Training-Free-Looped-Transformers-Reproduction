"""CPU-only entry planning, failure preservation and raw-schema integration."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_accuracy import generation_record
from tflt.loopscope.phase11_analysis import verify_attempt
from tflt.loopscope.phase11_panel import MODEL, REVISION, score_manifest


REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "scripts/loopscope/run_phase11_accuracy.py"
SPEC = importlib.util.spec_from_file_location("phase11_accuracy_entry", ENTRY)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def current_commit():
    git_dir = REPO / ".git"
    if git_dir.is_file():
        git_dir = (REPO / git_dir.read_text().strip().removeprefix("gitdir: ")).resolve()
    ref = (git_dir / "HEAD").read_text().strip().removeprefix("ref: ")
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    if (common / ref).is_file():
        return (common / ref).read_text().strip()
    for line in (common / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise AssertionError("test checkout lacks a resolvable source HEAD")


class EntryTests(unittest.TestCase):
    def fixture(self, directory, dataset="gpqa_main", online=False):
        rows = [{"identity": f"synthetic:{i}", "task": dataset, "split": "synthetic",
                 "subject": "synthetic", "prompt": "Synthetic engineering prompt",
                 "choices": ["triangle", "circle"], "choice_labels": ["A", "B"],
                 "input_ids": [11, 12], "prompt_token_length": 2, "fewshot_sample_ids": []} for i in range(3)]
        pool = {"schema_version": "loopscope.phase11.inputs.v1", "task": dataset,
                "model": MODEL, "model_revision": REVISION, "tokenizer_revision": REVISION,
                "scope": "PREFLIGHT_ONLY", "engineering_synthetic": True,
                "target_gold_loaded": False, "rows": rows, "test_identities": [r["identity"] for r in rows]}
        pool_path, manifest_path = Path(directory) / "pool.json", Path(directory) / "manifest.json"
        pool_path.write_text(json.dumps(pool))
        manifest = score_manifest(dataset, pool_path, "PREFLIGHT_ONLY")
        manifest_path.write_text(json.dumps(manifest))
        cell = next(c for c in manifest["cells"] if c["arm"] == "Online" and c["k"] == 2 and
                    c["direction_policy"] == "fixed_t0" and c["strength"] == .5) if online else manifest["cells"][0]
        arguments = ["--dataset", dataset, "--manifest", str(manifest_path), "--pool", str(pool_path),
                     "--cell-id", cell["cell_id"], "--scope", "PREFLIGHT_ONLY",
                     "--run-root", str(Path(directory) / "fresh"), "--commit", current_commit()]
        return arguments, manifest, pool

    def invoke(self, args):
        env = dict(os.environ)
        env.pop("SLURM_JOB_ID", None)
        env.pop("SLURM_JOB_PARTITION", None)
        env["PYTHONPATH"] = str(REPO / "src")
        return subprocess.run([sys.executable, "-S", str(ENTRY), *args], text=True, capture_output=True,
                              env=env, timeout=30)

    def test_help_and_dry_run_load_no_model_libraries(self):
        result = self.invoke(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--engineering-generate-steps", result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(directory)
            result = self.invoke(args + ["--dry-run"])
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report["model_loading"])
            self.assertFalse(report["target_gold_loaded"])
            self.assertEqual(report["metadata"]["max_new_tokens"], 2048)
            self.assertEqual(report["count"], 3)
            self.assertFalse((Path(directory) / "fresh").exists())

    def test_modulo_shard_and_engineering_alias_remain_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(directory, online=True)
            result = self.invoke(args + ["--shard", "1/2", "--engineering-check", "k2-policy",
                                         "--engineering-generate-steps", "3", "--dry-run"])
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)["metadata"]
            self.assertEqual(metadata["indices"], [1])
            self.assertEqual(metadata["cell"]["direction_policy"], "fixed_t0")
            self.assertEqual(metadata["acquisition_cell"]["direction_policy"], "lag1")
            self.assertEqual(metadata["max_new_tokens"], 3)

    def test_wrong_source_and_no_slurm_stop_before_model_load_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(directory)
            parsed = RUNNER.parser().parse_args(args)
            with patch.object(RUNNER, "load_runtime") as loader:
                parsed.commit = "different-source-revision"
                with self.assertRaisesRegex(ValueError, "source Git HEAD differs"):
                    RUNNER.run(parsed)
                parsed.commit = current_commit()
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, "allocated Slurm job"):
                        RUNNER.run(parsed)
                loader.assert_not_called()
            self.assertFalse(parsed.run_root.exists())

    def test_runtime_load_failure_preserves_failed_record_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(directory)
            parsed = RUNNER.parser().parse_args(args)
            with patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-unit-test", "SLURM_JOB_PARTITION": "debug"}), \
                 patch.object(RUNNER, "load_runtime", side_effect=RuntimeError("synthetic load failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
                    RUNNER.run(parsed)
            summary = json.loads((parsed.run_root / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["count"], summary["failed_stage"]), ("FAILED", 0, "runtime_load"))
            records = [json.loads(line) for line in (parsed.run_root / "records.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["status"], "RUNTIME_FAILED")
            self.assertFalse(summary["target_gold_loaded"])

    def test_fake_raw_producer_matches_actual_preflight_verifier_interface(self):
        cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                               max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
        torch = SimpleNamespace(set_grad_enabled=lambda value: None, manual_seed=lambda value: None, cuda=cuda)
        def acquire(torch, model, tokenizer, adapter, eos, row, index, cell, config, scope, max_tokens, steps):
            return generation_record(row, "Final answer: (A)", [151645], eos, max_tokens, index), {"synthetic_fake": True}
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, pool = self.fixture(directory)
            parsed = RUNNER.parser().parse_args(args)
            with patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-unit-test", "SLURM_JOB_PARTITION": "debug"}), \
                 patch.object(RUNNER, "load_runtime", return_value=(torch, None, None, None, [151645, 151643], {})), \
                 patch.object(RUNNER, "ForwardTiming", return_value=SimpleNamespace(summary=lambda: {}, close=lambda: None)), \
                 patch.object(RUNNER, "acquire_row", side_effect=acquire), redirect_stdout(io.StringIO()):
                self.assertEqual(RUNNER.run(parsed), 0)
            verified = verify_attempt(manifest, pool, parsed.run_root)
            self.assertEqual(verified["status"], "ATTEMPT_RAW_VERIFIED")
            self.assertEqual(verified["records_count"], 3)
            self.assertEqual(verified["indices"], [0, 1, 2])
            self.assertFalse(verified["target_gold_loaded"])


if __name__ == "__main__":
    unittest.main()
