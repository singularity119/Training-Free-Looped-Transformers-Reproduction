import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from tflt.loopscope.phase10_accuracy import (
    check_position_metadata, engineering_cell, finite_scores, load_pool, score_record, write_json_once,
)
from tflt.loopscope.phase10_panel import score_manifest


REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "loopscope" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(index):
    identity = f"allenai/ai2_arc@210d026faf9955653af8916fad021475a3f00453:ARC-Challenge:test:{index}:synthetic"
    return {"identity":identity, "sample_id": identity, "task":"arc_challenge", "split":"test",
        "source_index":index, "question":"question", "choices":["long answer", "short", "third"],
        "choice_labels":["1", "2", "3"], "prompt":"Question: question\nAnswer:",
        "fewshot_sample_ids":[f"train-{i}" for i in range(25)],
        "tokenization": {model:{"candidate_prefix_consistent":True, "context_length":10}
            for model in ("Qwen/Qwen3-4B-Base", "Qwen/Qwen3-1.7B-Base")}}


class AccuracyTests(unittest.TestCase):
    def test_engineering_controls_cannot_enter_formal(self):
        cell={"arm":"Online","cell_id":"synthetic","k":2,"direction_policy":"fixed_t0","strength":.5}
        zero=engineering_cell(cell,"zero-strength","PREFLIGHT_ONLY")
        self.assertEqual(zero["strength"],0)
        self.assertEqual(zero["cell_id"],"synthetic-engineering-zero-strength")
        lag=engineering_cell(cell,"k2-policy","PREFLIGHT_ONLY")
        self.assertEqual(lag["direction_policy"],"lag1")
        self.assertEqual(cell["direction_policy"],"fixed_t0")
        for check in ("zero-strength","k2-policy"):
            with self.assertRaises(ValueError):
                engineering_cell(cell,check,"FORMAL_TEST")
        with self.assertRaises(ValueError):
            engineering_cell({**cell,"arm":"Loop"},"zero-strength","PREFLIGHT_ONLY")
        current_zero=engineering_cell({**cell,"direction_policy":"current_t"},"zero-strength","PREFLIGHT_ONLY")
        self.assertEqual(current_zero["direction_policy"],"current_t")
        self.assertEqual(current_zero["strength"],0)
        with self.assertRaises(ValueError):
            engineering_cell({**cell,"direction_policy":"current_t"},"k2-policy","PREFLIGHT_ONLY")
    def test_all_candidates_and_char_lengths(self):
        row = sample(0)
        record = score_record(row, "arc_challenge", [(-3.,False),(-2.,True),(-7.,False)],
            [{"continuation_length":4},{"continuation_length":2},{"continuation_length":3}])
        self.assertEqual(record["scores"], [-3.,-2.,-7.])
        self.assertEqual(record["choice_text_lengths"], [11,5,5])
        self.assertEqual(record["continuation_token_lengths"], [4,2,3])
        with self.assertRaises(ValueError):
            finite_scores([1,2],3)
        with self.assertRaises(ValueError):
            finite_scores([1,float("nan"),2],3)

    def test_mask_checks_actual_token_ids_and_retained_boundary(self):
        positions = [{"position":1, "valid_positions":[0,1], "effective_valid_positions":[0,1],
            "actual_tokens":[10,11,12,13], "prefix_tokens":[10,11], "input_length":4,
            "continuation_length":3, "left_truncated_tokens":0}]
        tokenizer = SimpleNamespace(all_special_ids=[1],pad_token_id=None)
        self.assertEqual(check_position_metadata(positions,tokenizer)["multitoken_requests"],1)
        tokenizer.all_special_ids = [11]
        with self.assertRaises(ValueError):
            check_position_metadata(positions,tokenizer)

    def test_input_and_result_write_once_gold_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/"pool.jsonl"
            pool.write_text(json.dumps(sample(0))+"\n")
            self.assertEqual(len(load_pool(pool,"arc_challenge","PREFLIGHT_ONLY")),1)
            pool.write_text(json.dumps({**sample(0), "answerKey":"2"})+"\n")
            with self.assertRaises(ValueError):
                load_pool(pool,"arc_challenge","PREFLIGHT_ONLY")
            output=Path(directory)/"result.json"
            write_json_once(output,{"target_gold_loaded":False})
            with self.assertRaises(FileExistsError):
                write_json_once(output,{})

    def test_runner_help_dry_run_without_site_packages_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pool=root/"pool.jsonl"
            pool.write_text("\n".join(json.dumps(sample(i)) for i in range(2))+"\n")
            manifest_path=root/"manifest.json"
            manifest=score_manifest("arc_challenge",pool,"PREFLIGHT_ONLY")
            write_json_once(manifest_path,manifest)
            import os
            env={**os.environ,"PYTHONPATH":str(REPO/"src")}
            for script in ("run_phase10_accuracy", "verify_phase10_scores"):
                result=subprocess.run([sys.executable,"-S",str(REPO/"scripts"/"loopscope"/(script+".py")),"--help"],
                    capture_output=True,text=True,env=env)
                self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([sys.executable,"-S",str(REPO/"scripts"/"loopscope"/"run_phase10_accuracy.py"),
                "--dataset","arc_challenge","--pool",str(pool),"--manifest",str(manifest_path),
                "--cell-id",manifest["cells"][0]["cell_id"],"--run-root",str(root/"no_run_created"),
                "--scope","PREFLIGHT_ONLY","--commit","synthetic","--dry-run"],
                capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/"no_run_created").exists())

    def test_complete_scores_closure_and_reject_partial_or_outcomes(self):
        verifier=load_script("verify_phase10_scores")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pool=root/"pool.jsonl"
            rows=[sample(i) for i in range(2)]
            pool.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            manifest_path=root/"manifest.json"
            manifest=score_manifest("arc_challenge",pool,"PREFLIGHT_ONLY")
            cell=manifest["cells"][0]
            write_json_once(manifest_path,manifest)
            attempt=root/"attempt"
            attempt.mkdir()
            metadata={"cell":cell,"dataset":"arc_challenge","scope":"PREFLIGHT_ONLY","pool":str(pool),
                "manifest":str(manifest_path),"indices":[0,1],"source_commit":"synthetic","target_gold_loaded":False}
            write_json_once(attempt/"command_args.json",metadata)
            write_json_once(attempt/"env.json",{"model_revision":cell["revision"], "model_dtype":"torch."+cell["dtype"],
                "dataset":"arc_challenge","source_commit":"synthetic","versions":{"lm_eval":"0.4.11"}})
            write_json_once(attempt/"summary.json",{"status":"SCORES_COMPLETE","metadata":metadata,"count":2,
                "target_gold_loaded":False,"positions_and_runtime":{"callback_by_t":{},"applied_by_t":{},
                    "t0_mutation_count":0,"nonanswer_mutation_count":0}})
            records=[score_record(row,"arc_challenge",[-1,-2,-3],[{"continuation_length":2}]*3) for row in rows]
            scores=attempt/"scores.jsonl"
            scores.write_text("\n".join(json.dumps(row) for row in records)+"\n")
            closure,aligned=verifier.verify([attempt],manifest_path,pool,"PREFLIGHT_ONLY")
            self.assertEqual(closure["status"],"SCORE_ROOTS_CLOSED")
            self.assertFalse(closure["target_gold_loaded"])
            self.assertEqual(aligned["cells"][cell["cell_id"]],[[-1.,-2.,-3.]]*2)
            for bad in ([records[0]], [{**records[0],"correct":True},records[1]],
                        [{**records[0],"scores":[-1,-2]},records[1]]):
                scores.write_text("\n".join(json.dumps(row) for row in bad)+"\n")
                with self.assertRaises(ValueError):
                    verifier.verify([attempt],manifest_path,pool,"PREFLIGHT_ONLY")

    def test_current_t_closure_requires_t0_application_and_current_fit_evidence(self):
        verifier=load_script("verify_phase10_scores")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pool=root/"pool.jsonl"
            rows=[sample(i) for i in range(2)]
            pool.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            manifest_path=root/"manifest.json"
            manifest=score_manifest("arc_challenge",pool,"PREFLIGHT_ONLY")
            cell=next(cell for cell in manifest["cells"] if cell["direction_policy"]=="current_t" and cell["k"]==2)
            write_json_once(manifest_path,manifest)
            attempt=root/"attempt"
            attempt.mkdir()
            metadata={"cell":cell,"acquisition_cell":cell,"engineering_check":None,
                "dataset":"arc_challenge","scope":"PREFLIGHT_ONLY","pool":str(pool),
                "manifest":str(manifest_path),"indices":[0,1],"source_commit":"synthetic","target_gold_loaded":False}
            write_json_once(attempt/"command_args.json",metadata)
            write_json_once(attempt/"env.json",{"model_revision":cell["revision"],"model_dtype":"torch."+cell["dtype"],
                "dataset":"arc_challenge","source_commit":"synthetic","versions":{"lm_eval":"0.4.11"}})
            stats={"callback_by_t":{"0":6,"1":6},"applied_by_t":{"0":6,"1":6},
                "direction_fit_by_t":{"0":6,"1":6},"direction_used_fit_by_t":{"0":6,"1":6},
                "direction_schedule":[[0,0,0],[1,1,1]],"model_context_count":6,
                "t0_mutation_count":6,"unexpected_t0_mutation_count":0,"nonanswer_mutation_count":0}
            summary={"status":"SCORES_COMPLETE","metadata":metadata,"count":2,"target_gold_loaded":False,
                "positions_and_runtime":stats}
            write_json_once(attempt/"summary.json",summary)
            records=[score_record(row,"arc_challenge",[-1,-2,-3],[{"continuation_length":2}]*3) for row in rows]
            (attempt/"scores.jsonl").write_text("\n".join(json.dumps(row) for row in records)+"\n")
            closure,_=verifier.verify([attempt],manifest_path,pool,"PREFLIGHT_ONLY")
            self.assertEqual(closure["counts_by_cell"][cell["cell_id"]],2)
            for invalid in ({**stats,"applied_by_t":{"0":0,"1":6}},
                            {**stats,"direction_fit_by_t":{"0":6,"1":0}},
                            {**stats,"direction_schedule":[[0,None,0],[1,0,None]]}):
                (attempt/"summary.json").write_text(json.dumps({**summary,"positions_and_runtime":invalid}))
                with self.assertRaises(ValueError):
                    verifier.verify([attempt],manifest_path,pool,"PREFLIGHT_ONLY")


if __name__ == "__main__":
    unittest.main()
