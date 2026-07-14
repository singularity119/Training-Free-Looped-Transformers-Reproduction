import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tflt import eval_runner
from tflt.loopscope.phase2_schema import (
    FULL_IDENTITY_NAMESPACE,
    SchemaError,
    atomic_write_new_json,
    make_hashed_manifest,
    make_identity_manifest,
    make_source_provenance,
    protocol_cell_id,
)


ROOT = Path(__file__).resolve().parents[1]


def identity(index=7):
    return {"task": "mmlu_x", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}


def logged_sample(sample_identity, scores, gold):
    top = max(range(4), key=lambda index: scores[index])
    return {
        "doc_id": sample_identity["doc_id"],
        "doc_hash": sample_identity["doc_hash"],
        "filtered_resps": list(scores),
        "target": gold,
        "metrics": {"acc,none": int(top == gold)},
    }


class Phase2EvalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(
            (ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json").read_text()
        )

    def test_default_loop_config_has_no_collector(self):
        args = argparse.Namespace(
            model="qwen3-1.7b-base", window="12:15", k=2, iteration_mode="block",
            strategy="damped_euler", alpha=1.0, beta=0.0, cache_strategy="last",
            decode_mode="bypass", first_n=None,
        )
        self.assertIsNone(eval_runner._loop_config(args).audit_collector)

    def test_default_command_args_and_results_schema_have_no_phase2_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "eval"
            with mock.patch.object(eval_runner, "run_lm_eval", return_value={"results": {}}):
                rc = eval_runner.main(
                    ["--model", "qwen3-1.7b-base", "--tasks", "sciq", "--output-dir", str(output)]
                )
            self.assertEqual(rc, 0)
            args = json.loads((output / "command_args.json").read_text(encoding="utf-8"))
            self.assertNotIn("phase2_final_output_manifest", args)
            self.assertFalse((output / "phase2_final_outputs.json").exists())
            self.assertEqual(json.loads((output / "results.json").read_text())["results"], {})

    def test_opt_in_auto_batch_is_allowed_and_sidecar_remains_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text("{}", encoding="utf-8")
            output = Path(tmp) / "out"
            context = {"request": {}, "card": {}, "identity_manifest": {}}
            result = {"results": {"mmlu_x": {"acc,none": 1.0}}}
            with mock.patch.object(
                eval_runner, "_load_phase2_final_output_manifest", return_value=context
            ) as load, mock.patch.object(
                eval_runner, "_build_phase2_final_output_sidecar", return_value={"artifact": "final-only"}
            ), mock.patch.object(
                eval_runner, "verify_full_final_output_artifact", return_value={}
            ), mock.patch.object(eval_runner, "run_lm_eval", return_value=result):
                rc = eval_runner.main(
                    [
                        "--model", "qwen3-1.7b-base", "--tasks", "mmlu_x",
                        "--output-dir", str(output),
                        "--phase2-final-output-manifest", str(request),
                    ]
                )
            self.assertEqual(rc, 0)
            load.assert_called_once()
            self.assertEqual(json.loads((output / "phase2_final_outputs.json").read_text()), {"artifact": "final-only"})
            self.assertEqual(json.loads((output / "results.json").read_text()), result)

    def test_adapter_mode_requires_a_new_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text("{}", encoding="utf-8")
            output = Path(tmp) / "existing"
            output.mkdir()
            context = {"request": {}, "card": {}, "identity_manifest": {}}
            with mock.patch.object(
                eval_runner, "_load_phase2_final_output_manifest", return_value=context
            ), mock.patch.object(eval_runner, "run_lm_eval", return_value={"results": {}}):
                with self.assertRaises(FileExistsError):
                    eval_runner.main(
                        [
                            "--model", "qwen3-1.7b-base", "--revision",
                            self.card["science"]["revision"], "--tasks", "mmlu",
                            "--num-fewshot", "5", "--dtype", "float16",
                            "--output-dir", str(output),
                            "--phase2-final-output-manifest", str(request),
                        ]
                    )

    def test_adapter_request_outside_governed_workspace_fails_before_ref_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside-request.json"
            outside.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(output_dir=str(workspace / "output"))
            with mock.patch.object(eval_runner, "PHASE2_WORKSPACE_ROOT", str(workspace)):
                with self.assertRaises(SchemaError):
                    eval_runner._load_phase2_final_output_manifest(outside, args)

    def test_adapter_argv_matches_active_cell_and_ignores_inactive_baseline_defaults(self):
        baseline = {
            "cell_id": protocol_cell_id("baseline_no_loop", "none", 1, 1.0),
            "protocol": "baseline_no_loop",
            "window": "none",
            "k": 1,
            "alpha": 1.0,
            "step_size": 1.0,
        }
        args = argparse.Namespace(
            model="qwen3-1.7b-base", revision=self.card["science"]["revision"],
            tasks="mmlu", output_dir="unused", limit=None, num_fewshot=5,
            batch_size="auto", dtype="float16", loop=False, window="22:25", k=99,
            iteration_mode="layer", strategy="unused", alpha=9.0, beta=8.0,
            cache_strategy="none", decode_mode="full", first_n=7,
            phase2_final_output_manifest="request.json",
        )
        with self.assertRaisesRegex(ValueError, "reuse cell"):
            eval_runner._validate_phase2_adapter_argv(args, self.card, baseline)

        loop_cell = {
            "cell_id": protocol_cell_id("fixed_step", "12:15", 3, 1.5),
            "protocol": "fixed_step",
            "window": "12:15",
            "k": 3,
            "alpha": 1.5,
            "step_size": 0.5,
        }
        args.loop = True
        args.window = "11:14"
        args.k = 3
        args.iteration_mode = "block"
        args.strategy = "damped_euler"
        args.alpha = 1.5
        args.beta = 0.0
        args.cache_strategy = "last"
        args.decode_mode = "bypass"
        args.first_n = None
        with self.assertRaises(ValueError):
            eval_runner._validate_phase2_adapter_argv(args, self.card, loop_cell)

    def test_adapter_loads_real_attempt_and_receipt_refs_and_binds_actual_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            card_path = root / "card.json"
            atomic_write_new_json(card_path, self.card)
            canonical = [identity(index) for index in range(14042)]
            identity_manifest = make_identity_manifest(
                self.card,
                identity_namespace=FULL_IDENTITY_NAMESPACE,
                split="mmlu_test_full",
                identities=canonical,
                source_provenance=make_source_provenance(
                    identity_namespace=FULL_IDENTITY_NAMESPACE,
                    verification_status="live_verified",
                    source_manifest_path=(
                        "/hpc2hdd/home/xhuang225/workspaces/"
                        "training_free_looped_transformers/runs/"
                        "loopscope-qwen17-mmlu-phase1-20260711-053022/"
                        "control/phase1_run_manifest.json"
                    ),
                    source_manifest_sha256="a" * 64,
                    renderer_subset_sha256="b" * 64,
                    identity_artifacts=[{
                        "path": (
                            "/hpc2hdd/home/xhuang225/workspaces/"
                            "training_free_looped_transformers/runs/"
                            "loopscope-qwen17-mmlu-phase1-20260711-053022/"
                            "gate-e-full/baseline-full/results.json"
                        ),
                        "sha256": "c" * 64,
                    }],
                ),
            )
            atomic_write_new_json(root / "identity.json", identity_manifest)
            cell = {
                "cell_id": protocol_cell_id("fixed_step", "12:15", 3, 1.5),
                "protocol": "fixed_step",
                "window": "12:15",
                "k": 3,
                "alpha": 1.5,
                "step_size": 0.5,
            }
            output = root / "out"
            attempt = make_hashed_manifest(
                {
                    "schema_version": "loopscope.phase2-full-attempt.v2",
                    "artifact_kind": "full_final_output_attempt",
                    "producer_kind": "lm_eval_logged_samples_adapter",
                    "card_manifest_sha256": self.card["manifest_sha256"],
                    "cell_id": cell["cell_id"],
                    "revision": self.card["science"]["revision"],
                    "executor_thread_id": "executor-test",
                    "created_at_utc": "2026-07-14T01:02:03Z",
                    "authorization_root": str(root.resolve()),
                    "output_root": str(output.resolve()),
                }
            )
            atomic_write_new_json(root / "attempt.json", attempt)
            receipt = make_hashed_manifest(
                {
                    "schema_version": "loopscope.phase2-full-receipt.v2",
                    "artifact_kind": "full_final_output_execution_receipt",
                    "producer_kind": "lm_eval_logged_samples_adapter",
                    "card_manifest_sha256": self.card["manifest_sha256"],
                    "cell_id": cell["cell_id"],
                    "revision": self.card["science"]["revision"],
                    "attempt_manifest_sha256": attempt["manifest_sha256"],
                    "executor_thread_id": "executor-test",
                    "created_at_utc": "2026-07-14T01:02:04Z",
                    "authorization_root": str(root.resolve()),
                    "output_root": str(output.resolve()),
                }
            )
            atomic_write_new_json(root / "receipt.json", receipt)
            request = make_hashed_manifest(
                {
                    "schema_version": "loopscope.phase2-final-output-adapter-manifest.v2",
                    "artifact_kind": "full_final_output_adapter_request",
                    "card": {"path": "card.json", "sha256": self.card["manifest_sha256"]},
                    "identity_manifest": {
                        "path": "identity.json",
                        "sha256": identity_manifest["manifest_sha256"],
                    },
                    "cell": cell,
                    "attempt_manifest": {
                        "path": "attempt.json",
                        "sha256": attempt["manifest_sha256"],
                    },
                    "receipt_manifest": {
                        "path": "receipt.json",
                        "sha256": receipt["manifest_sha256"],
                    },
                    "identity_join": "unordered_exact_task_doc_id_doc_hash_to_canonical_manifest",
                    "score_source": "lm_eval_acc_none_raw_per_choice_loglikelihood",
                    "output_filename": "phase2_final_outputs.json",
                }
            )
            request_path = root / "request.json"
            atomic_write_new_json(request_path, request)
            args = argparse.Namespace(
                model="qwen3-1.7b-base", revision=self.card["science"]["revision"],
                tasks="mmlu", output_dir=str(output), limit=None, num_fewshot=5,
                batch_size="auto", dtype="float16", loop=True, window="12:15", k=3,
                iteration_mode="block", strategy="damped_euler", alpha=1.5, beta=0.0,
                cache_strategy="last", decode_mode="bypass", first_n=None,
                phase2_final_output_manifest=str(request_path),
            )
            with mock.patch.object(
                eval_runner, "verify_live_source_provenance", return_value={}
            ) as live_check, mock.patch.object(
                eval_runner,
                "validate_phase2_workspace_output_path",
                side_effect=lambda value, card, context: Path(value).resolve(),
            ), mock.patch.object(
                eval_runner, "PHASE2_WORKSPACE_ROOT", str(root.resolve()),
            ):
                context = eval_runner._load_phase2_final_output_manifest(request_path, args)
            live_check.assert_called_once()
            self.assertEqual(context["attempt"], attempt)
            self.assertEqual(context["receipt"], receipt)

            output.mkdir()
            result = {"results": {"mmlu": {"acc,none": 0.5}}}
            atomic_write_new_json(output / "command_args.json", vars(args))
            atomic_write_new_json(output / "env.json", eval_runner._env_snapshot())
            atomic_write_new_json(
                output / "model_revision.json",
                {
                    "repo_id": self.card["science"]["model"],
                    "model_commit": self.card["science"]["revision"],
                    "tokenizer_commit": self.card["science"]["revision"],
                    "manifest_commit": self.card["science"]["revision"],
                    "match": True,
                },
            )
            atomic_write_new_json(output / "results.json", result)
            with mock.patch.object(
                eval_runner,
                "validate_phase2_workspace_output_path",
                side_effect=lambda value, card, context: Path(value).resolve(),
            ), mock.patch.object(
                eval_runner,
                "validate_contained_path",
                side_effect=lambda value, root, **kwargs: Path(value).resolve(),
            ):
                actual = eval_runner._verify_phase2_adapter_actual_files(
                    output_dir=output, args=args, result=result, context=context
                )
            self.assertEqual(set(actual), {
                "command_sha256", "environment_sha256", "revision_report_sha256", "results_sha256"
            })

    def test_exact_join_restores_canonical_order_and_rejects_extra(self):
        first, second = identity(1), identity(2)
        result = {
            "samples": {
                "mmlu_x": [
                    logged_sample(second, [1, 4, 0, -1], 1),
                    logged_sample(first, [4, 1, 0, -1], 0),
                ]
            }
        }
        rows = eval_runner._join_phase2_logged_samples(result, [first, second])
        self.assertEqual([row["sample_identity"] for row in rows], [first, second])
        extra = json.loads(json.dumps(result))
        extra["samples"]["mmlu_x"].append(logged_sample(identity(3), [4, 1, 0, -1], 0))
        with self.assertRaisesRegex(ValueError, "extra=1"):
            eval_runner._join_phase2_logged_samples(extra, [first, second])

    def test_exact_join_accepts_lm_eval_0_4_11_logged_acc_shape(self):
        expected = identity(1)
        sample = logged_sample(expected, [4, 1, 0, -1], 0)
        sample["metrics"] = ["acc"]
        sample["acc"] = 1.0

        rows = eval_runner._join_phase2_logged_samples(
            {"samples": {"mmlu_x": [sample]}}, [expected]
        )
        self.assertTrue(rows[0]["evaluator_acc_none"])

        ambiguous = dict(sample)
        ambiguous["metrics"] = ["acc", "other"]
        with self.assertRaisesRegex(ValueError, "raw choice/evaluator correctness"):
            eval_runner._join_phase2_logged_samples(
                {"samples": {"mmlu_x": [ambiguous]}}, [expected]
            )

    def test_exact_join_rejects_duplicate_and_missing_four_scores(self):
        expected = [identity(1)]
        sample = logged_sample(expected[0], [4, 1, 0, -1], 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            eval_runner._join_phase2_logged_samples(
                {"samples": {"mmlu_x": [sample, dict(sample)]}}, expected
            )
        broken = dict(sample)
        broken["filtered_resps"] = [4, 1, 0]
        broken["resps"] = []
        with self.assertRaisesRegex(ValueError, "four raw choice"):
            eval_runner._join_phase2_logged_samples({"samples": {"mmlu_x": [broken]}}, expected)


if __name__ == "__main__":
    unittest.main()
