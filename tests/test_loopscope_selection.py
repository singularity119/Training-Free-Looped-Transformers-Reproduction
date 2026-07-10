import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.schema import (
    PROBE_SCHEMA_VERSION,
    canonical_json_bytes,
    manifest_sha256,
)
from tflt.loopscope.selection import (
    DEFAULT_CRITERION,
    PRIMARY_SIGNALS,
    SelectionError,
    cmd_score_windows,
    rank_candidates,
    score_windows,
)


def _probe_envelope(layer_metrics=None, window_metrics=None, examples=None):
    sample_ids = ["s1", "s2"]
    digest = "a" * 64
    windows = list(window_metrics or [])
    renderer = {
        "renderer_entrypoint": "fixture.renderer",
        "lm_eval_version": "0.4.11",
        "renderer_source_sha256": "1" * 64,
        "template_sha256": "2" * 64,
        "render_contract_sha256": "3" * 64,
        "render_sha256": "4" * 64,
        "renderer_manifest_sha256": "5" * 64,
    }
    rendering_records = [
        {"id": "s1", "render_sha256": "6" * 64},
        {"id": "s2", "render_sha256": "7" * 64},
    ]
    render_subset_hash = hashlib.sha256(
        canonical_json_bytes(rendering_records)
    ).hexdigest()
    report = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "git": {
            "root": "/fixture/loopscope-tflt",
            "branch": "loopscope",
            "commit": "abc",
            "dirty": False,
        },
        "model": {
            "alias": "qwen",
            "repo_id": "Qwen/fixture",
            "revision": "r",
            "layer_count": 4,
        },
        "tokenizer": {
            "revision": "r",
            "choice_token_ids": {"A": [1], "B": [2]},
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "versions": {
                "python": "3.9",
                "torch": "fixture",
                "transformers": "fixture",
                "lm_eval": "fixture",
                "tflt": "fixture",
            },
        },
        "probe_pool": {
            "source": "fixture",
            "split": "dev",
            "count": 2,
            "seed": 1,
            "manifest_sha256": digest,
            "source_manifest_sha256": "b" * 64,
            "selected_subset_sha256": digest,
            "source_manifest_count": 2,
            "sample_ids": sample_ids,
            "records": [
                {"id": "s1", "prompt_sha256": "d" * 64},
                {"id": "s2", "prompt_sha256": "e" * 64},
            ],
            "task_group": "mmlu",
            "num_fewshot": 5,
            "uses_target_gold_labels": False,
            "fewshot_answers_present": True,
            "renderer": renderer,
            "render_contract_sha256": "3" * 64,
            "render_contract_sha256": renderer["render_contract_sha256"],
            "rendering_records": rendering_records,
            "source_render_contract_subset_sha256": "8" * 64,
            "selected_render_contract_subset_sha256": render_subset_hash,
        },
        "position_rule": "last_non_padding",
        "layer_metrics": list(layer_metrics or []),
        "window_metrics": windows,
        "window_grid": {
            "manifest_sha256": "c" * 64,
            "layer_count": 4,
            "candidate_windows": [item["window"] for item in windows],
        },
        "examples": list(examples or []),
        "warnings": [],
    }
    pool = report["probe_pool"]
    selected = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": pool["source"],
        "split": pool["split"],
        "count": pool["count"],
        "seed": pool["seed"],
        "sample_ids": pool["sample_ids"],
        "records": pool["records"],
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": renderer,
        "render_contract_sha256": "3" * 64,
        "rendering_records": rendering_records,
        "render_contract_subset_sha256": render_subset_hash,
        "source_manifest_sha256": pool["source_manifest_sha256"],
    }
    pool["manifest_sha256"] = manifest_sha256(selected)
    pool["selected_subset_sha256"] = pool["manifest_sha256"]
    return report


def _summary(value):
    return {"count": 2.0, "mean": value, "median": value, "p90": value}


def _layer_report():
    entropy = [1.0, 0.5, 0.9, 0.85]
    kl = [0.8, 0.2, 0.7, 0.65]
    erank = [2.0, 1.8, 2.1, 2.3]
    layers = [
        {
            "layer_index": index,
            "choice_entropy": _summary(entropy[index]),
            "kl_to_final": _summary(kl[index]),
            "top1_to_final_agreement": _summary(1.0),
            "effective_rank": erank[index],
            "effective_rank_sampling": {
                "representation_space": "answer-position hidden vectors",
                "count": 2,
                "sample_ids": ["s1", "s2"],
            },
        }
        for index in range(4)
    ]
    examples = []
    for sample_id, offset in (("s1", 0.0), ("s2", 0.02)):
        examples.append(
            {
                "id": sample_id,
                "layer_metrics": [
                    {
                        "layer_index": index,
                        "choice_entropy": entropy[index] + offset,
                        "kl_to_final": kl[index] + offset,
                    }
                    for index in range(4)
                ],
            }
        )
    return _probe_envelope(layer_metrics=layers, examples=examples)


def _window_report():
    windows = []
    for window, start, end, activity, contraction in (
        ("0:1", 0, 1, 0.4, 0.5),
        ("2:3", 2, 3, 0.6, 1.2),
    ):
        examples = [
            {
                "sample_id": "s1",
                "valid": True,
                "answer_position_metrics": {"r": activity - 0.05, "q": contraction - 0.05},
            },
            {
                "sample_id": "s2",
                "valid": True,
                "answer_position_metrics": {"r": activity + 0.05, "q": contraction + 0.05},
            },
        ]
        windows.append(
            {
                "window": window,
                "start": start,
                "end": end,
                "valid": True,
                "answer_position": {"r": _summary(activity), "q": _summary(contraction)},
                "all_non_padding_tokens": {
                    "r": _summary(activity),
                    "q": _summary(contraction),
                },
                "sample_count": 2,
                "valid_sample_count": 2,
                "examples": examples,
                "errors": [],
            }
        )
    return _probe_envelope(window_metrics=windows)


class LoopScopeSelectionTest(unittest.TestCase):
    def test_parallel_rankings_keep_all_candidates_and_directions(self):
        report = score_windows(_layer_report(), [_window_report()])
        self.assertEqual(report["manifest_sha256"], manifest_sha256(report))
        self.assertEqual(report["probe_provenance"]["model"]["revision"], "r")
        self.assertEqual(
            report["probe_provenance"]["probe_pool"]["renderer"][
                "render_contract_sha256"
            ],
            "3" * 64,
        )
        self.assertEqual(report["primary_signal"], "r_times_one_minus_q")
        self.assertEqual(report["criterion"]["tie_policy"]["ranking_ties"], "retain_all")
        self.assertEqual(
            report["criterion"]["tie_policy"]["phase_gate_regret"], "worst_case"
        )
        self.assertEqual(len(report["candidates"]), 2)
        self.assertEqual(set(report["rankings"]), set(PRIMARY_SIGNALS))
        self.assertEqual(report["rankings"]["entropy_drop"][0]["window"], "0:1")
        self.assertEqual(report["rankings"]["entropy_flatness"][0]["window"], "2:3")
        self.assertEqual(report["rankings"]["negative_contraction_q"][0]["window"], "0:1")
        self.assertEqual(report["rankings"]["r_times_one_minus_q"][0]["window"], "0:1")

        by_window = {item["window"]: item for item in report["candidates"]}
        self.assertAlmostEqual(by_window["0:1"]["scores"]["negative_contraction_q"], -0.5)
        self.assertAlmostEqual(by_window["0:1"]["scores"]["r_times_one_minus_q"], 0.2)
        self.assertEqual(
            by_window["0:1"]["auxiliary_effective_rank"]["phase"], "contracting"
        )
        self.assertEqual(
            by_window["2:3"]["auxiliary_effective_rank"]["phase"], "expanding"
        )
        self.assertIsNone(
            by_window["0:1"]["auxiliary_effective_rank"]["gate"]["passed"]
        )
        self.assertEqual(len(by_window["0:1"]["sample_scores"]), 2)
        self.assertNotIn("label", json.dumps(report["candidates"]))

    def test_rank_ties_are_explicit_and_average_ranked(self):
        candidates = [
            {"window": "0:1", "scores": {signal: 1.0 for signal in PRIMARY_SIGNALS}},
            {"window": "2:3", "scores": {signal: 1.0 for signal in PRIMARY_SIGNALS}},
            {"window": "4:5", "scores": {signal: 0.0 for signal in PRIMARY_SIGNALS}},
        ]
        ranking = rank_candidates(candidates, "activity_r")
        self.assertEqual([item["rank"] for item in ranking], [1.5, 1.5, 3.0])

    def test_criterion_rejects_unsafe_tie_policy(self):
        criterion = json.loads(json.dumps(DEFAULT_CRITERION))
        criterion["tie_policy"]["phase_gate_regret"] = "display_tiebreak"
        with self.assertRaisesRegex(SelectionError, "tie_policy"):
            score_windows(_layer_report(), [_window_report()], criterion=criterion)

    def test_duplicate_window_probe_is_rejected(self):
        window_report = _window_report()
        with self.assertRaises(SelectionError):
            score_windows(_layer_report(), [window_report, window_report])

    def test_frozen_grid_must_be_fully_scored(self):
        window_report = _window_report()
        window_report["window_grid"]["candidate_windows"].append("4:5")
        with self.assertRaisesRegex(SelectionError, "cover the frozen candidate grid exactly"):
            score_windows(_layer_report(), [window_report])

    def test_render_contract_mismatch_is_rejected(self):
        window_report = _window_report()
        window_report["probe_pool"]["source_render_contract_subset_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            SelectionError, "source_render_contract_subset_sha256"
        ):
            score_windows(_layer_report(), [window_report])

    def test_cli_output_directory_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layer_path = root / "layer.json"
            window_path = root / "window.json"
            output_path = root / "scores"
            layer_path.write_text(json.dumps(_layer_report()), encoding="utf-8")
            window_path.write_text(json.dumps(_window_report()), encoding="utf-8")
            args = argparse.Namespace(
                layer_probe=str(layer_path),
                window_probe=[str(window_path)],
                output_dir=str(output_path),
                criterion=None,
                layer_stat="mean",
                window_stat="median",
            )
            self.assertEqual(cmd_score_windows(args), 0)
            self.assertTrue((output_path / "window_scores.json").is_file())
            self.assertTrue((output_path / "window_scores_summary.md").is_file())
            with self.assertRaises(FileExistsError):
                cmd_score_windows(args)


if __name__ == "__main__":
    unittest.main()
