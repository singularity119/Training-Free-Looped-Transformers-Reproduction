import json
import math
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.phase2_fixed_horizon import (
    CARD_ID,
    FixedHorizonStepwiseCollector,
    FixedHorizonError,
    lens_metrics_from_scores,
    register_five_boundary_hooks,
    remove_hooks,
    scalar_step_from_vectors,
    validate_fixed_horizon_card,
)
from tflt.loopscope.phase2_fixed_horizon_analysis import (
    BootstrapEngine,
    _percentile,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/qwen17_mmlu_h2_fixed_horizon_stepwise_v1.json"


class FakeHookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        if self.removed:
            raise AssertionError("hook removed twice")
        self.removed = True


class FakeLayer:
    def __init__(self):
        self.handles = []

    def register_forward_pre_hook(self, callback):
        self.pre = callback
        handle = FakeHookHandle()
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, callback):
        self.forward = callback
        handle = FakeHookHandle()
        self.handles.append(handle)
        return handle


class FakeWrapperHandle:
    def __init__(self):
        self.originals = {index: FakeLayer() for index in range(12, 16)}


class FixedHorizonContractTests(unittest.TestCase):
    def test_card_freezes_exact_science_and_namespace_separation(self):
        card = json.loads(CARD.read_text(encoding="utf-8"))
        validate_fixed_horizon_card(card)
        self.assertEqual(card["card_id"], CARD_ID)
        self.assertEqual(card["science"]["k_values"], [1, 2, 3, 4])
        self.assertEqual(card["science"]["maximum_new_full_run_count"], 0)
        self.assertFalse(card["source_namespaces"]["cross_namespace_sample_join"])

    def test_card_rejects_science_and_runtime_drift(self):
        card = json.loads(CARD.read_text(encoding="utf-8"))
        card["science"]["k_values"].append(5)
        with self.assertRaisesRegex(FixedHorizonError, "science"):
            validate_fixed_horizon_card(card)
        card = json.loads(CARD.read_text(encoding="utf-8"))
        card["executor_thread_id"] = "another-thread"
        with self.assertRaisesRegex(FixedHorizonError, "binding"):
            validate_fixed_horizon_card(card)

    def test_k1_raw_applied_relation_and_not_applicable_repeated_fields(self):
        step = scalar_step_from_vectors(
            [3.0, 4.0], [6.0, 8.0], h=1.0,
            previous_residual=None, native_continuation=None,
        )
        self.assertAlmostEqual(step["raw_residual_norm"], 5.0)
        self.assertAlmostEqual(step["h_scaled_applied_update_norm"], 5.0)
        self.assertIsNone(step["residual_ratio_to_previous"])
        self.assertIsNone(step["adjacent_residual_cosine"])
        self.assertIsNone(step["repeated_step_nca"])

    def test_fixed_horizon_h_scales_raw_without_changing_raw_direction(self):
        step = scalar_step_from_vectors(
            [1.0, 0.0], [1.0, 2.0], h=0.25,
            previous_residual=[0.0, 1.0], native_continuation=[0.0, 3.0],
        )
        self.assertAlmostEqual(step["raw_residual_norm"], 2.0)
        self.assertAlmostEqual(step["h_scaled_applied_update_norm"], 0.5)
        self.assertAlmostEqual(step["residual_ratio_to_previous"], 2.0)
        self.assertAlmostEqual(step["adjacent_residual_cosine"], 1.0)
        self.assertAlmostEqual(step["repeated_step_nca"], 1.0)

    def test_choice_lens_entropy_margin_top1(self):
        value = lens_metrics_from_scores([0.0, 2.0, 1.0, -1.0])
        self.assertEqual(value["top1_index"], 1)
        self.assertAlmostEqual(value["top_margin_raw"], 1.0)
        self.assertAlmostEqual(sum(value["choice_probabilities"]), 1.0)
        self.assertGreater(value["entropy_nats"], 0.0)

    def test_five_hook_handles_are_registered_and_removed(self):
        wrapper = FakeWrapperHandle()
        collector = type("Collector", (), {"hook_entry": object(), "hook_layer": lambda self, i: object()})()
        handles = register_five_boundary_hooks(wrapper, collector)
        self.assertEqual(len(handles), 5)
        proof = remove_hooks(handles)
        self.assertEqual(proof["hook_handles_registered"], 5)
        self.assertEqual(proof["hook_handles_removed"], 5)
        self.assertEqual(proof["hooks_active_after_removal"], 0)
        self.assertTrue(all(handle.removed for handle in handles))

    def test_bootstrap_seed_stream_is_deterministic(self):
        first = BootstrapEngine(replicates=20, seed=20260716).summarize([1.0, 2.0, 3.0, 4.0])
        second = BootstrapEngine(replicates=20, seed=20260716).summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(first, second)
        self.assertEqual(first["seed_stream_origin"], 20260716)

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(_percentile([0.0, 10.0], 0.25), 2.5)

    def test_chinese_atlas_markdown_renders_frozen_tables(self):
        metric = {name: None for name in (
            "raw_relative_activity", "residual_ratio_to_previous", "adjacent_residual_cosine",
            "repeated_step_nca", "h_scaled_applied_relative_magnitude", "raw_entropy_drop_nats",
            "applied_entropy_drop_nats", "raw_call_entropy_flatness_negative_population_std",
            "raw_kl_to_reference_drop_nats", "applied_kl_to_reference_drop_nats",
        )}
        latent = {
            "kwise": [{"k": 1, "steps": [{
                "body_call_t": 0, "tau_entry": 0.0, "tau_applied": 1.0,
                "metrics": metric,
                "cohort_effective_rank": {
                    "entry_effective_rank": 2.0,
                    "raw_exit_effective_rank": 3.0,
                    "applied_state_effective_rank": 3.0,
                },
            }]}]
        }
        absolute = {"kwise": [{
            "k": 1, "accuracy_percent": 62.0, "mean_entropy_nats": 1.0,
            "mean_top_margin_raw": 2.0, "mean_correct_margin_raw": 0.5,
            "mean_js_to_k1_nats": 0.0, "top1_retained_vs_k1_fraction": 1.0,
        }]}
        contrasts = {"contrasts": [{
            "contrast": "k1_to_k2",
            "accuracy": {"delta_acc_pp": 0.1},
            "mechanism": {
                "transitions": {"wrong_to_right": 2, "right_to_wrong": 1, "right_to_right": 5, "wrong_to_wrong": 4},
                "mean_entropy_delta_nats": -0.1,
                "mean_top_margin_delta_raw": 0.2,
                "wrong_overconfidence": {"fraction": 0.25},
            },
        }]}
        atlas = {"headline": {"best_observed_full_accuracy_k": 2}}
        markdown = render_markdown(latent, absolute, contrasts, atlas)
        self.assertIn("不是 selector 的确认性证据", markdown)
        self.assertIn("validation-512", markdown)
        self.assertIn("observed best full K=2", markdown)

    def test_collector_closes_two_step_scalar_only_trace(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch is unavailable")

        def lens(vector):
            scores = vector[:4].tolist()
            return lens_metrics_from_scores(scores)

        collector = FixedHorizonStepwiseCollector(torch_module=torch, k=2, lens=lens)
        identity = {"task": "mmlu_subject", "doc_id": "1", "doc_hash": "a" * 64}
        for sample_index in range(2):
            collector.begin_sample(
                identity={**identity, "doc_id": str(sample_index)},
                answer_position=0,
                native_continuation=torch.tensor([1.0, 0.5, 0.25, 0.125]),
                reference_probabilities=[0.25, 0.25, 0.25, 0.25],
            )
            collector.record("wrapper_forward", {"bypass": False})
            for t in range(2):
                entry = torch.tensor([[[1.0 + sample_index + t, 2.0, 3.0, 4.0]]])
                boundaries = [entry + offset for offset in (0.25, 0.5, 0.75, 1.0)]
                collector.hook_entry(None, (entry,))
                for offset, value in enumerate(boundaries):
                    collector.hook_layer(offset)(None, (), (value,))
                collector.record("body_call", {})
                collector.record_tensor_diff("g_minus_x", entry, boundaries[-1])
            collector.record_tensor_diff("looped_hidden_vs_input", torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
            collector.record("stash_pass", {})
            stash = torch.ones(1, 1, 4)
            collector.hook_entry(None, (stash,))
            for offset in range(4):
                collector.hook_layer(offset)(None, (), (stash,))
            for _ in range(3):
                collector.record("identity_forward", {})
            output = {
                "sample_identity": {**identity, "doc_id": str(sample_index)},
                "choice_labels": ["A", "B", "C", "D"],
                "choice_score_source": "direct_probe_next_token_log_probability_over_frozen_choice_token_ids",
                "raw_choice_scores": [4.0, 3.0, 2.0, 1.0],
                "choice_probabilities": lens_metrics_from_scores([4.0, 3.0, 2.0, 1.0])["choice_probabilities"],
                "probability_dtype": "float32",
                "entropy_nats": lens_metrics_from_scores([4.0, 3.0, 2.0, 1.0])["entropy_nats"],
                "top1_index": 0,
                "top_margin_raw": 1.0,
                "gold_index": None,
                "correctness": None,
                "correct_margin_raw": None,
                "evaluator_acc": None,
            }
            sample = collector.end_sample(output)
            self.assertEqual(sample["body_call_count"], 2)
            self.assertEqual(sample["hook_capture_proof"]["loop_boundary_capture_count"], 10)
            self.assertIsNone(sample["steps"][0]["residual_ratio_to_previous"])
            self.assertIsNotNone(sample["steps"][1]["residual_ratio_to_previous"])
        erank = collector.effective_rank_summary()
        self.assertEqual(len(erank), 2)
        self.assertTrue(all(row["sample_count"] == 2 for row in erank))


if __name__ == "__main__":
    unittest.main()
