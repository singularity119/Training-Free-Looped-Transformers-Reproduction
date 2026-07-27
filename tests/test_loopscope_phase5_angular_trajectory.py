from __future__ import annotations

import inspect
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tflt.loopscope.phase5_angular_trajectory import (
    GateIAngularTrajectoryError,
    _capture_raw_final_boundary,
    _plot_with_gnuplot,
    _validate_plot_script_state,
    acquire,
    aggregate_records,
    analyze_data,
    angular_distance,
    cosine_to_angle,
    figure_source_rows,
    kneedle_direction,
    linear_quantile,
    plot_figures,
    subject_stratified_bootstrap,
    validate_sanitized_record,
)


def model(layers: int = 28):
    return {
        "repo": "Qwen/example",
        "revision": "a" * 40,
        "decoder_layers": layers,
    }


def record(layers: int = 28, subject: str = "s0", offset: float = 0.0):
    return {
        "schema_version": "loopscope.phase5.gate-i-angular-distance-record.v1",
        "model_key": "qwen3_1p7b" if layers == 28 else "qwen3_4b",
        "model_repo": "Qwen/example",
        "model_revision": "a" * 40,
        "identity": {"task": subject, "doc_id": subject + ":0", "doc_hash": "b" * 64},
        "subject": subject,
        "split": "validation",
        "prompt_sha256": "c" * 64,
        "renderer_provenance": {
            "dataset_repo": "cais/mmlu",
            "dataset_revision": "d" * 40,
            "renderer_id": "natural",
            "renderer_source_sha256": "d" * 64,
            "source_projection_sha256": "e" * 64,
            "render_contract_sha256": "f" * 64,
            "source_manifest_sha256": "1" * 64,
            "pool_manifest_sha256": "2" * 64,
        },
        "answer_position": 7,
        "sequence_length": 8,
        "loop_insertions": 0,
        "boundary_count": layers + 1,
        "transition_count": layers,
        "adjacent_angular_distance": [
            offset + (index + 1) / (1000.0 + layers) for index in range(layers)
        ],
        "raw_residual_before_final_norm": True,
    }


class FakeTensor:
    def __init__(self, value: float):
        self.value = float(value)
        self.shape = (1, 1, 1)

    def __sub__(self, other):
        return FakeTensor(self.value - other.value)

    def detach(self):
        return self

    def abs(self):
        return FakeTensor(abs(self.value))

    def max(self):
        return self

    def to(self, **_kwargs):
        return self

    def cpu(self):
        return self

    def __float__(self):
        return self.value


class FakeHandle:
    def __init__(self, owner, hook):
        self.owner = owner
        self.hook = hook

    def remove(self):
        self.owner.hooks.remove(self.hook)


class FakeNorm:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self, hook)

    def __call__(self, value):
        for hook in list(self.hooks):
            hook(self, (value,))
        return FakeTensor(value.value * 2.0)


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeTorch:
    float64 = "float64"

    @staticmethod
    def inference_mode():
        return FakeInferenceMode()

    @staticmethod
    def equal(left, right):
        return left.value == right.value


class FakeOutputs:
    def __init__(self, post):
        self.hidden_states = (FakeTensor(0.0), post)


class FixedLocator:
    def __init__(self, x, _y, **_kwargs):
        self.all_knees = {x[2]}


class MultipleLocator:
    def __init__(self, x, _y, **_kwargs):
        self.all_knees = {x[1], x[2]}


class UnstableLocator:
    calls = 0

    def __init__(self, x, _y, **_kwargs):
        type(self).calls += 1
        self.all_knees = {x[2] if type(self).calls == 1 else x[1]}


class AngularMetricTests(unittest.TestCase):
    def test_identical_orthogonal_antipodal(self):
        self.assertEqual(angular_distance([1, 0], [1, 0]), 0.0)
        self.assertEqual(angular_distance([1, 0], [0, 1]), 0.5)
        self.assertEqual(angular_distance([1, 0], [-1, 0]), 1.0)

    def test_clamp_zero_norm_and_nonfinite(self):
        self.assertEqual(cosine_to_angle(1.0 + 1e-12), 0.0)
        self.assertEqual(cosine_to_angle(-1.0 - 1e-12), 1.0)
        with self.assertRaises(GateIAngularTrajectoryError):
            angular_distance([0, 0], [1, 0])
        with self.assertRaises(GateIAngularTrajectoryError):
            angular_distance([math.nan, 0], [1, 0])
        with self.assertRaises(GateIAngularTrajectoryError):
            cosine_to_angle(math.inf)


class RawFinalBoundaryTests(unittest.TestCase):
    def test_final_norm_prehook_count_closure_and_removal(self):
        norm = FakeNorm()
        raw = FakeTensor(3.0)
        outputs, captured, calls, maximum = _capture_raw_final_boundary(
            norm,
            lambda: FakeOutputs(norm(raw)),
            torch_module=FakeTorch,
        )
        self.assertIs(captured, raw)
        self.assertEqual(calls, 1)
        self.assertEqual(maximum, 0.0)
        self.assertEqual(outputs.hidden_states[-1].value, 6.0)
        self.assertEqual(norm.hooks, [])


class RecordSchemaTests(unittest.TestCase):
    def test_transition_lengths_28_and_36(self):
        self.assertEqual(len(validate_sanitized_record(record(28), model(28))[
            "adjacent_angular_distance"
        ]), 28)
        self.assertEqual(len(validate_sanitized_record(record(36), model(36))[
            "adjacent_angular_distance"
        ]), 36)

    def test_forbidden_persistence_is_rejected(self):
        value = record()
        value["logits"] = [1.0]
        with self.assertRaises(GateIAngularTrajectoryError):
            validate_sanitized_record(value, model())


class AggregateTests(unittest.TestCase):
    def test_sd_quantile_and_bootstrap_deterministic_recompute(self):
        records = [
            record(subject="s0", offset=0.00),
            record(subject="s0", offset=0.02),
            record(subject="s1", offset=0.04),
            record(subject="s1", offset=0.06),
        ]
        first = subject_stratified_bootstrap(records, 40, 20260716)
        second = subject_stratified_bootstrap(records, 40, 20260716)
        self.assertEqual(first, second)
        aggregate = aggregate_records(records, first)
        self.assertGreater(aggregate[0]["sample_sd_ddof_1"], 0.0)
        values = [r["adjacent_angular_distance"][0] for r in records]
        self.assertEqual(aggregate[0]["median"], linear_quantile(values, 0.5))
        self.assertEqual(aggregate[0]["q25"], linear_quantile(values, 0.25))
        self.assertEqual(aggregate[0]["q75"], linear_quantile(values, 0.75))


class KneedleTests(unittest.TestCase):
    @staticmethod
    def decreasing_convex(length=8):
        return [(1.0 - index / length) ** 2 for index in range(length)]

    def test_forward_and_reverse_mapping(self):
        curve = self.decreasing_convex()
        boot = [curve] * 10
        forward = kneedle_direction(
            curve, boot, reverse=False, locator_factory=FixedLocator
        )
        self.assertEqual(forward["status"], "STABLE_KNEE")
        self.assertEqual(forward["mapped_transition"], 2)
        self.assertEqual(forward["candidate_boundary"], 3)
        reverse = kneedle_direction(
            list(reversed(curve)), [list(reversed(curve))] * 10,
            reverse=True, locator_factory=FixedLocator
        )
        self.assertEqual(reverse["status"], "STABLE_KNEE")
        self.assertEqual(reverse["mapped_transition"], len(curve) - 3)
        self.assertEqual(reverse["candidate_boundary"], len(curve) - 3)

    def test_nonunique_shape_invalid_and_unstable(self):
        curve = self.decreasing_convex()
        boot = [curve] * 10
        nonunique = kneedle_direction(
            curve, boot, reverse=False, locator_factory=MultipleLocator
        )
        self.assertEqual(nonunique["status"], "NO_STABLE_KNEE_NON_UNIQUE")
        invalid = kneedle_direction(
            list(reversed(curve)), [list(reversed(curve))] * 10,
            reverse=False, locator_factory=FixedLocator
        )
        self.assertEqual(invalid["status"], "NO_STABLE_KNEE_SHAPE_ASSUMPTION")
        UnstableLocator.calls = 0
        unstable = kneedle_direction(
            curve, boot, reverse=False, locator_factory=UnstableLocator
        )
        self.assertEqual(unstable["status"], "NO_STABLE_KNEE_UNSTABLE")
        self.assertEqual(unstable["bootstrap_frequency"], 0.0)


class FigureSourceTests(unittest.TestCase):
    def test_exact_model_and_transition_membership(self):
        models = {
            "qwen3_1p7b": {"repo": "Qwen/1.7B", "decoder_layers": 28},
            "qwen3_4b": {"repo": "Qwen/4B", "decoder_layers": 36},
        }
        aggregates = {
            key: [
                {
                    "transition_index": index,
                    "normalized_depth": index / value["decoder_layers"],
                    "mean": 0.1,
                    "bootstrap_ci95_low": 0.09,
                    "bootstrap_ci95_high": 0.11,
                }
                for index in range(value["decoder_layers"])
            ]
            for key, value in models.items()
        }
        rows = figure_source_rows(aggregates, models)
        self.assertEqual(len(rows), 64)
        self.assertEqual({row["model_key"] for row in rows}, set(models))
        broken = dict(aggregates)
        broken["qwen3_1p7b"] = broken["qwen3_1p7b"][:-1]
        with self.assertRaises(GateIAngularTrajectoryError):
            figure_source_rows(broken, models)

    def test_each_render_resets_panel_a_xrange_and_arrows(self):
        models = {
            "qwen3_1p7b": {
                "repo": "Qwen/1.7B",
                "decoder_layers": 28,
                "color": "#C98022",
            },
            "qwen3_4b": {
                "repo": "Qwen/4B",
                "decoder_layers": 36,
                "color": "#3775BA",
            },
        }
        card = {
            "models": models,
            "figure": {"base": "plot-state-regression"},
        }
        aggregates = {
            key: [
                {
                    "transition_index": index,
                    "normalized_depth": index / value["decoder_layers"],
                    "mean": 0.1,
                    "bootstrap_ci95_low": 0.09,
                    "bootstrap_ci95_high": 0.11,
                }
                for index in range(value["decoder_layers"])
            ]
            for key, value in models.items()
        }
        source_rows = figure_source_rows(aggregates, models)
        direction = {
            "status": "STABLE_KNEE",
            "mapped_transition": 2,
            "quadratic_coefficients": [0.1, -0.2, 0.3],
        }
        knees = {
            key: {
                "model_status": "STABLE_INTERVAL",
                "forward": dict(direction),
                "reverse": dict(direction),
            }
            for key in models
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_gnuplot(arguments, **_kwargs):
                for suffix in ("svg", "pdf", "png"):
                    (root / ("plot-state-regression." + suffix)).write_text(
                        "nonempty", encoding="utf-8"
                    )
                return mock.Mock(returncode=0, stderr="")

            with mock.patch(
                "tflt.loopscope.phase5_angular_trajectory.subprocess.run",
                side_effect=fake_gnuplot,
            ):
                _plot_with_gnuplot(root, card, source_rows, knees)

            script = (root / "plot_inputs" / "plot.gp").read_text(encoding="utf-8")

        _validate_plot_script_state(script)
        blocks = script.split("set title 'A  Raw transition index' left")[1:]
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            panel_a = block.split("set title 'B  Normalized depth' left", 1)[0]
            self.assertLess(panel_a.index("set autoscale x"), panel_a.index("unset arrow"))
            self.assertLess(panel_a.index("unset arrow"), panel_a.index("set arrow"))
            self.assertLess(panel_a.index("set arrow"), panel_a.index("plot "))


class DataThenPlotOrderingTests(unittest.TestCase):
    def test_data_analysis_does_not_render_figures(self):
        source = inspect.getsource(analyze_data)
        self.assertNotIn("_plot_with_gnuplot", source)
        self.assertIn('"plot_executed": False', source)

    def test_plotter_does_not_read_raw_identity_records(self):
        source = inspect.getsource(plot_figures)
        self.assertNotIn("load_jsonl", source)
        self.assertNotIn("sanitized_angular_distance_records", source)


class ReceiptRuntimeRegressionTests(unittest.TestCase):
    def test_receipt_serialization_uses_runtime_torch_without_module_global(self):
        self.assertNotIn("torch", acquire.__globals__)

        class FakeCuda:
            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def get_device_name(_index):
                return "fake-a800"

        class FakeTorchRuntime:
            __version__ = "runtime-torch"
            cuda = FakeCuda()

        class FakeNative:
            pass

        class FakeCausal:
            def __init__(self):
                self.model = FakeNative()

        class FakeNorm:
            pass

        members = [
            {
                "ordinal": index,
                "identity": {
                    "task": "task",
                    "doc_id": "task:validation:%d" % index,
                    "doc_hash": ("%064x" % (index + 1)),
                },
                "subject": "s%d" % (index // 2),
                "split": "validation",
                "prompt_sha256": "%064x" % (index + 10),
            }
            for index in range(4)
        ]
        pool = [dict(member) for member in members]
        fake_card = {
            "models": {
                "qwen3_1p7b": {
                    "repo": "Qwen/fake",
                    "revision": "a" * 40,
                    "dtype": "float16",
                    "decoder_layers": 28,
                    "snapshot": "/fake",
                }
            }
        }
        fake_record = record()
        evidence = {
            "identity": fake_record["identity"],
            "answer_position": 7,
            "answer_position_is_last_non_padding": True,
            "sequence_length": 8,
            "final_norm_prehook_calls": 1,
            "final_norm_closure_max_abs": 0.0,
            "boundary_count": 29,
            "transition_count": 28,
        }
        runtime = (FakeTorchRuntime, object(), FakeCausal(), FakeNorm())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory.AUTHORIZED_RUN_ROOT",
                    root,
                ),
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory.load_card",
                    return_value=fake_card,
                ),
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory.load_json",
                    return_value={
                        "records": members,
                        "membership_sha256": "b" * 64,
                    },
                ),
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory._pool_and_manifests",
                    return_value=(pool, {}, {}),
                ),
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory._load_runtime",
                    return_value=runtime,
                ),
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory._acquire_one",
                    return_value=(fake_record, evidence),
                ),
                mock.patch(
                    "tflt.loopscope.phase5_angular_trajectory.implementation_hashes",
                    return_value={"module": "c" * 64},
                ),
            ):
                receipt = acquire(
                    Path("unused-card"),
                    root,
                    "qwen3_1p7b",
                    "smoke",
                    smoke_attempt=2,
                )
        self.assertEqual(receipt["runtime"]["torch"], "runtime-torch")
        self.assertEqual(receipt["runtime"]["device_name"], "fake-a800")


if __name__ == "__main__":
    unittest.main()
