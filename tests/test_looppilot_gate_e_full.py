import copy
import json
import tempfile
import unittest
from pathlib import Path

from tflt.looppilot.full import (
    GATE_E_CANDIDATE_FIELDS,
    assign_task_shards,
    build_split_rows,
    select_candidate,
    terminal_recommendation,
    validate_gate_e_config,
    validate_leaf_tasks,
    verify_three_arm_rows,
)
from scripts.looppilot.analyze_gate_e import _cluster_ci, _random_null
from scripts.looppilot.analyze_gate_e import main as analyze_main
from scripts.looppilot.verify_gate_e_final import main as verify_final_main
from scripts.looppilot.prepare_gate_e_full import _expand_group
from scripts.looppilot.verify_gate_e_shards import main as verify_shards_main


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/looppilot/gate_e_full.json").read_text())


def sample(arm, key="k0", prediction=0, gold=0, revision="r"):
    return {
        "arm": arm,
        "doc_key": key,
        "task": "mmlu_x",
        "task_version": "1",
        "renderer_hash": "a" * 64,
        "doc_hash": ("b" * 63) + key[-1],
        "occurrence_id": "test:%s" % key[-1],
        "revision_closure_id": revision,
        "subject": "x",
        "gold": gold,
        "prediction": prediction,
        "choice_loglikelihoods": [-1.0, -2.0, -3.0, -4.0],
        "prompt_length": 10,
    }


def decision(key="k0"):
    return {
        "doc_key": key,
        "task": "mmlu_x",
        "task_version": "1",
        "renderer_hash": "a" * 64,
        "doc_hash": ("b" * 63) + key[-1],
        "occurrence_id": "test:%s" % key[-1],
        "action": "LOOP_K2",
        "decision_reason": "always_loop_engineering_control",
        "controller_input_fields": list(GATE_E_CANDIDATE_FIELDS),
        "choice_indices": [0, 1, 2, 3],
        "r0_median": 0.1,
        "c0": 0.2,
        "n0_median": 0.3,
    }


def signals(key="k0"):
    rows = []
    for choice in range(4):
        row = decision(key)
        row.update(
            choice_index=choice,
            question_token_count=8,
            r0_p90=0.2,
            r0_max=0.3,
            n0_p90=0.4,
            n0_max=0.5,
            q1=0.6,
            residual_cosine=0.7,
            operator_body_calls=2,
            k_used=2,
            wrapper_restored=True,
        )
        row.pop("choice_indices")
        rows.append(row)
    return rows


class GateEManifestTest(unittest.TestCase):
    def test_shard_verifier_builds_paths_before_missing_shard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "shards_verified.json"
            with self.assertRaisesRegex(RuntimeError, "missing Gate E shard 0"):
                verify_shards_main(["--run-root", str(root), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_authoritative_tag_expansion_and_fail_closed_inputs(self):
        class NoYaml:
            @staticmethod
            def safe_load(_text):
                raise AssertionError("tag expansion must use TaskManager entry children")

        index = {
            "mmlu_tag": {"type": "tag", "task": ["mmlu_a", "mmlu_nested"], "yaml_path": -1},
            "mmlu_nested": {"type": "tag", "task": ["mmlu_b"], "yaml_path": -1},
            "mmlu_a": {"type": "task", "yaml_path": "/unused/a.yaml"},
            "mmlu_b": {"type": "task", "yaml_path": "/unused/b.yaml"},
        }
        self.assertEqual(_expand_group("mmlu_tag", index, NoYaml, set()), ["mmlu_a", "mmlu_b"])
        for entry in (
            {"type": "tag", "task": [], "yaml_path": -1},
            {"type": "tag", "task": [1], "yaml_path": -1},
            {"type": "unknown", "task": ["mmlu_a"], "yaml_path": -1},
        ):
            changed = dict(index)
            changed["bad"] = entry
            with self.subTest(entry=entry):
                with self.assertRaises(RuntimeError):
                    _expand_group("bad", changed, NoYaml, set())
        recursive = {"loop": {"type": "tag", "task": ["loop"], "yaml_path": -1}}
        with self.assertRaises(RuntimeError):
            _expand_group("loop", recursive, NoYaml, set())

    def test_checked_in_config_is_frozen_and_drift_rejected(self):
        validate_gate_e_config(config())
        for field, value in (("batch_size", "auto"), ("k", 3), ("shard_count", 7)):
            changed = config()
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_gate_e_config(changed)

    def test_authoritative_expansion_is_exactly_57_unique_mmlu_leaves(self):
        tasks = ["mmlu_subject_%02d" % i for i in range(57)]
        self.assertEqual(validate_leaf_tasks(tasks), tuple(sorted(tasks)))
        for bad in (tasks[:-1], tasks + [tasks[0]], tasks[:-1] + ["other"]):
            with self.assertRaises(ValueError):
                validate_leaf_tasks(bad)

    def test_split_and_shards_are_deterministic_and_subject_balanced(self):
        docs = [
            {"doc_key": "subject-a-%d" % i, "task": "mmlu_a"} for i in range(20)
        ] + [{"doc_key": "subject-b-%d" % i, "task": "mmlu_b"} for i in range(20)]
        left = build_split_rows(docs)
        self.assertEqual(left, build_split_rows(copy.deepcopy(docs)))
        self.assertEqual({row["split"] for row in left}, {"development", "holdout"})
        counts = {name: [0, 0] for name in ("mmlu_a", "mmlu_b")}
        for row in left:
            counts[row["task"]][row["bucket"]] += 1
        self.assertTrue(all(all(value > 0 for value in pair) for pair in counts.values()))
        task_counts = {"mmlu_%02d" % i: i + 1 for i in range(57)}
        shards = assign_task_shards(task_counts, 8)
        self.assertEqual(shards, assign_task_shards(dict(reversed(list(task_counts.items()))), 8))
        self.assertEqual(sorted(task for shard in shards for task in shard["tasks"]), sorted(task_counts))


class GateEJoinAndAnalysisTest(unittest.TestCase):
    def test_three_arm_join_self_flip_and_signal_closure(self):
        a = [sample("baseline_a")]
        b = [sample("baseline_b", prediction=1)]
        loop = [sample("always_loop", prediction=0)]
        report = verify_three_arm_rows(a, b, loop, signals(), [decision()])
        self.assertEqual(report["joined_doc_count"], 1)
        self.assertEqual(report["self_flip_count"], 1)
        self.assertTrue(report["forbidden_fields_absent"])
        for changed in (
            (a + a, b, loop, signals(), [decision()]),
            (a, b, [sample("always_loop", revision="bad")], signals(), [decision()]),
            (a, b, loop, signals()[:-1], [decision()]),
        ):
            with self.assertRaises(ValueError):
                verify_three_arm_rows(*changed)

    def test_candidate_tie_break_and_holdout_freeze_contract(self):
        rows = []
        for index in range(8):
            rows.append(
                {
                    "doc_key": str(index),
                    "split": "development",
                    "r0_median": float(index),
                    "c0": float(index),
                    "n0_median": float(index),
                    "baseline_correct": index < 4,
                    "loop_correct": index >= 4,
                }
            )
        selected, grid = select_candidate(rows)
        self.assertEqual(len(grid), 24)
        self.assertEqual(selected["signal"], "r0_median")
        self.assertEqual(selected["direction"], "BASELINE_IF_LOW")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_policy.json"
            path.write_text(json.dumps(selected))
            with self.assertRaises(FileExistsError):
                path.open("x")

    def test_terminal_routing_order(self):
        self.assertEqual(terminal_recommendation({"technical_closed": False}), "BLOCK")
        self.assertEqual(
            terminal_recommendation({"technical_closed": True, "assessable": False}),
            "PIVOT_TO_PHASE3",
        )
        flags = {
            "technical_closed": True,
            "assessable": True,
            "pass_to_online": False,
            "subject_capture_ge_80pct": True,
            "hidden_capture_le_subject": True,
        }
        self.assertEqual(terminal_recommendation(flags), "PIVOT_TO_COARSE")
        flags["pass_to_online"] = True
        self.assertEqual(terminal_recommendation(flags), "PASS_TO_ONLINE")

    def test_bootstrap_and_matched_random_null_are_deterministic(self):
        rows = []
        actions = []
        for index in range(12):
            rows.append(
                {
                    "subject": "s%d" % (index % 3),
                    "baseline_correct": index % 2 == 0,
                    "loop_correct": index % 3 == 0,
                    "outcome": "loop_hurts" if index % 4 == 0 else "loop_helps",
                }
            )
            actions.append(index % 2 == 0)
        left = [row["baseline_correct"] for row in rows]
        right = [row["loop_correct"] for row in rows]
        self.assertEqual(_cluster_ci(rows, left, right, 100, 7), _cluster_ci(rows, left, right, 100, 7))
        self.assertEqual(_random_null(rows, actions, 100, 9), _random_null(rows, actions, 100, 9))

    def test_synthetic_analysis_freezes_policy_then_seals_terminal_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "receipts").mkdir()
            (root / "preflight").mkdir()
            (root / "receipts/shards_verified.json").write_text("{}\n")
            split_rows = []
            for shard_id in range(8):
                shard = root / ("shards/shard-%d" % shard_id)
                shard.mkdir(parents=True)
                arm_rows = {name: [] for name in ("baseline_a", "baseline_b", "always_loop")}
                decision_rows = []
                signal_rows = []
                for offset in range(2):
                    index = shard_id * 2 + offset
                    key_name = "k%x" % index
                    gold = 0
                    base_prediction = 0 if index % 3 else 1
                    loop_prediction = 0 if index % 4 else 1
                    arm_rows["baseline_a"].append(sample("baseline_a", key_name, base_prediction, gold))
                    arm_rows["baseline_b"].append(sample("baseline_b", key_name, base_prediction, gold))
                    arm_rows["always_loop"].append(sample("always_loop", key_name, loop_prediction, gold))
                    row = decision(key_name)
                    row["r0_median"] = float(index)
                    row["c0"] = float(index)
                    row["n0_median"] = float(index)
                    decision_rows.append(row)
                    current_signals = signals(key_name)
                    for signal_row in current_signals:
                        signal_row["r0_median"] = float(index)
                        signal_row["c0"] = float(index)
                        signal_row["n0_median"] = float(index)
                    signal_rows.extend(current_signals)
                    split_rows.append({"doc_key": key_name, "split": "development" if index % 2 == 0 else "holdout"})
                for name, rows in arm_rows.items():
                    (shard / (name + "_samples.jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
                (shard / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decision_rows))
                (shard / "signal_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in signal_rows))
            (root / "preflight/split_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in split_rows))
            analysis = root / "analysis"
            self.assertEqual(analyze_main(["--run-root", str(root), "--output-dir", str(analysis)]), 0)
            self.assertTrue((analysis / "selected_policy.json").is_file())
            self.assertEqual(verify_final_main(["--analysis-dir", str(analysis)]), 0)
            self.assertTrue((analysis / "analysis_complete.txt").is_file())


if __name__ == "__main__":
    unittest.main()
