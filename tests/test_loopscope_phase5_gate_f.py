import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

from tflt.loopscope.phase5_variable_width_outcome import (
    Phase5GateFError,
    build_eval_argv,
    build_launch_manifest,
    build_supplement_launch_manifest,
    build_statistical_analysis,
    derive_supplemental_cell,
    derive_top4,
    exact_mcnemar_p,
    joint_paired_bootstrap,
    validate_extension_card,
    validate_launch_manifest,
    validate_scheduler_rows,
    validate_smoke_structural_info,
    validate_supplement_launch_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase5_multiwidth_top4_outcome_card.json"
RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase5-gate-f-multiwidth-top4-outcome-20260723T120000Z"
)
SUPPLEMENT_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase5-gate-f-supplement-15-19-outcome-20260723T120000Z"
)
IMPLEMENTATION_COMMIT = "1" * 40
SOURCE_COMMIT = "f33a37a4e5c0cfc461184a3471277897c77cf83b"


def _score_payload(card):
    rows = []
    for rank in range(1, 43):
        if rank <= 4:
            frozen = card["frozen_cells"][rank - 1]
            width = frozen["width"]
            start = frozen["start"]
            window = frozen["window"]
            score = frozen["score"]
        elif rank == 11:
            frozen = card["supplemental_cell"]
            width = frozen["width"]
            start = frozen["start"]
            window = frozen["window"]
            score = frozen["score"]
        else:
            width = 4
            start = 11 + ((rank - 5) % 11)
            window = "%d:%d" % (start, start + width - 1)
            score = -float(rank)
        rows.append(
            {
                "global_rank": rank,
                "width": width,
                "start": start,
                "window": window,
                "score": score,
                "width_rank": 2 if rank == 11 else 1,
            }
        )
    return {
        "manifest_sha256": card["source"]["gate_e_score_manifest_sha256"],
        "outcome_values_consumed": False,
        "test_split_consumed": False,
        "post_terminal_exploratory_no_selection": True,
        "rows": list(reversed(rows)),
    }


class Phase5GateFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
        cls.score = _score_payload(cls.card)

    def test_card_and_exact_top4_derivation(self):
        validate_extension_card(self.card)
        derived = derive_top4(self.score, self.card)
        self.assertEqual(derived, self.card["frozen_cells"])
        self.assertEqual(
            [(row["gate_e_rank"], row["width"], row["window"]) for row in derived],
            [(1, 5, "13:17"), (2, 6, "15:20"), (3, 6, "12:17"), (4, 3, "15:17")],
        )
        changed = copy.deepcopy(self.score)
        changed["rows"][-1]["score"] += 1e-9
        with self.assertRaisesRegex(Phase5GateFError, "derived top4"):
            derive_top4(changed, self.card)
        supplement = derive_supplemental_cell(self.score, self.card)
        self.assertEqual(supplement, self.card["supplemental_cell"])
        changed = copy.deepcopy(self.score)
        next(row for row in changed["rows"] if row["global_rank"] == 11)["width_rank"] = 3
        with self.assertRaisesRegex(Phase5GateFError, "supplemental identity"):
            derive_supplemental_cell(changed, self.card)

    def test_launch_is_four_cells_and_only_width_window_vary(self):
        manifest = build_launch_manifest(
            mode="formal",
            run_root=RUN_ROOT,
            implementation_commit=IMPLEMENTATION_COMMIT,
            expected_source_commit=SOURCE_COMMIT,
            partition="emergency_gpu",
            gres="gpu:a800:1",
            qos=None,
            account=None,
            array_throttle=4,
            time_limit="24:00:00",
            batch_size="16",
            created_at_utc="2026-07-23T00:00:00Z",
            card=self.card,
            score_payload=self.score,
            implementation_sha256={path: "a" * 64 for path in (
                "configs/loopscope/phase5_multiwidth_top4_outcome_card.json",
                "src/tflt/loopscope/phase5_variable_width_outcome.py",
                "scripts/loopscope/run_qwen4base_phase5_gate_f.py",
                "tests/test_loopscope_phase5_gate_f.py",
            )},
        )
        validate_launch_manifest(manifest, mode="formal", card=self.card)
        self.assertEqual(manifest["cell_count"], 4)
        self.assertEqual(manifest["scheduler"]["array"], "0-3%4")
        self.assertEqual(manifest["recipe"]["k"], 3)
        self.assertEqual(manifest["recipe"]["step_size"], 1.0 / 3.0)
        self.assertEqual(manifest["recipe"]["total_horizon"], 1.0)
        self.assertIsNone(manifest["recipe"]["limit"])
        for cell in manifest["cells"]:
            self.assertIn("--loop", cell["eval_argv"])
            self.assertEqual(cell["eval_argv"], build_eval_argv(
                "formal", Path(cell["eval_output_dir"]), cell["window"], "16"
            ))
            start, end = map(int, cell["window"].split(":"))
            self.assertEqual(end - start + 1, cell["width"])

    def test_supplement_is_exactly_one_full_cell(self):
        manifest = build_supplement_launch_manifest(
            run_root=SUPPLEMENT_ROOT,
            implementation_commit=IMPLEMENTATION_COMMIT,
            expected_source_commit=SOURCE_COMMIT,
            partition="emergency_gpu",
            gres="gpu:a800:1",
            qos=None,
            account=None,
            time_limit="24:00:00",
            batch_size="16",
            created_at_utc="2026-07-23T00:00:00Z",
            card=self.card,
            score_payload=self.score,
            implementation_sha256={
                path: "a" * 64
                for path in (
                    "configs/loopscope/phase5_multiwidth_top4_outcome_card.json",
                    "src/tflt/loopscope/phase5_variable_width_outcome.py",
                    "scripts/loopscope/run_qwen4base_phase5_gate_f.py",
                    "tests/test_loopscope_phase5_gate_f.py",
                )
            },
        )
        validate_supplement_launch_manifest(manifest, card=self.card)
        self.assertEqual(manifest["cell_count"], 1)
        self.assertEqual(manifest["scheduler"]["array"], "0-0%1")
        self.assertEqual(manifest["cells"][0]["gate_e_rank"], 11)
        self.assertEqual(manifest["cells"][0]["width_rank"], 2)
        self.assertEqual(manifest["cells"][0]["window"], "15:19")
        self.assertEqual(manifest["recipe"]["k"], 3)
        self.assertEqual(manifest["recipe"]["operator_body_calls_per_prefill"], 3)
        self.assertEqual(manifest["recipe"]["step_size"], 1.0 / 3.0)
        self.assertIsNone(manifest["recipe"]["limit"])

    def test_smoke_recipe_and_scheduler_exactness(self):
        argv = build_eval_argv("smoke", Path("/tmp/eval"), "15:17", "16")
        self.assertEqual(argv[argv.index("--limit") + 1], "2")
        self.assertEqual(argv[argv.index("--k") + 1], "3")
        self.assertEqual(argv[argv.index("--strategy") + 1], "euler")
        self.assertEqual(argv[argv.index("--alpha") + 1], "1.0")
        self.assertEqual(argv[argv.index("--cache-strategy") + 1], "first")
        rows = [
            {
                "JobID": "123_%d" % index,
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "ElapsedRaw": "10",
                "AllocTRES": "cpu=8,gres/gpu:a800=1,mem=64G",
                "Partition": "emergency_gpu",
                "NodeList": "gpu3-1",
            }
            for index in range(4)
        ]
        self.assertEqual(len(validate_scheduler_rows(rows, "123")), 4)
        with self.assertRaisesRegex(Phase5GateFError, "exact four"):
            validate_scheduler_rows(rows[:3], "123")
        validate_smoke_structural_info(
            {
                "operator_body_calls_per_prompt": [3, 3, 3, 3, 3, 3],
                "prompt_count": 6,
                "restore_allclose_all_prompts": True,
                "overall_decision": "loop_effective_logits_changed",
            }
        )
        with self.assertRaisesRegex(Phase5GateFError, "structural closure"):
            validate_smoke_structural_info(
                {
                    "operator_body_calls_per_prompt": [3, 3, 2],
                    "prompt_count": 3,
                    "restore_allclose_all_prompts": True,
                    "overall_decision": "loop_effective_logits_changed",
                }
            )

    def test_joint_bootstrap_analysis_and_claim_boundary(self):
        correctness = np.asarray(
            [
                [0, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [0, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 1, 1],
            ],
            dtype=np.int8,
        )
        subjects = ["a", "a", "b", "b", "c", "c"]
        first = joint_paired_bootstrap(correctness, subjects, replicates=41, seed=7)
        second = joint_paired_bootstrap(correctness, subjects, replicates=41, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["cell_gain_summaries"]), 5)
        identities = [
            {"task": "mmlu_x", "doc_id": str(index), "doc_hash": "%064x" % index}
            for index in range(6)
        ]
        analysis = build_statistical_analysis(
            correctness=correctness,
            identities=identities,
            subjects=subjects,
            cells=self.card["frozen_cells"] + [self.card["supplemental_cell"]],
            replicates=41,
            seed=7,
        )
        self.assertEqual(analysis["population"]["matrix_shape"], [6, 6])
        self.assertEqual(len(analysis["cells"]), 6)
        self.assertFalse(
            analysis["claim_boundary"]["prospective_selected_window_success_claimed"]
        )
        row = analysis["cells"][1]
        self.assertEqual(sum(row["transitions_vs_baseline"].values()), 6)
        self.assertAlmostEqual(
            row["gain_percentage_points"],
            (correctness[:, 1].mean() - correctness[:, 0].mean()) * 100.0,
        )
        self.assertTrue(math.isfinite(row["exact_two_sided_mcnemar_p"]))
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
