import argparse
import hashlib
import json
import shlex
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from tflt.loopscope.analysis import (
    AnalysisError,
    _phase_assessment,
    _same_or_adjacent,
    _signal_regret,
    add_analyze_window_grid_args,
    analyze_window_grid,
    cmd_analyze_window_grid,
    exact_mcnemar_p,
    extract_accuracy,
    paired_comparison,
)
from tflt.loopscope.schema import (
    SELECTION_SCHEMA_VERSION,
    WINDOW_GRID_SCHEMA_VERSION,
    attach_manifest_sha256,
    canonical_json_bytes,
    manifest_sha256,
)

MODEL_COMMIT = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
from tflt.loopscope.selection import PRIMARY_SIGNALS


def _result(accuracy, correctness):
    return {
        "results": {"mmlu": {"acc,none": accuracy}},
        "samples": {
            "mmlu": [
                {"doc_id": index, "acc": value}
                for index, value in enumerate(correctness)
            ]
        },
    }


def _score_report():
    candidates = []
    for window, high in (("0:1", True), ("2:3", False)):
        scores = {
            "entropy_drop": 2.0 if high else 1.0,
            "entropy_flatness": 1.0 if high else 2.0,
            "kl_to_final_drop": 2.0 if high else 1.0,
            "activity_r": 2.0 if high else 1.0,
            "negative_contraction_q": 2.0 if high else 1.0,
            "r_times_one_minus_q": 0.25 if high else 0.0,
        }
        sample_scores = []
        for sample_id in ("s0", "s1", "s2", "s3", "s4", "s5"):
            activity = 0.5 if high else 0.2
            contraction = 0.5 if high else 1.2
            sample_scores.append(
                {
                    "id": sample_id,
                    "scores": {
                        "entropy_drop": scores["entropy_drop"],
                        "entropy_flatness": scores["entropy_flatness"],
                        "kl_to_final_drop": scores["kl_to_final_drop"],
                        "activity_r": activity,
                        "negative_contraction_q": -contraction,
                        "r_times_one_minus_q": activity * max(0.0, 1.0 - contraction),
                    },
                }
            )
        candidates.append(
            {"window": window, "scores": scores, "sample_scores": sample_scores}
        )
    grid = _grid()
    report = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "primary_signal": "r_times_one_minus_q",
        "primary_top_windows": ["0:1"],
        "criterion": {
            "tie_policy": {
                "ranking_ties": "retain_all",
                "display_tiebreak": "frozen_candidate_order",
                "phase_gate_regret": "worst_case",
            },
            "bootstrap": {
                "replicates": 5,
                "seed": 20260710,
                "stable_if_same_or_adjacent_at_least": 4,
            }
        },
        "window_grid": {
            "manifest_sha256": grid["manifest_sha256"],
            "layer_count": grid["layer_count"],
            "candidate_windows": [item["window"] for item in grid["windows"]],
        },
        "candidates": candidates,
    }
    report["criterion_sha256"] = manifest_sha256(report["criterion"])
    report["manifest_sha256"] = manifest_sha256(report)
    return report


def _grid():
    payload = {
        "schema_version": WINDOW_GRID_SCHEMA_VERSION,
        "layer_count": 12,
        "generation": {
            "width": 2,
            "candidate_count_actual": 2,
            "random_count": 5,
            "min_center_fraction": 0.0,
            "max_center_fraction": 1.0,
        },
        "anchors": {"required": "0:1", "fixed_depth": "0:1"},
        "windows": [
            {
                "window": "0:1",
                "start": 0,
                "end": 1,
                "center": 0.5,
                "roles": ["candidate", "fixed_depth", "anchor"],
            },
            {
                "window": "2:3",
                "start": 2,
                "end": 3,
                "center": 2.5,
                "roles": ["candidate"],
            },
        ],
        "comparison_windows": {
            "fixed_depth": "0:1",
            "random_in_band": ["2:3", "4:5", "6:7", "8:9", "10:11"],
        },
    }
    return attach_manifest_sha256(payload)


def _window_results():
    baseline = [1, 0, 1, 0]
    return {
        "0:1": _result(0.75, [1, 1, 1, 0]),
        "2:3": _result(0.25, [0, 0, 1, 0]),
        "4:5": _result(0.50, baseline),
        "6:7": _result(0.50, baseline),
        "8:9": _result(0.50, baseline),
        "10:11": _result(0.50, baseline),
    }


def _phase1_grid():
    payload = {
        "schema_version": WINDOW_GRID_SCHEMA_VERSION,
        "layer_count": 28,
        "generation": {
            "width": 4,
            "candidate_count_actual": 2,
            "random_count": 5,
            "min_center_fraction": 0.0,
            "max_center_fraction": 1.0,
        },
        "anchors": {"required": "0:3", "fixed_depth": "0:3"},
        "windows": [
            {
                "window": "0:3",
                "start": 0,
                "end": 3,
                "center": 1.5,
                "roles": ["anchor", "candidate", "fixed_depth"],
            },
            {
                "window": "8:11",
                "start": 8,
                "end": 11,
                "center": 9.5,
                "roles": ["candidate"],
            },
        ],
        "comparison_windows": {
            "fixed_depth": "0:3",
            "random_in_band": ["4:7", "8:11", "12:15", "16:19", "20:23"],
        },
    }
    return attach_manifest_sha256(payload)


def _full_pool_contract():
    sample_ids = ["s%d" % index for index in range(6)]
    records = [
        {"id": sample_id, "prompt_sha256": ("%x" % (index + 1)) * 64}
        for index, sample_id in enumerate(sample_ids)
    ]
    rendering_records = [
        {"id": sample_id, "render_sha256": ("%x" % (index + 7)) * 64}
        for index, sample_id in enumerate(sample_ids)
    ]
    renderer = {
        "renderer_entrypoint": "fixture.renderer",
        "lm_eval_version": "0.4.11",
        "renderer_source_sha256": "a" * 64,
        "source_files_sha256": "a" * 64,
        "template_sha256": "b" * 64,
        "task_configs_sha256": "b" * 64,
        "render_contract_sha256": "c" * 64,
        "dataset_revision": "f" * 40,
        "dataset_fingerprint_sha256": "1" * 64,
        "source_projection_sha256": "2" * 64,
        "render_sha256": "d" * 64,
        "renderer_manifest_sha256": "e" * 64,
    }
    render_hash = hashlib.sha256(canonical_json_bytes(rendering_records)).hexdigest()
    source_manifest = {
        "schema_version": "loopscope.probe-pool-manifest.v1",
        "source": "fixture-source",
        "split": "auxiliary_train",
        "count": len(sample_ids),
        "seed": 20260710,
        "sample_ids": sample_ids,
        "records": records,
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": renderer,
        "rendering_records": rendering_records,
        "render_contract_subset_sha256": render_hash,
    }
    source_manifest["manifest_sha256"] = manifest_sha256(source_manifest)
    selected = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": source_manifest["source"],
        "split": source_manifest["split"],
        "count": len(sample_ids),
        "seed": source_manifest["seed"],
        "sample_ids": sample_ids,
        "records": records,
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": renderer,
        "render_contract_sha256": renderer["render_contract_sha256"],
        "rendering_records": rendering_records,
        "render_contract_subset_sha256": render_hash,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
    }
    pool = dict(selected)
    pool.update(
        {
            "manifest_sha256": manifest_sha256(selected),
            "source_manifest_count": len(sample_ids),
            "selected_subset_sha256": manifest_sha256(selected),
            "source_render_contract_subset_sha256": render_hash,
            "selected_render_contract_subset_sha256": render_hash,
        }
    )
    return source_manifest, pool


def _phase1_score_report(grid, pool):
    report = _score_report()
    frozen = [item["window"] for item in grid["windows"]]
    for candidate, window in zip(report["candidates"], frozen):
        candidate["window"] = window
    report["primary_top_windows"] = [frozen[0]]
    report["window_grid"] = {
        "manifest_sha256": grid["manifest_sha256"],
        "layer_count": grid["layer_count"],
        "candidate_windows": frozen,
    }
    report["probe_provenance"] = {
        "git": {"branch": "loopscope", "commit": "code-commit", "dirty": False},
        "model": {
            "alias": "qwen3-1.7b-base",
            "repo_id": "Qwen/Qwen3-1.7B-Base",
            "revision": MODEL_COMMIT,
            "layer_count": 28,
        },
        "tokenizer_revision": MODEL_COMMIT,
        "revision_closure": {
            "schema_version": "loopscope.revision-closure.v1",
            "manifest_commit": MODEL_COMMIT,
            "model_commit": MODEL_COMMIT,
            "tokenizer_commit": MODEL_COMMIT,
            "match": True,
        },
        "runtime_dtype": "float16",
        "probe_pool": {
            key: pool[key]
            for key in (
                "source",
                "split",
                "count",
                "seed",
                "source_manifest_count",
                "sample_ids",
                "records",
                "rendering_records",
                "task_group",
                "num_fewshot",
                "uses_target_gold_labels",
                "fewshot_answers_present",
                "source_manifest_sha256",
                "selected_subset_sha256",
                "source_render_contract_subset_sha256",
                "selected_render_contract_subset_sha256",
                "render_contract_sha256",
                "renderer",
            )
        },
    }
    report["manifest_sha256"] = manifest_sha256(report)
    return report


def _phase1_eval_argv(output_dir, window=None, limit=None):
    argv = [
        "python",
        "-m",
        "tflt.eval_runner",
        "--model",
        "qwen3-1.7b-base",
        "--revision",
        MODEL_COMMIT,
        "--tasks",
        "mmlu",
        "--output-dir",
        str(output_dir),
        "--num-fewshot",
        "5",
        "--batch-size",
        "auto",
        "--dtype",
        "float16",
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    if window is not None:
        argv.extend(
            [
                "--loop",
                "--window",
                window,
                "--k",
                "2",
                "--iteration-mode",
                "block",
                "--strategy",
                "damped_euler",
                "--alpha",
                "1.0",
                "--beta",
                "0.0",
                "--cache-strategy",
                "last",
                "--decode-mode",
                "bypass",
            ]
        )
    return argv


def _write_phase1_eval_artifact(output_dir, accuracy, correctness, window, revision):
    output_dir.mkdir(parents=True)
    command_args = {
        "model": "qwen3-1.7b-base",
        "revision": MODEL_COMMIT,
        "tasks": "mmlu",
        "output_dir": str(output_dir),
        "limit": None,
        "num_fewshot": 5,
        "batch_size": "auto",
        "dtype": "float16",
        "loop": window is not None,
        "window": window or "12:15",
        "k": 2,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "first_n": None,
    }
    (output_dir / "results.json").write_text(
        json.dumps(_result(accuracy, correctness)), encoding="utf-8"
    )
    (output_dir / "command_args.json").write_text(
        json.dumps(command_args), encoding="utf-8"
    )
    (output_dir / "model_revision.json").write_text(
        json.dumps(
            {
                "schema_version": "loopscope.model-tokenizer-revision.v1",
                "repo_id": "Qwen/Qwen3-1.7B-Base",
                "commit_hash": revision,
                "model_commit": revision,
                "tokenizer_commit": revision,
                "manifest_commit": MODEL_COMMIT,
                "match": revision == MODEL_COMMIT,
            }
        ),
        encoding="utf-8",
    )


def _write_full_cli_fixture(root):
    run_root = root / "loopscope-qwen17-mmlu-phase1-fixture"
    grid = _phase1_grid()
    grid_path = run_root / "manifests" / "window_grid.json"
    grid_path.parent.mkdir(parents=True)
    grid_path.write_text(json.dumps(grid), encoding="utf-8")
    source_pool_manifest, probe_pool = _full_pool_contract()
    pool_manifest_path = run_root / "manifests" / "probe_pool_manifest.json"
    pool_manifest_path.write_text(json.dumps(source_pool_manifest), encoding="utf-8")
    pool_path = run_root / "manifests" / "probe_pool.jsonl"
    pool_bytes = "".join(
        json.dumps({"id": sample_id}) + "\n"
        for sample_id in source_pool_manifest["sample_ids"]
    ).encode("utf-8")
    pool_path.write_bytes(pool_bytes)
    layer_probe_path = (
        run_root / "gate-e-probe" / "probe-layers-full" / "probe_report.json"
    )
    window_probe_path = (
        run_root / "gate-e-probe" / "probe-windows-full" / "probe_report.json"
    )
    for path in (layer_probe_path, window_probe_path):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model": {"revision": MODEL_COMMIT},
                    "tokenizer": {"revision": MODEL_COMMIT},
                    "revision_closure": {
                        "schema_version": "loopscope.revision-closure.v1",
                        "manifest_commit": MODEL_COMMIT,
                        "model_commit": MODEL_COMMIT,
                        "tokenizer_commit": MODEL_COMMIT,
                        "match": True,
                    },
                    "probe_pool": probe_pool,
                }
            ),
            encoding="utf-8",
        )
    score = _phase1_score_report(grid, probe_pool)
    score["inputs"] = {
        "paths": {
            "layer_probe": str(layer_probe_path),
            "window_probes": [str(window_probe_path)],
            "criterion": str(run_root / "manifests" / "criterion_v0.json"),
        },
        "layer_probe_manifest_sha256": probe_pool["manifest_sha256"],
        "window_probe_manifest_sha256": [probe_pool["manifest_sha256"]],
    }
    score["manifest_sha256"] = manifest_sha256(score)
    score_output = run_root / "gate-e-score" / "score-windows"
    score_output.mkdir(parents=True)
    score_path = score_output / "window_scores.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    criterion_path = run_root / "manifests" / "criterion_v0.json"
    criterion_path.write_text(json.dumps(score["criterion"]), encoding="utf-8")
    criterion_file_hash = hashlib.sha256(criterion_path.read_bytes()).hexdigest()

    baseline_output = run_root / "gate-e-full" / "baseline-full"
    baseline_correctness = [1, 0, 1, 0]
    _write_phase1_eval_artifact(
        baseline_output, 0.5, baseline_correctness, None, MODEL_COMMIT
    )
    jobs = []

    def job(job_id, output_dir, window):
        argv = _phase1_eval_argv(output_dir, window=window)
        return {
            "job_id": job_id,
            "stage": "gate-e-full",
            "output_dir": str(output_dir),
            "claim_dir": str(output_dir) + ".claim",
            "argv": argv,
            "command": shlex.join(argv),
            "revision_artifact": str(output_dir / "model_revision.json"),
            "revision_keys": {
                "model_commit": ["model_commit"],
                "tokenizer_commit": ["tokenizer_commit"],
                "manifest_commit": ["manifest_commit"],
            },
            "automatic_retry": False,
        }

    jobs.append(job("baseline-full", baseline_output, None))
    window_results = {
        "0:3": (0.75, [1, 1, 1, 0]),
        "8:11": (0.25, [0, 0, 1, 0]),
        "4:7": (0.50, baseline_correctness),
        "12:15": (0.50, baseline_correctness),
        "16:19": (0.50, baseline_correctness),
        "20:23": (0.50, baseline_correctness),
    }
    specifications = []
    for window, (accuracy, correctness) in window_results.items():
        output = run_root / "gate-e-full" / ("window-" + window.replace(":", "-") + "-full")
        _write_phase1_eval_artifact(
            output, accuracy, correctness, window, MODEL_COMMIT
        )
        jobs.append(job("window-%s-full" % window.replace(":", "-"), output, window))
        specifications.append("%s=%s" % (window, output / "results.json"))

    manifest = {
        "schema_version": "loopscope.phase1-run.v1",
        "run_root": str(run_root),
        "inputs": {
            "window_grid": {
                "snapshot_path": str(grid_path),
                "manifest_sha256": grid["manifest_sha256"],
            },
            "criterion": {
                "snapshot_path": str(criterion_path),
                "file_sha256": criterion_file_hash,
                "criterion_sha256": score["criterion_sha256"],
            },
            "probe_pool": {
                "snapshot_path": str(pool_path),
                "manifest_snapshot_path": str(pool_manifest_path),
                "pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
                "manifest_sha256": source_pool_manifest["manifest_sha256"],
                "count": source_pool_manifest["count"],
                "render_contract_sha256": source_pool_manifest["renderer"][
                    "render_contract_sha256"
                ],
                "render_contract_subset_sha256": source_pool_manifest[
                    "render_contract_subset_sha256"
                ],
            },
        },
        "git": {"commit": "code-commit"},
        "frozen_recipe": {
            "model": "qwen3-1.7b-base",
            "repo_id": "Qwen/Qwen3-1.7B-Base",
            "revision": MODEL_COMMIT,
            "task": "mmlu",
            "num_fewshot": 5,
            "dtype": "float16",
            "k": 2,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "window_width": 4,
        },
        "revision_policy": {
            "cli_pins_revision": True,
            "manifest_commit": MODEL_COMMIT,
        },
        "stages": {
            "gate-e-probe": {
                "jobs": [
                    {
                        "job_id": "probe-layers-full",
                        "revision_artifact": str(layer_probe_path),
                        "revision_keys": {
                            "model_commit": ["model", "revision"],
                            "tokenizer_commit": ["tokenizer", "revision"],
                            "manifest_commit": ["revision_closure", "manifest_commit"],
                        },
                    },
                    {
                        "job_id": "probe-windows-full",
                        "revision_artifact": str(window_probe_path),
                        "revision_keys": {
                            "model_commit": ["model", "revision"],
                            "tokenizer_commit": ["tokenizer", "revision"],
                            "manifest_commit": ["revision_closure", "manifest_commit"],
                        },
                    },
                ]
            },
            "gate-e-score": {
                "jobs": [
                    {
                        "job_id": "score-windows-full",
                        "output_dir": str(score_output),
                    }
                ]
            },
            "gate-e-full": {
                "scientific_selection_allowed": True,
                "jobs": jobs,
            }
        },
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    manifest_path = run_root / "control" / "phase1_run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    provenance_dir = run_root / "control" / "provenance"
    provenance_dir.mkdir()
    revision_report = {
        "schema_version": "loopscope.model-tokenizer-revision-check.v2",
        "run_manifest_sha256": manifest["manifest_sha256"],
        "stage": "gate-e-probe",
        "manifest_commit": MODEL_COMMIT,
        "match": True,
        "unique_revisions": [MODEL_COMMIT],
        "observations": [
            {
                "job_id": job_id,
                "stage": "gate-e-probe",
                "artifact": str(path),
                "model_commit": MODEL_COMMIT,
                "tokenizer_commit": MODEL_COMMIT,
                "manifest_commit": MODEL_COMMIT,
                "match": True,
            }
            for job_id, path in (
                ("probe-layers-full", layer_probe_path),
                ("probe-windows-full", window_probe_path),
            )
        ],
    }
    revision_report["manifest_sha256"] = manifest_sha256(revision_report)
    revision_path = provenance_dir / "gate-e-probe-model-revisions.json"
    revision_path.write_text(json.dumps(revision_report), encoding="utf-8")
    freeze = {
        "schema_version": "loopscope.score-freeze.v1",
        "run_manifest_sha256": manifest["manifest_sha256"],
        "score_job_id": "score-windows-full",
        "score_report_path": str(score_path),
        "score_report_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "score_report_manifest_sha256": score["manifest_sha256"],
        "criterion_path": str(criterion_path),
        "criterion_file_sha256": criterion_file_hash,
        "criterion_sha256": score["criterion_sha256"],
        "window_grid_manifest_sha256": grid["manifest_sha256"],
        "candidate_windows": [item["window"] for item in grid["windows"]],
        "gate_e_probe_revision_report_sha256": revision_report["manifest_sha256"],
        "full_probe_evidence": {
            "run_manifest_sha256": manifest["manifest_sha256"],
            "window_grid_manifest_sha256": grid["manifest_sha256"],
            "criterion_sha256": score["criterion_sha256"],
            "probe_pool_manifest_path": str(pool_manifest_path.resolve()),
            "probe_pool_path": str(pool_path.resolve()),
            "count": source_pool_manifest["count"],
            "sample_ids": source_pool_manifest["sample_ids"],
            "source_manifest_sha256": source_pool_manifest["manifest_sha256"],
            "selected_subset_sha256": probe_pool["selected_subset_sha256"],
            "render_contract_subset_sha256": source_pool_manifest[
                "render_contract_subset_sha256"
            ],
            "sample_ids_sha256": hashlib.sha256(
                canonical_json_bytes(source_pool_manifest["sample_ids"])
            ).hexdigest(),
            "revision_binding": {
                "manifest_commit": MODEL_COMMIT,
                "model_commit": MODEL_COMMIT,
                "tokenizer_commit": MODEL_COMMIT,
            },
            "probe_reports": {
                label: {
                    "path": str(path.resolve()),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "canonical_sha256": manifest_sha256(
                        json.loads(path.read_text(encoding="utf-8"))
                    ),
                    "full_pool_count": source_pool_manifest["count"],
                    "sample_ids": source_pool_manifest["sample_ids"],
                    "sample_ids_sha256": hashlib.sha256(
                        canonical_json_bytes(source_pool_manifest["sample_ids"])
                    ).hexdigest(),
                    "source_manifest_sha256": source_pool_manifest["manifest_sha256"],
                    "selected_subset_sha256": probe_pool["selected_subset_sha256"],
                    "render_contract_subset_sha256": source_pool_manifest[
                        "render_contract_subset_sha256"
                    ],
                    "probe_pool_manifest_sha256": probe_pool["manifest_sha256"],
                    "revision_closure": {
                        "schema_version": "loopscope.revision-closure.v1",
                        "manifest_commit": MODEL_COMMIT,
                        "model_commit": MODEL_COMMIT,
                        "tokenizer_commit": MODEL_COMMIT,
                        "match": True,
                    },
                }
                for label, path in (
                    ("layer", layer_probe_path),
                    ("window", window_probe_path),
                )
            },
        },
    }
    freeze["manifest_sha256"] = manifest_sha256(freeze)
    (provenance_dir / "gate-e-score-freeze.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )
    args = argparse.Namespace(
        baseline_results=str(baseline_output / "results.json"),
        window_result=specifications,
        selection_report=str(score_path),
        window_grid=str(grid_path),
        run_manifest=str(manifest_path),
        output_dir=str(run_root / "analysis"),
        task="mmlu",
        metric="acc,none",
        paired_bootstrap_replicates=20,
        paired_bootstrap_seed=20260710,
        analysis_scope="full",
    )
    return args, run_root


def _rewrite_score_as_four_sample_prefix(args, run_root):
    score_path = Path(args.selection_report)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    pool = score["probe_provenance"]["probe_pool"]
    pool["count"] = 4
    for key in ("sample_ids", "records", "rendering_records"):
        pool[key] = pool[key][:4]
    render_hash = hashlib.sha256(
        canonical_json_bytes(pool["rendering_records"])
    ).hexdigest()
    pool["selected_render_contract_subset_sha256"] = render_hash
    selected = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": pool["source"],
        "split": pool["split"],
        "count": 4,
        "seed": pool["seed"],
        "sample_ids": pool["sample_ids"],
        "records": pool["records"],
        "task_group": pool["task_group"],
        "num_fewshot": pool["num_fewshot"],
        "uses_target_gold_labels": pool["uses_target_gold_labels"],
        "fewshot_answers_present": pool["fewshot_answers_present"],
        "renderer": pool["renderer"],
        "render_contract_sha256": pool["render_contract_sha256"],
        "rendering_records": pool["rendering_records"],
        "render_contract_subset_sha256": render_hash,
        "source_manifest_sha256": pool["source_manifest_sha256"],
    }
    pool["selected_subset_sha256"] = manifest_sha256(selected)
    for candidate in score["candidates"]:
        candidate["sample_scores"] = candidate["sample_scores"][:4]
    score["inputs"]["layer_probe_manifest_sha256"] = pool["selected_subset_sha256"]
    score["inputs"]["window_probe_manifest_sha256"] = [
        pool["selected_subset_sha256"]
    ]
    for relative in (
        "gate-e-probe/probe-layers-full/probe_report.json",
        "gate-e-probe/probe-windows-full/probe_report.json",
    ):
        report_path = run_root / relative
        report_path.write_text(json.dumps({"probe_pool": pool}), encoding="utf-8")
    score["manifest_sha256"] = manifest_sha256(score)
    score_path.write_text(json.dumps(score), encoding="utf-8")
    freeze_path = run_root / "control" / "provenance" / "gate-e-score-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["score_report_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    freeze["score_report_manifest_sha256"] = score["manifest_sha256"]
    freeze["manifest_sha256"] = manifest_sha256(freeze)
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")


class LoopScopeAnalysisTest(unittest.TestCase):
    def test_analysis_scope_is_required_by_cli_parser(self):
        parser = argparse.ArgumentParser()
        add_analyze_window_grid_args(parser)
        argv = [
            "--baseline-results",
            "baseline/results.json",
            "--window-result",
            "0:3=window/results.json",
            "--selection-report",
            "scores.json",
            "--window-grid",
            "grid.json",
            "--output-dir",
            "analysis",
        ]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(argv)

    def test_full_cli_requires_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = _write_full_cli_fixture(Path(tmp))
            args.run_manifest = None
            with self.assertRaisesRegex(AnalysisError, "requires --run-manifest"):
                cmd_analyze_window_grid(args)

    def test_full_cli_validates_and_records_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = _write_full_cli_fixture(Path(tmp))
            self.assertEqual(cmd_analyze_window_grid(args), 0)
            report = json.loads(
                (Path(args.output_dir) / "selection_report.json").read_text(
                    encoding="utf-8"
                )
            )
            provenance = report["execution_provenance"]
            self.assertTrue(provenance["validated"])
            self.assertEqual(provenance["stage"], "gate-e-full")
            self.assertEqual(provenance["model_revision"], MODEL_COMMIT)
            self.assertEqual(provenance["revision_closure"]["tokenizer_commit"], MODEL_COMMIT)
            baseline_hashes = provenance["artifacts"]["baseline"]["artifact_sha256"]
            self.assertIn("command_args.json", baseline_hashes)

    def test_full_cli_rejects_limit_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            command_path = run_root / "gate-e-full" / "baseline-full" / "command_args.json"
            command = json.loads(command_path.read_text(encoding="utf-8"))
            command["limit"] = 5
            command_path.write_text(json.dumps(command), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "limit artifact"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_mixed_model_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            revision_path = (
                run_root / "gate-e-full" / "window-8-11-full" / "model_revision.json"
            )
            revision = json.loads(revision_path.read_text(encoding="utf-8"))
            revision["commit_hash"] = "b" * 40
            revision["model_commit"] = "b" * 40
            revision["tokenizer_commit"] = "b" * 40
            revision["manifest_commit"] = "b" * 40
            revision_path.write_text(json.dumps(revision), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "frozen manifest revision"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_probe_eval_revision_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = _write_full_cli_fixture(Path(tmp))
            score_path = Path(args.selection_report)
            score = json.loads(score_path.read_text(encoding="utf-8"))
            score["probe_provenance"]["model"]["revision"] = "b" * 40
            score["manifest_sha256"] = manifest_sha256(score)
            score_path.write_text(json.dumps(score), encoding="utf-8")
            freeze_path = (
                Path(args.run_manifest).parent
                / "provenance"
                / "gate-e-score-freeze.json"
            )
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze["score_report_sha256"] = hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest()
            freeze["score_report_manifest_sha256"] = score["manifest_sha256"]
            freeze["manifest_sha256"] = manifest_sha256(freeze)
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "probe model mismatch"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_reforged_scores_after_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = _write_full_cli_fixture(Path(tmp))
            score_path = Path(args.selection_report)
            score = json.loads(score_path.read_text(encoding="utf-8"))
            score["candidates"][0]["scores"]["activity_r"] = 999.0
            score["manifest_sha256"] = manifest_sha256(score)
            score_path.write_text(json.dumps(score), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "changed after freeze"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_frozen_score_copied_to_another_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            copied_dir = run_root / "gate-e-score" / "copied"
            copied_dir.mkdir()
            copied_path = copied_dir / "window_scores.json"
            copied_path.write_bytes(Path(args.selection_report).read_bytes())
            args.selection_report = str(copied_path)
            with self.assertRaisesRegex(AnalysisError, "exact frozen score path"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_full_probe_tampered_after_freeze(self):
        for label, directory in (
            ("layer", "probe-layers-full"),
            ("window", "probe-windows-full"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                args, run_root = _write_full_cli_fixture(Path(tmp))
                path = run_root / "gate-e-probe" / directory / "probe_report.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                report["tampered_after_freeze"] = True
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(
                    AnalysisError,
                    "frozen %s full-probe evidence mismatch at file_sha256" % label,
                ):
                    cmd_analyze_window_grid(args)

    def test_full_cli_rejects_missing_frozen_full_probe_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            freeze_path = (
                run_root / "control" / "provenance" / "gate-e-score-freeze.json"
            )
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            del freeze["full_probe_evidence"]
            freeze["manifest_sha256"] = manifest_sha256(freeze)
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "missing full_probe_evidence"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_four_sample_prefix_score_even_if_refrozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            _rewrite_score_as_four_sample_prefix(args, run_root)
            with self.assertRaisesRegex(AnalysisError, "complete frozen probe pool"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_mixed_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            command_path = (
                run_root / "gate-e-full" / "window-8-11-full" / "command_args.json"
            )
            command = json.loads(command_path.read_text(encoding="utf-8"))
            command["dtype"] = "bfloat16"
            command_path.write_text(json.dumps(command), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "recipe mismatch"):
                cmd_analyze_window_grid(args)

    def test_full_cli_rejects_mixed_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, run_root = _write_full_cli_fixture(Path(tmp))
            manifest_path = Path(args.run_manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_root"] = str(run_root / "different-run")
            manifest["manifest_sha256"] = manifest_sha256(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "does not match its frozen run_root"):
                cmd_analyze_window_grid(args)

    def test_analysis_rejects_tampered_window_scores(self):
        score = _score_report()
        score["candidates"][0]["scores"]["activity_r"] = 999.0
        with self.assertRaisesRegex(AnalysisError, "manifest_sha256 mismatch"):
            analyze_window_grid(
                baseline_result=_result(0.5, [1, 0, 1, 0]),
                window_results=_window_results(),
                score_report=score,
                window_grid=_grid(),
                paired_bootstrap_replicates=10,
            )

    def test_full_analysis_requires_all_frozen_comparisons(self):
        results = _window_results()
        del results["10:11"]
        with self.assertRaisesRegex(AnalysisError, "missing required window results"):
            analyze_window_grid(
                baseline_result=_result(0.5, [1, 0, 1, 0]),
                window_results=results,
                score_report=_score_report(),
                window_grid=_grid(),
                paired_bootstrap_replicates=10,
            )

    def test_misaligned_pairs_block_phase_eligibility(self):
        results = _window_results()
        results["0:1"]["samples"]["mmlu"][0]["doc_id"] = 99
        report = analyze_window_grid(
            baseline_result=_result(0.5, [1, 0, 1, 0]),
            window_results=results,
            score_report=_score_report(),
            window_grid=_grid(),
            paired_bootstrap_replicates=10,
        )
        self.assertFalse(report["phase_assessment"]["paired_sample_alignment_ready"])
        self.assertFalse(report["phase_assessment"]["eligible"])

    def test_adjacency_uses_frozen_index_order_with_non_unit_spacing(self):
        frozen = ["0:3", "8:11", "20:23"]
        self.assertTrue(_same_or_adjacent(["8:11"], ["0:3"], frozen))
        self.assertFalse(_same_or_adjacent(["20:23"], ["0:3"], frozen))
        self.assertFalse(_same_or_adjacent(["1:4"], ["0:3"], frozen))

    def test_tied_selector_gates_on_worst_case_regret(self):
        regret = _signal_regret(
            {"0:3": 1.0, "8:11": 1.0},
            {"0:3": 1.0, "8:11": -1.0},
            candidate_order=["0:3", "8:11"],
        )
        self.assertEqual(regret["selected_window"], "0:3")
        self.assertAlmostEqual(regret["absolute_regret_pp"], 0.0)
        self.assertAlmostEqual(regret["worst_case_regret_pp"], 2.0)
        assessment = _phase_assessment(
            {"available": True, "rho": 1.0},
            regret,
            {"random_in_band": {"selector_outperforms_median": True}},
            {"available": True, "stable": True},
            True,
            "full",
        )
        self.assertFalse(assessment["eligible"])
        self.assertEqual(assessment["decision"], "correlated_but_regret_too_large")
        self.assertAlmostEqual(assessment["phase_gate_regret_pp"], 2.0)

    def test_full_analysis_reports_delta_correlation_regret_and_pairs(self):
        report = analyze_window_grid(
            baseline_result=_result(0.5, [1, 0, 1, 0]),
            window_results=_window_results(),
            score_report=_score_report(),
            window_grid=_grid(),
            paired_bootstrap_replicates=100,
        )
        by_window = {item["window"]: item for item in report["windows"]}
        self.assertAlmostEqual(by_window["0:1"]["delta_acc_pp"], 25.0)
        self.assertAlmostEqual(by_window["2:3"]["delta_acc_pp"], -25.0)
        self.assertTrue(by_window["2:3"]["harmful"])
        self.assertAlmostEqual(report["signal_correlations"]["activity_r"]["rho"], 1.0)
        self.assertAlmostEqual(
            report["signal_correlations"]["entropy_flatness"]["rho"], -1.0
        )
        self.assertAlmostEqual(
            report["regret_by_signal"]["activity_r"]["absolute_regret_pp"], 0.0
        )
        self.assertAlmostEqual(
            report["regret_by_signal"]["entropy_flatness"]["absolute_regret_pp"],
            50.0,
        )

        paired = report["paired_analysis"]["0:1"]
        self.assertEqual(
            paired["counts"],
            {"both_correct": 2, "baseline_only": 0, "window_only": 1, "both_wrong": 1},
        )
        self.assertEqual(paired["contingency_table"]["n01_window_only"], 1)
        self.assertAlmostEqual(paired["mcnemar"]["p_value"], 1.0)
        self.assertEqual(report["calibration_bootstrap"]["same_or_adjacent_count"], 5)
        self.assertTrue(report["calibration_bootstrap"]["stable"])
        self.assertTrue(report["comparisons"]["random_in_band"]["complete"])
        self.assertTrue(
            report["comparisons"]["random_in_band"]["selector_outperforms_median"]
        )
        self.assertTrue(report["phase_assessment"]["eligible"])

    def test_harmful_threshold_is_strict(self):
        score = _score_report()
        score["candidates"] = [score["candidates"][0]]
        score["candidates"][0]["window"] = "0:1"
        score["primary_top_windows"] = ["0:1"]
        score["manifest_sha256"] = manifest_sha256(score)
        report = analyze_window_grid(
            baseline_result=_result(0.5, [1, 0]),
            window_results={"0:1": _result(0.497, [1, 0])},
            score_report=score,
            window_grid=None,
            paired_bootstrap_replicates=10,
        )
        self.assertAlmostEqual(report["windows"][0]["delta_acc_pp"], -0.3)
        self.assertFalse(report["windows"][0]["harmful"])

    def test_limit_smoke_does_not_rank_or_select_from_accuracy(self):
        report = analyze_window_grid(
            baseline_result=_result(0.5, [1, 0, 1, 0]),
            window_results=_window_results(),
            score_report=_score_report(),
            window_grid=_grid(),
            paired_bootstrap_replicates=10,
            analysis_scope="limit-smoke",
        )
        self.assertTrue(all(item["harmful"] is None for item in report["windows"]))
        self.assertIsNone(report["signal_correlations"]["activity_r"]["rho"])
        self.assertIsNone(report["regret_by_signal"]["activity_r"]["absolute_regret_pp"])
        self.assertEqual(report["phase_assessment"]["decision"], "engineering_smoke_only")

    def test_paired_alignment_and_exact_mcnemar(self):
        paired = paired_comparison(
            {"a": True, "b": False, "c": True},
            {"a": False, "b": True, "d": True},
            bootstrap_replicates=20,
            bootstrap_seed=1,
        )
        self.assertFalse(paired["sample_alignment"]["exact_match"])
        self.assertEqual(paired["sample_alignment"]["matched_count"], 2)
        self.assertEqual(paired["counts"]["baseline_only"], 1)
        self.assertEqual(paired["counts"]["window_only"], 1)
        self.assertAlmostEqual(exact_mcnemar_p(1, 1), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_group_accuracy_uses_effective_sample_counts(self):
        payload = {
            "results": {
                "mmlu_a": {"acc,none": 0.5},
                "mmlu_b": {"acc,none": 1.0},
            },
            "n-samples": {
                "mmlu_a": {"effective": 2},
                "mmlu_b": {"effective": 1},
            },
        }
        self.assertAlmostEqual(extract_accuracy(payload), 2.0 / 3.0)

    def test_cli_final_report_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            score_path = root / "scores.json"
            grid_path = root / "grid.json"
            baseline_path.write_text(
                json.dumps(_result(0.5, [1, 0, 1, 0])), encoding="utf-8"
            )
            score_path.write_text(json.dumps(_score_report()), encoding="utf-8")
            grid_path.write_text(json.dumps(_grid()), encoding="utf-8")
            specifications = []
            for window, payload in _window_results().items():
                path = root / (window.replace(":", "-") + ".json")
                path.write_text(json.dumps(payload), encoding="utf-8")
                specifications.append("%s=%s" % (window, path))
            output_path = root / "analysis"
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                window_result=specifications,
                selection_report=str(score_path),
                window_grid=str(grid_path),
                output_dir=str(output_path),
                task="mmlu",
                metric="acc,none",
                paired_bootstrap_replicates=20,
                paired_bootstrap_seed=20260710,
                analysis_scope="limit-smoke",
                run_manifest=None,
            )
            self.assertEqual(cmd_analyze_window_grid(args), 0)
            self.assertTrue((output_path / "selection_report.json").is_file())
            self.assertTrue((output_path / "selection_summary.md").is_file())
            with self.assertRaises(FileExistsError):
                cmd_analyze_window_grid(args)

    def test_fixture_has_every_primary_signal(self):
        for candidate in _score_report()["candidates"]:
            self.assertEqual(set(candidate["scores"]), set(PRIMARY_SIGNALS))


if __name__ == "__main__":
    unittest.main()
