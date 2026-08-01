"""Command line interface."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from tflt.audit import add_audit_args, cmd_audit_loop_effect
from tflt.config import LoopConfig
from tflt.loopscope.analysis import add_analyze_window_grid_args, cmd_analyze_window_grid
from tflt.loopscope.grid import add_make_window_grid_args, cmd_make_window_grid
from tflt.loopscope.probe import add_probe_layers_args, cmd_probe_layers
from tflt.loopscope.selection import add_score_windows_args, cmd_score_windows
from tflt.loopscope.window_probe import add_probe_window_args, cmd_probe_window
from tflt.loopscope.phase6_mmlu0_verifier import (
    dry_run_payload as dry_run_phase6_mmlu0,
    run_self_test as self_test_phase6_mmlu0,
    verify_gate_h_contracts,
)
from tflt.loopscope.phase2_analysis import (
    analysis_command_record,
    analysis_environment_record,
    analyze_phase2_evidence,
    derive_analysis_execution_provenance,
    load_phase2_analysis_evidence,
    verify_phase2_analysis_report,
)
from tflt.loopscope.phase2_trajectory import (
    add_probe_phase2_trajectory_args,
    cmd_probe_phase2_trajectory,
)
from tflt.loopscope.phase2_reuse import (
    build_phase2_freeze_candidate,
    verify_full_final_output_artifact,
)
from tflt.loopscope.phase2_schema import (
    atomic_write_new_json,
    validate_calibration_baseline_envelope,
    validate_calibration_cell_envelope,
    validate_identity_manifest,
    validate_phase2_card,
    validate_phase2_workspace_output_path,
)
from tflt.loopscope.schema import ensure_new_directory
from tflt.models import load_model_registry, resolve_model
from tflt.remote import (
    EvalSpec,
    build_lm_eval_command,
    env_snapshot,
    make_run_dir,
    render_slurm_script,
    render_ssh_script,
    shell_join,
    write_text,
)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="tflt")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect-model", help="Resolve a model alias and optionally inspect config.")
    p.add_argument("--model", required=True)
    p.add_argument("--registry", default=None)
    p.add_argument("--remote", action="store_true", help="Allow transformers to query Hugging Face.")
    p.set_defaults(func=cmd_inspect_model)

    p = sub.add_parser("smoke", help="Print or run a tiny remote smoke command.")
    add_eval_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("eval", help="Build an lm-eval-harness command.")
    add_eval_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("launch", help="Create Slurm or SSH control scripts.")
    add_eval_args(p)
    p.add_argument("--backend", choices=["slurm", "ssh"], required=True)
    p.add_argument("--profile", default=None)
    p.add_argument("--run-root", default=None)
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("collect", help="Print an rsync command for collecting remote results.")
    p.add_argument("--remote", required=True, help="Example: hpc2-hkustgz:/scratch/.../results")
    p.add_argument("--local", default="results/collected")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("report", help="Merge metrics JSON files into a summary CSV.")
    p.add_argument("--results", default="results")
    p.add_argument("--output", default="results/summary.csv")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("audit-loop-effect", help="Audit whether a loop patch changes hidden states/logits.")
    add_audit_args(p)
    p.set_defaults(func=cmd_audit_loop_effect)

    p = sub.add_parser("probe-layers", help="Collect baseline raw-logit-lens layer signals.")
    add_probe_layers_args(p)
    p.set_defaults(func=cmd_probe_layers)

    p = sub.add_parser("probe-window", help="Measure real K=2 Euler window activity/contraction.")
    add_probe_window_args(p)
    p.set_defaults(func=cmd_probe_window)

    p = sub.add_parser("make-window-grid", help="Freeze a deterministic width-4 candidate grid.")
    add_make_window_grid_args(p)
    p.set_defaults(func=cmd_make_window_grid)

    p = sub.add_parser("score-windows", help="Rank frozen windows from label-free probe signals.")
    add_score_windows_args(p)
    p.set_defaults(func=cmd_score_windows)

    p = sub.add_parser(
        "analyze-window-grid",
        help="Compare frozen signal rankings with paired evaluation results.",
    )
    add_analyze_window_grid_args(p)
    p.set_defaults(func=cmd_analyze_window_grid)

    p = sub.add_parser(
        "probe-phase2-trajectory",
        help="Collect the controlled Phase 2 no-loop boundary and 15 scalar loop cells.",
    )
    add_probe_phase2_trajectory_args(p)
    p.set_defaults(func=cmd_probe_phase2_trajectory)

    p = sub.add_parser(
        "prepare-phase2-h1",
        help="Validate the H1 V2 freeze candidate and write a local-only logical plan.",
    )
    p.add_argument("--card", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_prepare_phase2_h1)

    p = sub.add_parser(
        "validate-phase2-trace",
        help="Validate one scalar-only Phase 2 trajectory/final-output sidecar.",
    )
    p.add_argument("--trace", required=True)
    p.add_argument("--card", required=True)
    p.add_argument("--identity-manifest", required=True)
    p.set_defaults(func=cmd_validate_phase2_trace)

    p = sub.add_parser(
        "analyze-phase2-h1",
        help="Apply the frozen H1 and independent NCA decision rules.",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_analyze_phase2_h1)

    p = sub.add_parser(
        "phase6-mmlu0-gate-h",
        help="Validate the outcome-blind Phase 6 MMLU 0-shot Gate H contract.",
    )
    p.add_argument(
        "--card",
        default="configs/loopscope/phase6_mmlu0_prefix_card.json",
    )
    p.add_argument(
        "--schema",
        default="configs/loopscope/phase6_mmlu0_trajectory_schema.json",
    )
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--verify", action="store_true")
    p.set_defaults(func=cmd_phase6_mmlu0_gate_h)

    p = sub.add_parser(
        "verify-phase2-analysis",
        help="Reload canonical sources and verify a Phase 2 report by exact recomputation.",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(func=cmd_verify_phase2_analysis)

    args = parser.parse_args(argv)
    return int(args.func(args))


def cmd_prepare_phase2_h1(args: argparse.Namespace) -> int:
    card_path = Path(args.card)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    payload = build_phase2_freeze_candidate(card, Path(__file__).resolve().parents[2])
    output_dir = ensure_new_directory(Path(args.output_dir))
    atomic_write_new_json(output_dir / "phase2_freeze_candidate.json", payload)
    print(str(output_dir / "phase2_freeze_candidate.json"))
    return 0


def cmd_validate_phase2_trace(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    identity = json.loads(Path(args.identity_manifest).read_text(encoding="utf-8"))
    validate_phase2_card(card)
    validate_identity_manifest(identity, card)
    kind = payload.get("artifact_kind")
    if kind == "calibration_no_loop_baseline":
        validate_calibration_baseline_envelope(payload, card, identity)
    elif kind == "calibration_trajectory_cell":
        validate_calibration_cell_envelope(payload, card, identity)
    elif kind == "full_final_output_cell":
        verify_full_final_output_artifact(Path(args.trace), card, identity)
    else:
        raise ValueError("unsupported Phase 2 artifact kind")
    print(str(Path(args.trace)))
    return 0


def cmd_analyze_phase2_h1(args: argparse.Namespace) -> int:
    evidence = load_phase2_analysis_evidence(Path(args.input))
    output_dir = validate_phase2_workspace_output_path(
        args.output_dir, evidence["card"], context="Phase 2 analysis output directory"
    )
    if output_dir != Path(evidence["analysis_control"]["output_root"]).resolve():
        raise ValueError("analysis output directory differs from authorization")
    output_dir = ensure_new_directory(output_dir)
    atomic_write_new_json(
        output_dir / "command_args.json",
        analysis_command_record(Path(args.input), output_dir),
    )
    atomic_write_new_json(output_dir / "env.json", analysis_environment_record())
    execution_provenance = derive_analysis_execution_provenance(evidence, output_dir)
    report = analyze_phase2_evidence(
        evidence, execution_provenance=execution_provenance
    )
    atomic_write_new_json(output_dir / "phase2_h1_analysis.json", report)
    print(str(output_dir / "phase2_h1_analysis.json"))
    return 0


def cmd_verify_phase2_analysis(args: argparse.Namespace) -> int:
    verify_phase2_analysis_report(Path(args.input), Path(args.report))
    print(str(Path(args.report)))
    return 0


def cmd_phase6_mmlu0_gate_h(args: argparse.Namespace) -> int:
    card = Path(args.card)
    schema = Path(args.schema)
    if args.dry_run:
        payload = dry_run_phase6_mmlu0(card, schema)
    elif args.self_test:
        payload = self_test_phase6_mmlu0(card, schema)
    else:
        payload = verify_gate_h_contracts(card, schema)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", default="sciq")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--window", default="12:15")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--iteration-mode", choices=["block", "layer"], default="block")
    parser.add_argument("--strategy", default="damped_euler")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--cache-strategy", choices=["first", "last", "none"], default="last")
    parser.add_argument("--decode-mode", choices=["bypass", "full", "first_n"], default="bypass")
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")


def cmd_inspect_model(args: argparse.Namespace) -> int:
    info = resolve_model(args.model, args.registry)
    print(json.dumps(info, indent=2, sort_keys=True))
    if args.remote:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(info["repo_id"], trust_remote_code=True)
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True)[:4000])
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    spec = _eval_spec_from_args(args)
    cmd = build_lm_eval_command(spec)
    print(shell_join(cmd))
    if args.execute and not args.dry_run:
        return subprocess.call(cmd)
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile)
    run_root = args.run_root or profile.get("result_root", "runs")
    run_dir = make_run_dir(run_root)
    control_dir = run_dir / "control"
    output_dir = args.output_dir or str(run_dir / "eval")
    args.output_dir = output_dir
    spec = _eval_spec_from_args(args)
    command = build_lm_eval_command(spec)
    result_root = str(run_dir)

    if args.backend == "slurm":
        script = render_slurm_script(
            command=command,
            job_name="tflt-%s" % spec.model.replace("/", "-"),
            result_root=result_root,
            partition=profile.get("partition", "gpu"),
            gres=profile.get("gres"),
            gpus=int(profile.get("gpus", 1)),
            cpus=int(profile.get("cpus", 8)),
            mem=profile.get("mem", "64G"),
            time_limit=profile.get("time", "24:00:00"),
            remote_src=profile.get("remote_src"),
            hf_endpoint=profile.get("hf_endpoint", "https://hf-mirror.com"),
            hf_home=profile.get("hf_home"),
            transformers_cache=profile.get("transformers_cache"),
            hf_datasets_cache=profile.get("hf_datasets_cache"),
            uv_cache_dir=profile.get("uv_cache_dir"),
        )
        script_path = control_dir / "job.sbatch"
        write_text(script_path, script, executable=True)
        print(str(script_path))
        if args.submit and not args.dry_run:
            return subprocess.call(["sbatch", str(script_path)])
    else:
        script = render_ssh_script(
            command=command,
            result_root=result_root,
            session_name="tflt-%s" % run_dir.name,
        )
        script_path = control_dir / "run_ssh.sh"
        write_text(script_path, script, executable=True)
        print(str(script_path))

    write_text(control_dir / "env.json", json.dumps(env_snapshot(), indent=2, sort_keys=True))
    write_text(control_dir / "command.txt", shell_join(command) + "\n")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    cmd = ["rsync", "-avh", "--progress", args.remote.rstrip("/") + "/", args.local.rstrip("/") + "/"]
    print(shell_join(cmd))
    if args.execute:
        return subprocess.call(cmd)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = []
    for path in sorted(Path(args.results).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        results = payload.get("results")
        if isinstance(results, dict):
            for task, metrics in results.items():
                if isinstance(metrics, dict):
                    row: Dict[str, Any] = {"task": task, "source": str(path)}
                    row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
                    rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("task,source\n", encoding="utf-8")
        print(str(output))
        return 0
    keys = sorted({key for row in rows for key in row.keys()})
    lines = [",".join(_csv_cell(key) for key in keys)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(key, "")) for key in keys))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(output))
    return 0


def _eval_spec_from_args(args: argparse.Namespace) -> EvalSpec:
    loop_config = None
    if args.loop:
        loop_config = LoopConfig.from_window_string(
            model_alias=args.model,
            window=args.window,
            k=args.k,
            iteration_mode=args.iteration_mode,
            strategy=args.strategy,
            alpha=args.alpha,
            beta=args.beta,
            cache_strategy=args.cache_strategy,
            decode_mode=args.decode_mode,
            first_n=args.first_n,
        )
    output_dir = args.output_dir or "results/%s" % args.model.replace("/", "_")
    return EvalSpec(
        model=args.model,
        tasks=args.tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        dtype=args.dtype,
        loop_config=loop_config,
        output_dir=output_dir,
    )


def _load_profile(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    data: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", "\n", '"']):
        return '"' + text.replace('"', '""') + '"'
    return text


if __name__ == "__main__":
    raise SystemExit(main())
