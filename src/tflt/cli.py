"""Command line interface."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from tflt.audit import add_audit_args, cmd_audit_loop_effect
from tflt.config import LoopConfig
from tflt.loopscope.grid import add_make_window_grid_args, cmd_make_window_grid
from tflt.loopscope.probe import add_probe_layers_args, cmd_probe_layers
from tflt.loopscope.window_probe import add_probe_window_args, cmd_probe_window
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

    args = parser.parse_args(argv)
    return int(args.func(args))


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
