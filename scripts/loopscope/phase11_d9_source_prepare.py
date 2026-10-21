#!/usr/bin/env python3
"""CPU-only, bounded source diagnosis followed by complete byte comparison."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from phase11_d9_bootstrap import remaining, run_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--timeout-seconds", required=True, type=int)
    a = p.parse_args()
    if not 0 < a.timeout_seconds <= 3600:
        raise ValueError("CPU preparation must fit the authorized 3600-second bound")
    if int(os.environ.get("RANK", "0")) != 0:
        return
    began = time.monotonic()
    deadline = began + a.timeout_seconds
    a.output_root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("phase11_d9_model_source.py")
    base = [sys.executable, str(script), "--model-dir", str(a.model_dir)]
    report = {"status": "FAILED", "cpu_only": True, "forward_executed": False,
              "model_modified": False, "gpu_requested": 0,
              "allocation_env": {k: os.environ.get(k) for k in
                  ("RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES")}}
    try:
        with (a.output_root / "probe.log").open("x") as log:
            run_step(base + ["--probe-only", "--timeout-seconds", "120", "--output",
                str(a.output_root / "network-probe.json")], deadline,
                stdout=log, stderr=log)
        probe = json.loads((a.output_root / "network-probe.json").read_text())
        candidates = [r for r in probe["endpoint_probes"] if r["status"] == "PREFIX_ONLY_EQUAL"]
        if not candidates:
            raise RuntimeError("no reachable fixed-revision endpoint with equal prefix; inspect network-probe.json")
        # Endpoint choice uses measured access latency, never model outcomes.
        chosen = min(candidates, key=lambda r: r["elapsed_seconds"])["endpoint"]
        report["selected_endpoint"] = chosen
        print(json.dumps({"event": "source_endpoint_selected", "endpoint": chosen}), flush=True)
        with (a.output_root / "compare.log").open("x") as log:
            run_step(base + ["--endpoint", chosen, "--timeout-seconds",
                str(min(3300, remaining(deadline))), "--output",
                str(a.output_root / "model-source.json")], deadline,
                stdout=log, stderr=log)
        source = json.loads((a.output_root / "model-source.json").read_text())
        if source["status"] != "PASS":
            raise RuntimeError("full source comparison did not pass")
        report.update(status="SOURCE_PREPARATION_PASS", source_evidence=str(a.output_root / "model-source.json"))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400])
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-began
        with (a.output_root / "summary.json").open("x") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
