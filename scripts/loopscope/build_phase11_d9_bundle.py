#!/usr/bin/env python3
"""Export a committed, code-only D9 bundle locally. Never upload or submit."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
import shlex

from phase11_d9_lingjing import make_plan


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    "phase11_d8.py", "run_phase11_accuracy.py", "phase11_d9_lingjing.py",
    "phase11_d9_lingjing_worker.py", "prepare_phase11_d9_inputs.py",
    "analyze_phase11_d9.py", "analyze_phase11_d8.py", "phase11_d9_bootstrap.py",
    "check_phase11_d9_gpu.py", "check_phase11_d8_gpu.py",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def build(output):
    print(git("status", "--short", "--branch").decode(), end="")
    branch = git("rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    commit = git("rev-parse", "HEAD").decode().strip()
    top = git("rev-parse", "--show-toplevel").decode().strip()
    if branch != "loopscope" or Path(top) != ROOT:
        raise ValueError("D9 export requires the dedicated loopscope clone")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    "4f59bd93eca4da3cbf458a93508f91c5b23912bc", commit], check=True)
    paths = ["src/tflt", "configs/loopscope/phase11_panel.json", "configs/loopscope/phase11_d9_lingjing.json",
             "configs/loopscope/phase11_d9_requirements.txt", *["scripts/loopscope/" + s for s in SCRIPTS]]
    dirty = git("status", "--porcelain", "--", *paths).decode()
    if dirty:
        raise ValueError("commit the runtime inputs before exporting: " + dirty)
    archive_bytes = git("archive", "--format=tar", commit, "--", *paths)
    output.mkdir(parents=True, exist_ok=False)
    info = {"schema": "loopscope.phase11.d9.code_bundle.v1", "source_branch": branch,
            "source_commit": commit, "source_root": str(ROOT), "contains_model_or_data": False}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as source, tarfile.open(output / "runtime.tar.gz", "w:gz") as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        for name, data in (("bundle-info.json", (json.dumps(info, indent=2)+"\n").encode()),
                           ("requirements.txt", git("show", commit + ":configs/loopscope/phase11_d9_requirements.txt"))):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    (output / "start.py").write_bytes(git("show", commit + ":scripts/loopscope/phase11_d9_bootstrap.py"))
    (output / "bundle-info.json").write_text(json.dumps(info, indent=2)+"\n")
    config = json.loads(git("show", commit + ":configs/loopscope/phase11_d9_lingjing.json"))
    cloud = Path(config["cloud_root"])
    for stage in ("preflight", "formal"):
        plan = make_plan(cloud / "runs" / stage, cloud / "inputs" / stage / "thinking-pool.json",
                         commit, 1 if stage == "preflight" else config["submission_candidate"]["train_node_num"], stage == "preflight")
        (output / f"plan-{stage}.json").write_text(json.dumps(plan, indent=2)+"\n")
        (output / f"registry-{stage}.json").write_text(json.dumps(plan["registry"], indent=2)+"\n")
    candidate = config["submission_candidate"]
    arguments = ["--bundle", str(cloud / "runtime.tar.gz"), "--plan", str(cloud / "plan-formal.json"),
                 "--model-dir", config["model_mount"], "--admission", str(cloud / "admission-formal.json"),
                 "--launcher-output", str(cloud / "launch/formal-attempt1"),
                 "--receipt-file", str(cloud / "receipt-formal.json"), "--timeout-seconds", "86400"]
    candidate["training_start_params"] = " " + shlex.join(arguments)
    (output / "submission-candidate.json").write_text(json.dumps(candidate, indent=2)+"\n")
    (output / "admission-template.json").write_text(json.dumps({
        "source_commit": commit, "model_repo": config["model"], "model_revision": config["revision"],
        "tokenizer_revision": config["revision"], "model_source_evidence": None,
        "platform_preflight_route": None, "gpqa_alignment_receipt": None, "preflight_passed": False,
        "capacity": {"max_new_tokens": 32768, "packing": 1, "headroom_fraction": None, "evidence": None}
    }, indent=2)+"\n")
    print(json.dumps({"status": "CODE_ONLY_BUNDLE_READY", "output": str(output), **info}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    build(p.parse_args().output_dir)
