#!/usr/bin/env python3
"""Export the committed D12 code and explicit resource-bound worker plans; no upload/submit."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from phase11_qwen3_nothink_launch import make_plan

ROOT = Path(__file__).resolve().parents[2]


def build(output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args])
    branch = git("rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    commit = git("rev-parse", "HEAD").decode().strip()
    if branch != "loopscope" or Path(git("rev-parse", "--show-toplevel").decode().strip()) != ROOT:
        raise ValueError("dedicated loopscope checkout required")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    "4f59bd93eca4da3cbf458a93508f91c5b23912bc", commit], check=True)
    paths = ["src/tflt", "scripts/loopscope", "configs/loopscope/phase11_panel.json",
             "configs/loopscope/phase11_qwen3_nothink_lingjing.json",
             "configs/loopscope/phase11_d9_requirements.txt"]
    if git("status", "--porcelain", "--", *paths).strip():
        raise ValueError("commit runtime source before export")
    output.mkdir(parents=True, exist_ok=False)
    info = {"source_branch": branch, "source_commit": commit, "source_root": str(ROOT),
            "contains_model_or_data": False, "gate": "D-12"}
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", commit, "--", *paths))) as original:
        with tarfile.open(output / "runtime.tar.gz", "w:gz") as target:
            for member in original.getmembers():
                target.addfile(member, original.extractfile(member) if member.isfile() else None)
            for name, data in (("bundle-info.json", json.dumps(info).encode()),
                               ("requirements.txt", git("show", commit + ":configs/loopscope/phase11_d9_requirements.txt"))):
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                target.addfile(entry, io.BytesIO(data))
    config = json.loads(git("show", commit + ":configs/loopscope/phase11_qwen3_nothink_lingjing.json"))
    config["source_commit"] = commit
    cloud = Path(config["cloud_root"])
    admission = {"source_commit": commit, "model_repo": config["model"], "runtime_authorized": True,
                 "model_provenance": config["model_provenance"],
                 "gpqa_alignment_receipt": str(cloud / "inputs/formal/alignment.json")}
    items = {"experiment.json": config, "admission.json": admission, "bundle-info.json": info}
    for stage, synthetic in (("synthetic", True), ("formal", False)):
        plan = make_plan(cloud / "runs" / stage, cloud / "inputs" / stage / "nothink-pool.json",
                         commit, config["total_cards"], synthetic)
        items[f"plan-{stage}.json"] = plan
        if not synthetic:
            items["registry-formal.json"] = plan["registry"]
            pairs = [(job["cell_key"], i) for job in plan["registry"]["jobs"] for i in job["indices"]]
            if len(pairs) != 1344 or len(set(pairs)) != 1344:
                raise ValueError("formal cell/index coverage differs")
    candidate = dict(config["submission_candidate"])
    candidate["training_start_params"] = " " + shlex.join(["--root", str(cloud), "--timeout-seconds", "86400"])
    items["submission.json"] = candidate
    for name, data in items.items():
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    (output / "start.py").write_bytes(git("show", commit + ":scripts/loopscope/phase11_qwen3_nothink_start.py"))
    print(json.dumps({"status": "BUNDLE_READY", "output": str(output), "source_commit": commit,
                      "expected_formal": 1344, "workers": config["total_cards"], "cloud_root": str(cloud)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    build(parser.parse_args().output_dir)
