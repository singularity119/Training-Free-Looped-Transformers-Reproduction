#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_qwen17_phase1.sh --run-root PATH --stage STAGE \
         --approval-file PASS.json [--dry-run | --submit]

Submit exactly one immutable Slurm array prepared by prepare_qwen17_phase1.py.
The default is --dry-run.  This script never retries a failed submission.

Stages and required prior decisions:
  gate-c         requires explicit Gate A PASS
  gate-d-limit   requires explicit Gate C PASS; limit=5 is engineering-only
  gate-e-probe   requires explicit Gate D PASS; full calibration probes
  gate-e-score   requires Gate D PASS plus verified gate-e-probe revisions
  gate-e-full    requires Gate D PASS plus immutable score freeze

The approval JSON must contain:
  planning_thread_id, gate, decision="PASS", approved_commit
For Gate C/D decisions it must also contain run_manifest_sha256.
USAGE
}

run_root=""
stage=""
approval_file=""
do_submit=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --stage)
      stage="$2"
      shift 2
      ;;
    --approval-file)
      approval_file="$2"
      shift 2
      ;;
    --dry-run)
      do_submit=0
      shift
      ;;
    --submit)
      do_submit=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$run_root" || -z "$stage" || -z "$approval_file" ]]; then
  usage >&2
  exit 2
fi
case "$stage" in
  gate-c|gate-d-limit|gate-e-probe|gate-e-score|gate-e-full) ;;
  *)
    echo "Unsupported stage: $stage" >&2
    exit 2
    ;;
esac

if [[ ! -d "$run_root" ]]; then
  echo "Run root does not exist: $run_root" >&2
  exit 2
fi
if [[ ! -f "$approval_file" ]]; then
  echo "Approval file does not exist: $approval_file" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    set +e
    source /etc/profile.d/modules.sh >/dev/null 2>&1
    set -e
  fi
  if command -v module >/dev/null 2>&1; then
    module load anaconda3
  fi
fi
if command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  echo "Python is required for manifest validation" >&2
  exit 2
fi

manifest="$run_root/control/phase1_run_manifest.json"
runner="$run_root/jobs/$stage/runner.sbatch"
attempt="$run_root/control/submissions/$stage-attempt.json"
receipt="$run_root/control/submissions/$stage-receipt.json"
if [[ ! -f "$manifest" || ! -f "$runner" ]]; then
  echo "Prepared manifest/runner is missing for $stage" >&2
  exit 2
fi
if [[ -e "$attempt" || -e "$receipt" ]]; then
  echo "Refusing duplicate submission; a prior attempt/receipt already exists for $stage" >&2
  exit 73
fi

validation_output="$("$python_bin" - "$manifest" "$approval_file" "$stage" "$run_root" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
approval_path = pathlib.Path(sys.argv[2]).resolve()
stage = sys.argv[3]
run_root = pathlib.Path(sys.argv[4]).resolve()
planning_thread = "019f4b5a-79ac-78c1-9196-c7fd733cf04d"
required_gate = {
    "gate-c": "A",
    "gate-d-limit": "C",
    "gate-e-probe": "D",
    "gate-e-score": "D",
    "gate-e-full": "D",
}[stage]

def load(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("JSON object required: %s" % path)
    return value

def manifest_hash(value):
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    data = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

manifest = load(manifest_path)
approval = load(approval_path)
if manifest.get("schema_version") != "loopscope.phase1-run.v1":
    raise SystemExit("unsupported phase-one run manifest")
if manifest.get("manifest_sha256") != manifest_hash(manifest):
    raise SystemExit("phase-one run manifest SHA256 mismatch")
manifest_commit = str(manifest.get("revision_policy", {}).get("manifest_commit", ""))
if not re.fullmatch(r"[0-9a-f]{40,64}", manifest_commit):
    raise SystemExit("phase-one run manifest lacks an exact model revision")
if manifest.get("revision_policy", {}).get("cli_pins_revision") is not True:
    raise SystemExit("phase-one run manifest does not pin CLI revisions")
if manifest.get("frozen_recipe", {}).get("revision") != manifest_commit:
    raise SystemExit("phase-one revision policy differs from frozen recipe")
if pathlib.Path(manifest.get("run_root", "")).resolve() != run_root:
    raise SystemExit("run root does not match its manifest")
expected_repo = pathlib.Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
expected_run_base = pathlib.Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs"
)
if pathlib.Path(manifest.get("git", {}).get("root", "")) != expected_repo:
    raise SystemExit("run manifest does not use the exact dedicated HPC2 LoopScope clone")
if run_root.parent != expected_run_base or not re.fullmatch(
    r"loopscope-qwen17-mmlu-phase1-[0-9]{8}-[0-9]{6}", run_root.name
):
    raise SystemExit("run root is outside the exact protected HPC2 run base/prefix")
venv = pathlib.Path(manifest.get("venv", ""))
if venv.parent != expected_repo or not re.fullmatch(
    r"\.venv-loopscope-cu121-[0-9]{8}(?:-v[0-9]+)?", venv.name
):
    raise SystemExit("run manifest venv is not versioned inside the dedicated clone")
if approval.get("planning_thread_id") != planning_thread:
    raise SystemExit("approval is not from the designated planning thread")
if str(approval.get("gate", "")).upper() != required_gate:
    raise SystemExit("%s requires Gate %s PASS" % (stage, required_gate))
if approval.get("decision") != "PASS":
    raise SystemExit("an exact PASS decision is required; PASS_WITH_FIXES is not sufficient")
if approval.get("approved_commit") != manifest.get("git", {}).get("commit"):
    raise SystemExit("approval commit does not match the prepared checkout commit")
if required_gate != "A" and approval.get("run_manifest_sha256") != manifest.get("manifest_sha256"):
    raise SystemExit("approval does not bind the current immutable run manifest")

stage_info = manifest.get("stages", {}).get(stage)
if not isinstance(stage_info, dict) or stage_info.get("requires_gate") != required_gate:
    raise SystemExit("stage gate metadata is invalid")
jobs = stage_info.get("jobs")
if not isinstance(jobs, list) or not jobs:
    raise SystemExit("stage contains no jobs")
for job in jobs:
    output = pathlib.Path(job["output_dir"])
    claim = pathlib.Path(job["claim_dir"])
    if output.parent != run_root / stage or output.name != job.get("job_id"):
        raise SystemExit("job output escapes its immutable stage directory")
    if claim != pathlib.Path(str(output) + ".claim"):
        raise SystemExit("job output claim path is invalid")
    if output.exists():
        raise SystemExit("refusing to reuse existing output: %s" % output)
    if claim.exists():
        raise SystemExit("refusing duplicate execution; output claim exists: %s" % claim)
    if job.get("automatic_retry") is not False:
        raise SystemExit("job does not explicitly disable automatic retry: %s" % job.get("job_id"))

def revision_jobs(report_stage):
    stages = manifest["stages"]
    result = []
    if report_stage != "gate-c":
        result.append(stages["gate-c"]["jobs"][0])
    if report_stage in ("gate-e-probe", "gate-e-full"):
        result.append(stages["gate-d-limit"]["jobs"][0])
    if report_stage == "gate-e-full":
        result.extend(stages["gate-e-probe"]["jobs"])
    result.extend(stages[report_stage]["jobs"])
    return result

def revision_contract(job):
    return {
        "job_id": str(job["job_id"]),
        "stage": str(job["stage"]),
        "artifact": str(job["revision_artifact"]),
        "revision_keys": {
            str(name): [str(value) for value in path]
            for name, path in job["revision_keys"].items()
        },
    }

def validate_revision_report(report_stage):
    path = run_root / "control" / "provenance" / (report_stage + "-model-revisions.json")
    report = load(path) if path.is_file() else None
    if not report or report.get("schema_version") != "loopscope.model-tokenizer-revision-check.v2":
        raise SystemExit("%s model-revision report is missing/unsupported" % report_stage)
    if report.get("manifest_sha256") != manifest_hash(report):
        raise SystemExit("%s model-revision report canonical hash mismatch" % report_stage)
    if report.get("run_manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("%s model-revision report is bound to another run" % report_stage)
    if report.get("stage") != report_stage:
        raise SystemExit("model-revision report stage mismatch")
    expected = [revision_contract(job) for job in revision_jobs(report_stage)]
    if report.get("expected_jobs") != expected:
        raise SystemExit("model-revision expected jobs differ from run manifest")
    observations = report.get("observations")
    if not isinstance(observations, list) or len(observations) != len(expected):
        raise SystemExit("model-revision observation count mismatch")
    manifest_commit = str(manifest.get("revision_policy", {}).get("manifest_commit", ""))
    if report.get("manifest_commit") != manifest_commit:
        raise SystemExit("model-revision report manifest commit mismatch")
    revisions = []
    for observation, contract in zip(observations, expected):
        observed = {
            "job_id": str(observation.get("job_id", "")),
            "stage": str(observation.get("stage", "")),
            "artifact": str(observation.get("artifact", "")),
            "revision_keys": contract["revision_keys"],
        }
        if observed != contract:
            raise SystemExit("model-revision observation job/artifact mismatch")
        commits = [
            str(observation.get(key, "")).strip()
            for key in ("model_commit", "tokenizer_commit", "manifest_commit")
        ]
        if any(not value for value in commits):
            raise SystemExit("model/tokenizer/manifest revision observation is empty")
        if observation.get("match") is not True or len(set(commits)) != 1:
            raise SystemExit("model/tokenizer/manifest revisions differ")
        if commits[0] != manifest_commit:
            raise SystemExit("resolved revisions differ from run manifest")
        revisions.extend(commits)
    unique = sorted(set(revisions))
    if report.get("unique_revisions") != unique or report.get("match") is not True:
        raise SystemExit("model-revision report does not prove one exact revision")
    if len(unique) != 1:
        raise SystemExit("model/tokenizer/manifest revisions differ")
    return report

gate_e_probe_report = None
if stage == "gate-d-limit":
    validate_revision_report("gate-c")
if stage in ("gate-e-probe", "gate-e-score", "gate-e-full"):
    validate_revision_report("gate-d-limit")
if stage in ("gate-e-score", "gate-e-full"):
    gate_e_probe_report = validate_revision_report("gate-e-probe")

if stage == "gate-e-full":
    freeze_path = run_root / "control" / "provenance" / "gate-e-score-freeze.json"
    freeze = load(freeze_path) if freeze_path.is_file() else None
    if not freeze or freeze.get("schema_version") != "loopscope.score-freeze.v1":
        raise SystemExit("immutable gate-e score freeze is missing")
    if freeze.get("manifest_sha256") != manifest_hash(freeze):
        raise SystemExit("score-freeze canonical hash mismatch")
    if freeze.get("run_manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("score freeze is bound to another run")
    score_job = manifest["stages"]["gate-e-score"]["jobs"][0]
    score_path = pathlib.Path(score_job["output_dir"]) / "window_scores.json"
    if freeze.get("score_job_id") != score_job["job_id"] or freeze.get(
        "score_report_path"
    ) != str(score_path):
        raise SystemExit("score freeze job/path mismatch")
    score = load(score_path) if score_path.is_file() else None
    if not score or score.get("manifest_sha256") != manifest_hash(score):
        raise SystemExit("window score report canonical hash mismatch")
    if hashlib.sha256(score_path.read_bytes()).hexdigest() != freeze.get(
        "score_report_sha256"
    ):
        raise SystemExit("window score report file SHA256 changed after freeze")
    if score.get("manifest_sha256") != freeze.get("score_report_manifest_sha256"):
        raise SystemExit("score report manifest SHA256 differs from freeze")
    if freeze.get("criterion_sha256") != manifest["inputs"]["criterion"][
        "criterion_sha256"
    ] or score.get("criterion_sha256") != freeze.get("criterion_sha256"):
        raise SystemExit("frozen criterion SHA256 mismatch")
    criterion_path = pathlib.Path(manifest["inputs"]["criterion"]["snapshot_path"])
    criterion = load(criterion_path) if criterion_path.is_file() else None
    if not criterion or manifest_hash(criterion) != freeze.get("criterion_sha256"):
        raise SystemExit("frozen criterion content changed")
    if hashlib.sha256(criterion_path.read_bytes()).hexdigest() != freeze.get(
        "criterion_file_sha256"
    ):
        raise SystemExit("frozen criterion file SHA256 changed")
    if freeze.get("window_grid_manifest_sha256") != manifest["inputs"]["window_grid"][
        "manifest_sha256"
    ] or score.get("window_grid", {}).get("manifest_sha256") != freeze.get(
        "window_grid_manifest_sha256"
    ):
        raise SystemExit("frozen window-grid SHA256 mismatch")
    grid_path = pathlib.Path(manifest["inputs"]["window_grid"]["snapshot_path"])
    grid = load(grid_path) if grid_path.is_file() else None
    if not grid or grid.get("manifest_sha256") != manifest_hash(grid):
        raise SystemExit("frozen window-grid canonical hash mismatch")
    candidate_windows = [str(item["window"]) for item in grid.get("windows", [])]
    if freeze.get("candidate_windows") != candidate_windows or score.get(
        "window_grid", {}
    ).get("candidate_windows") != candidate_windows:
        raise SystemExit("score freeze candidates differ from the frozen grid")
    if freeze.get("gate_e_probe_revision_report_sha256") != gate_e_probe_report.get(
        "manifest_sha256"
    ):
        raise SystemExit("score freeze is not bound to verified full probes")

if stage == "gate-d-limit" and stage_info.get("limit") != 5:
    raise SystemExit("Gate D must remain a limit=5 engineering smoke")
if stage == "gate-d-limit" and stage_info.get("scientific_selection_allowed") is not False:
    raise SystemExit("Gate D limit accuracy must not be used for window selection")

repo_root = pathlib.Path(manifest["git"]["root"])
print(str(repo_root))
print(str(manifest["git"]["commit"]))
print(str(manifest["manifest_sha256"]))
print(hashlib.sha256(approval_path.read_bytes()).hexdigest())
print(len(jobs))
PY
)"

repo_root="$(printf '%s\n' "$validation_output" | sed -n '1p')"
approved_commit="$(printf '%s\n' "$validation_output" | sed -n '2p')"
run_manifest_sha256="$(printf '%s\n' "$validation_output" | sed -n '3p')"
approval_sha256="$(printf '%s\n' "$validation_output" | sed -n '4p')"
job_count="$(printf '%s\n' "$validation_output" | sed -n '5p')"

repo_root="$(cd "$repo_root" && pwd -P)"
git_root="$(git -C "$repo_root" rev-parse --show-toplevel)"
git_root="$(cd "$git_root" && pwd -P)"
if [[ "$git_root" != "$repo_root" ]]; then
  echo "Git root mismatch: $repo_root" >&2
  exit 2
fi
if [[ "$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)" != "loopscope" ]]; then
  echo "Remote checkout is not on loopscope" >&2
  exit 2
fi
if [[ "$(git -C "$repo_root" rev-parse HEAD)" != "$approved_commit" ]]; then
  echo "Remote checkout commit changed after preparation" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Remote checkout is dirty; refusing submission" >&2
  exit 2
fi

echo "stage=$stage"
echo "jobs=$job_count"
echo "runner=$runner"
case "$stage" in
  gate-c) required_gate="A" ;;
  gate-d-limit) required_gate="C" ;;
  gate-e-probe) required_gate="D" ;;
  gate-e-score) required_gate="D" ;;
  gate-e-full) required_gate="D" ;;
esac
echo "required prior gate=$required_gate"
if [[ "$stage" == "gate-d-limit" ]]; then
  echo "Gate D is limit=5 engineering smoke only; its accuracy cannot select a window."
elif [[ "$stage" == "gate-e-probe" ]]; then
  echo "Gate E probe uses the complete calibration pool; Gate C samples are not a substitute."
elif [[ "$stage" == "gate-e-score" ]]; then
  echo "Offline scoring requires the verified full-pool probe stage."
elif [[ "$stage" == "gate-e-full" ]]; then
  echo "Full grid is authorized only by the supplied explicit Gate D PASS."
fi

if [[ "$do_submit" -ne 1 ]]; then
  printf 'DRY RUN: sbatch %q\n' "$runner"
  exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available" >&2
  exit 2
fi

"$python_bin" - \
  "$attempt" "$stage" "$run_manifest_sha256" "$approval_file" "$approval_sha256" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "loopscope.submission-attempt.v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "stage": sys.argv[2],
    "run_manifest_sha256": sys.argv[3],
    "approval_file": str(pathlib.Path(sys.argv[4]).resolve()),
    "approval_sha256": sys.argv[5],
    "automatic_retry": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

set +e
sbatch_output="$(sbatch --parsable "$runner" 2>&1)"
sbatch_status=$?
set -e

"$python_bin" - "$receipt" "$stage" "$run_manifest_sha256" "$sbatch_status" "$sbatch_output" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "loopscope.submission-receipt.v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "stage": sys.argv[2],
    "run_manifest_sha256": sys.argv[3],
    "sbatch_exit_code": int(sys.argv[4]),
    "sbatch_output": sys.argv[5],
    "automatic_retry": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf '%s\n' "$sbatch_output"
if [[ "$sbatch_status" -ne 0 ]]; then
  echo "Submission failed once and was not retried; preserve the receipt and report it." >&2
  exit "$sbatch_status"
fi
echo "Submission recorded. Do not resubmit this stage."
