# D-3 execution evidence

Executor 01a12e14-57d6-7e51-9cfa-588f4d038352, planning 01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f.
Base c825b2e; runtime 0357d2df5020f783662aca8ccb9e21cb105cd811, loopscope. Protected runtime/wrapper/cache/strict parser and original config unchanged. D2 acceptance/control dirty state was explicitly allowed into this Gate commit.

Local commands and exit codes:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/loopscope/phase11_length_pilot.py freeze --out .planning/phase11/gate_d3_sample_membership.json`: 0.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase11_accuracy*.py'`: 0, 10 tests.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase11_length_pilot.py'`: first attempt 1 (test argument indexing typo), corrected test only; second attempt 0, 2 tests.
- `git diff --check` and `bash -n scripts/loopscope/phase11_length_pilot.sbatch`: 0.
- `git push origin loopscope`: 0, c825b2e to 0357d2d.

Remote source is a fresh real Git clone from bundle in `gate_d3_setup.json`; existing dedicated checkout was on a Phase9 historical branch and was left untouched. Remote freeze command uses the original GPQA safe input pool, producing the path in `gate_d3_remote_membership.json`. No gold, old correctness, raw prompt, generated response or candidate prediction was exported. Entire 448-input population max prompt=2819; required capacity=11011 versus native 262144. Remote targeted tests: 0, 2 tests.

Scheduler's `/usr/bin` entry initially failed for missing slurm.conf; cluster's `/opt/slurm/bin` entry works. Debug eligibility directly checked by scontrol/sacctmgr. A40 available; no A800 cost extrapolation admitted.

Synthetic submit: `/opt/slurm/bin/sbatch --parsable --partition=debug --qos=debuglimit --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:29:00 --job-name=ls-p11-d3-synthetic ... scripts/loopscope/phase11_length_pilot.sbatch SOURCE PINNED_PYTHON acquire --membership STAGE/membership.json --pool ORIGINAL_SYNTHETIC_GPQA_POOL --cell-id gpqa_main-q4-w15-18-k3-online-current-t-lambda0.9 --synthetic --run-root RUNS-synthetic1 --commit 0357d2df5020f783662aca8ccb9e21cb105cd811`: 0, job12983947. Exact expanded paths in setup and remote job Command/command_args.json. One GPU, packing1, reserved 0.483333 GPUh.

Synthetic closure: job12983947 COMPLETED 0:0, allocation73s=0.0202778 GPUh; 82 generated synthetic tokens, prefill1/decode81, frozen8192/nativeEOS, no decode direction fit, directions released. peakreserved8,153,726,976 bytes on A40 50,899,648,512 bytes. This validates functionality only, not long-sequence packing. Actual scheduler QoS was debug after debuglimit was requested.

Pilot resource canaries submitted as job12984219 Native/index208 and job12984220 K3/current/0.9/index208; exact submission commands persisted in gate_d3_job_ledger.json. Each time limit1740 seconds and1 A40, packing1. Initial aggregate actual+reserved0.98694444 GPUh. Successful pilot records will be retained, never rerun for measurement.

Native canary12984219 COMPLETED 0:0, allocation60s=0.0166667 GPUh, 1280 tokens/nativeEOS, generation46.2766s. This is an early-EOS record; it cannot size the8192 tail. K3 canary12984220 confirmed RUNNING with clean startup; sole30-minute heartbeat `gate-d-3-gpqa-length-pilot` created and visibly confirmed, exact executor self-wake/relay route frozen in gate_d3_monitor.json. No planning polling and no Gate terminal sent. Initial automation create was rejected before creation for missing destination; corrected destination=thread succeeded.

2026-10-12 terminal monitor completed one read-only check, paused automation and visibly delivered AUTOMATION_TERMINAL_RESUME to exact executor. Incoming packet was received during the same active turn; executor resumed only after monitor lifecycle receipt was preserved. K3/index208 independent raw/token/cache/direction verification passed; combined two-canary closure reports2/336 and source0357d2d. All3 jobs COMPLETED0:0,total0.0641666667 GPUh. Both pilot canaries ended below2048,so packing remains1. Follow-on Native indices4,404,25 and K3/current/0.9 indices4,404 preserve random-list order and use new roots; no successful record repeated. One SCP attempted before CPU summarizer finished and returned missing-file; retried after successful CPU closure,transfer0.

Next jobs12984987/12984988 both confirmed RUNNING for57/54s with clean model startup. Same sole30-minute heartbeat retargeted and reactivated with visible confirmation; waits for both successful terminal or any abnormal event,then pause+exact self-resume. Current actual+reserved1.0308333333 GPUh. Runtime remains0357d2d; later commits are evidence only.

Second heartbeat observed jobs12984987/12984988 COMPLETED0:0,paused and confirmed exact self-resume. Five new records independently verified;7/336 total and0.3980555556 actualGPUh. BothNative/K3 current index404 reached8192 with noEOS (right-censored),question299.8462/428.3442s; peakreserved10,892,607,488/10,932,453,376bytes on50,899,648,512byteA40. Packing2 bounded trial admitted from this measuredtail margin; new independent processpool uses original acquire path and retains batch1/cachefirst. Summary now reports all source commits to preserve original7 records while new orchestration is introduced; acquire/runtime/cache numerical path unchanged.
