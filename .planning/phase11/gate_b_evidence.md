# Phase11 Gate B execution evidence

Executor `01a11af4-0dbc-7fd0-92ec-bb66169ea433/local`; planning
`01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local`. Authority is the exact Gate B
handoff and control, with Gate A PASS at `c60c091`. No executor PASS decision.

## Source preparation and engineering increment

The nested local repository, `loopscope` branch, HEAD and fixed ancestor were
verified before grouped writes and Git operations. The initial dirty state was
only planning's control/acceptance/handoff. Protected core and science unchanged.
Worker `resource_script` owned only the resource script, launcher and its tests;
the bound executor reviewed/integrated them and retained Git/remote/GPU authority.

Implementation commit `ffb1e3824e85484e346b1090b7b5f6f572f6b4f3` adds the
minimal synthetic resource script/launcher and CUDA lambda0.1/0.9 application
checks, plus the already authorized planning files. Normal `git push origin
loopscope` completed exit0. Targeted command
`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase11*.py'`
ran55 tests, exit0; both resource/preflight `bash -n` and `git diff --check`
completed exit0. New synthetic orchestration launcher also passes `bash -n`.

Full authentic local Git object/ref databases were transported into fresh
Gate B staging, then `git clone --no-hardlinks --branch loopscope` created
independent compute sources; no fabricated `.git`/HEAD or source-check bypass.
Initial local bundle preparation stalled and was stopped; it consumed no GPU.
Mac transport metadata in the first private staging was removed before cloning.
New sources have actual clean `loopscope` HEAD and fixed-ancestor checks, exit0:

- base source: `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-b-20261008T100900Z-c60c091/cuda-source`
- current source: `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-b-20261008T101100Z-ffb1e38/cuda-source`
- pinned Python: `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python`
- logs: `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/logs/phase11-gate-b-20261008T100500Z`

In commands below, `<current-source>`, `<pinned-python>` and `<logs>` expand
to these exact paths; `<current-staging>` is the current source's parent
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-b-20261008T101100Z-ffb1e38`.

SSH uses BatchMode/ConnectTimeout10/ClearAllForwardings=yes. The old source
remains clean `phase9-gate-c-20260922` at `a2c9e58`; no old environment install.
Remote metadata confirms torch2.3.1+cu121, transformers4.51.3, lm_eval0.4.11,
datasets5.0.0 and the new source's runtime module path. Remote resource CPU
tests ran10, exit0; strict-source resource dry-run completed exit0 without
creating a run or loading weights. These facts are not CUDA/resource evidence.

## Scheduler attempts

Normal-user debug partition admission was inspected with `sinfo`, `scontrol`
and `sacctmgr`: one A40 GPU node, maximum30min, user permits debug/debuglimit,
debug QoS max2GPU/user and8 jobs/account. Every B submission requests29min,
1A40 GPU,8CPUs,64G host memory; concurrent B allocations capped2 and budget12GPUh.

1. Job12946490: original c60 checker, requested debug/debug QoS. Queued as
   `MaxJobsPerAccount`, no allocation/model forward. Own pending QoS update was
   rejected (`debug part can not extend time`); executor cancelled the still
   pending job, scheduler reports `CANCELLED by 204599`, elapsed00:00:00.
2. Job12946859: current ffb checker, requested debug/debuglimit. Cluster forces
   actual QoS=debug. After the ordinary debug queue wait, it completed on gpu3-9
   in70s, scheduler COMPLETED/exit0. The next admission observation already saw
   completion, so no retrospective heartbeat was created. CUDA checker elapsed
   58.549s; A40 total50,899,648,512 bytes, allocated/reserved peaks
   8,136,028,160/8,160,018,432 bytes. This small functional shape cannot size packing.
3. Job12946972: combined sequential synthetic producer/verifier and four
   packing-one resource cases, same ffb source/Python/debug1GPU8CPU64G29min.
   Run prefix `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-b-20261008T103800Z-prod-res1`.
   Outer launcher is preserved at
   `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-b-20261008T101100Z-ffb1e38/gate_b_producer_resources.sbatch`;
   its direct runtime children are the unchanged audited preflight/resource
   launchers. Source branch/HEAD/clean/ancestor and outer shell syntax passed
   before submission. Resource cases start only after all three producer
   verifications succeed; all remain synthetic/PREFLIGHT_ONLY.

Checker command after source admission:

```bash
sbatch --parsable --partition=debug --qos=debuglimit --gres=gpu:a40:1 \
  --cpus-per-task=8 --mem=64G --time=00:29:00 --job-name=p11-b-check2 \
  --output=<logs>/check2-%j.out --error=<logs>/check2-%j.err \
  <current-source>/scripts/loopscope/phase11_preflight.sbatch <current-source> <pinned-python> \
  check_phase11_gpu.py --run-root /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-b-20261008T101100Z-check2 \
  --commit ffb1e3824e85484e346b1090b7b5f6f572f6b4f3 --config configs/loopscope/phase11_panel.json
```

## CUDA checker outcome

All six comparisons closed with max absolute error0: Native original/monitored,
LoopK2/current-lambda0 and K2fixed/lag1, for actual greedy generation logits/IDs
and ARC complete multi-token HFLM scores/greedy flags. Eleven generation cases
each exercised prefill+two incremental decodes and all36 cache layers; every
loop forward has K body calls and one stash pass. K3fixed/lag1/current prefill
fit counts are1/2/3, decode fit count0, frozen banks unchanged, directions
released per question and nonanswer residual changes0. Strength0.1/0.9 both
reached actual current_t CUDA callbacks with positive target residual change.
Only scalar comparisons and synthetic IDs are saved; no target gold loaded.

## Synthetic producer and verifier closure

Job12946972 completed in374s, exit0. ARC, MMLU-Pro and GPQA each produced
two synthetic records and independently closed `ATTEMPT_RAW_VERIFIED`,
`scope=PREFLIGHT_ONLY`, `target_gold_loaded=false`, source `ffb1e38`.
Each cell is its task's `q4-w15-18-k3-online-current-t-lambda0.9`.
MMLU/GPQA each record has one prefill plus two true single-token cached decode
forwards, three generated tokens, closed cache growth, no decode SVD and a
released direction bank. ARC scores its full synthetic candidates through native
HFLM; the separate CUDA checker establishes exact full-candidate equivalence.

Producer/verifier commands and all four packing-one measurements are preserved
in `gate_b_synthetic_launcher.sbatch`; the exact executed remote copy is
`<current-staging>/gate_b_producer_resources.sbatch`. Its single argument is the
prod-res1 run prefix stated above; current source, pinned Python, ffb1e38 commit
and A synthetic input root are hard-bound in that archived launcher.
The remote source remains clean/immutable.
Each `<prod-res1>-<dataset>-attempt` contains raw closure; each
`<prod-res1>-<dataset>-verification/attempt_verification.json` contains verifier
closure. `gate_b_resource_facts.json` preserves fresh scalar closure/telemetry
and the precise paths without copying raw outputs into Git.

## Representative packing-one resources

All cases completed without OOM on one NVIDIA A40 (50,899,648,512 bytes).
Generation prompt2860 is the A-bound maximum MMLU-Pro prompt shape; GPQA's
bound2819 is covered by the larger engineering shape, not a separate long GPQA
measurement. Synthetic ordinary token content does not represent actual SVD
conditioning or EOS lengths. Online is K3/current_t/lambda0.9.

| Case | Load s | Prefill/scoring s | Decode s | Question s | Token/s | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native, 2860+2048 | 6.291 | 0.591 | 68.176 | 73.662 | 27.803 generated | 8.304 / 8.518 |
| Loop K3, 2860+2048 | 6.273 | 0.588 | 93.142 | 98.508 | 20.790 generated | 8.304 / 8.539 |
| Online K3, 2860+2048 | 7.720 | 10.208 | 95.949 | 110.885 | 18.470 generated | 8.304 / 8.549 |
| ARC Online K3, four full candidates | 11.142 | 7.663 | 0 | 7.698 | 23.902 scored | 8.646 / 9.006 |

All generation cases reached exactly2048 generated tokens, one prefill and
2047 incremental decodes. Every one of36 cache layers ends at4907 tokens;
the last generated token has not yet been fed back. Online records6144 callbacks,
2048 per t, exactly one prefill fit per t, zero decode fits and direction release.
The three CUDA gesvd fits have2860 retained rows ×2560 hidden width and take
3.397/3.087/3.087s. BF16 original residual is converted to FP32 for exact SVD.

ARC uses prefix1242 and continuation46 for each of four candidates; native HFLM
receives1287 input tokens per candidate and scores184 continuation tokens total.
Directions fit only1242 prompt rows; four forwards,12 fits/callbacks, no decode.
This combines A's separate maximum prompt/continuation lengths, an engineering
envelope, not a claim that one actual ARC candidate has both maxima.
The resource run roots are `<prod-res1>-resource-<dataset>-<family>`.
Per-stage allocated/reserved values and synchronized forward timings are in
the facts JSON; question timing includes per-question instrumentation overhead.

## Independent-process packing two

After packing-one completion/monitor pause, job12947039 ran the archived
`gate_b_packing2_launcher.sbatch`, same immutable source/Python, one allocated
A40,8CPUs,64G,29min, debug/debuglimit request with actual QoSdebug.
It completed245s, exit0. Command shape:

```bash
sbatch --parsable --partition=debug --qos=debuglimit --gres=gpu:a40:1 \
  --cpus-per-task=8 --mem=64G --time=00:29:00 --job-name=p11-b-pack2 \
  --output=<logs>/pack2-%j.out --error=<logs>/pack2-%j.err \
  <current-staging>/gate_b_packing2_launcher.sbatch \
  /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-b-20261008T105400Z-pack2
```

Two separate Python model processes share the allocated GPU with batch1 each,
four CPU threads each and no model/tensor sharing. Separate output roots and
PID/exit records are preserved. Both generation children reached2048 tokens and
all36 cache layers4907; both ARC children scored four full candidates. All four
children exit0 with no OOM, no decode SVD and released banks.

| Pair | Child question s | Aggregate question-window token/s | Startup-inclusive token/s | Sum of process reserved peaks GiB |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro Online K3 | 201.991 / 201.946 | 20.278 generated | 18.789 | 17.063 |
| ARC Online K3 | 15.606 / 15.723 | 23.404 scored | 14.154 | 18.012 |

The sum of process allocator peaks is not a measured device-wide peak; CUDA
contexts and other nonallocator memory are excluded. The generation aggregate
is9.8% above the single packing-one point, with each process about10.14token/s.
ARC steady aggregate is2.1% lower: this probe establishes capacity, not a
steady ARC throughput improvement. Pair wall including startup is218/26s.
No repetitions or statistical claim. A800 packing must be directly measured in
an authorized C admission; these A40 points do not freeze A800 packing.

## C time/cost scenarios and uncertainty

These are modeled A40-equivalent packing-one scenarios, not an authorized C
budget, runtime forecast or strict bound. All12480 generation identities use
the2860-token synthetic prompt. For an assumed output length T, interpolate
`q(T)=prefill_s+(T-1)*(question_s-prefill_s)/2047` from the three measured
families. The full panel proxy uses one Native +two Loop K3 +15 Online K3/current_t
arms per identity, deliberately replacing cheaper K2/fixed/lag1 variants by the
measured K3/current_t family. ARC adds18×1172×7.698120s=45.111GPUh by replacing
every arm with the one measured four-candidate Online shape.

| Assumed output tokens per generation identity | Generation GPUh | ARC proxy GPUh | Total A40-equivalent GPUh | Ideal two-GPU elapsed days at packing one |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1305.254 | 45.111 | 1350.365 | 28.133 |
| 512 | 2076.560 | 45.111 | 2121.671 | 44.201 |
| 2048 | 6704.394 | 45.111 | 6749.505 | 140.615 |

Interpolation assumes a constant average marginal decode cost; shorter cache
lengths, real prompt distributions, EOS lengths, SVD conditioning and removal
of synchronized debug instrumentation can change it materially. Model loads,
queue time, failures, output/verification I/O and formal analysis are excluded.
Packing-two gain is only an A40 Online single point and is not applied to the
full panel. A800 throughput is unmeasured. Consequently a defensible C budget
requires direct authorized A800 measurements and actual outcome-blind length
observations; no A800 speedup or currency cost is invented here.

## Allocation budget, monitoring and terminal boundary

Allocated jobs12946859/12946972/12947039 used70/374/245 GPU-seconds;
12946490 was cancelled pending with zero allocation. Total689/3600 =
0.191388889GPUh, below12GPUh; maximum concurrent B allocations one GPU,
below two. All three allocated jobs are COMPLETED/exit0; zero OOM.

One executor-owned heartbeat `phase11-gate-b-resource-probe`,30min,
was admitted after prod-res1 was RUNNING1:44 with no startup failure. The active
executor observed its terminal, paused the monitor before the follow-on, then
retargeted the same ID after packing-two was RUNNING2:07 with healthy children.
The active executor observed packing-two terminal and paused the same monitor;
automation_update explicitly confirmed PAUSED. No heartbeat woke before this
short probe finished, so no separate terminal self-resume was needed. There is
no active B monitor. Lifecycle/tool confirmations are in `gate_b_monitor.json`.

The executor requests planning acceptance on the three decisive groups:
CUDA equivalence/cache/direction application, three synthetic producer/verifier
closures, and representative full-limit resources including independent packing.
No executor PASS decision. No formal test forward, target gold/outcome, accuracy
analysis, Gate C, protected runtime modification, old source/environment mutation
or model/data download on Mac. The frozen science and accepted GPQA source
binding remain unchanged. The subsequent evidence commit changes only B
evidence/orchestration records and documentation; all executed runtime evidence
remains bound to ffb1e38.
