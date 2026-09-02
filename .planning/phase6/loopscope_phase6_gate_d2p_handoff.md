# GATE_D2P_HANDOFF

Project/phase: LoopScope Phase 6  
Gate: D-2P — exploratory partial selector on six verified shards  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor thread: `019fbe3e-1815-7b13-b243-4fd418c6a748`  
Executor title: `execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate D-2P`  
State: `AUTHORIZED`  

## Objective and status

Run exactly one outcome-blind V2 selector analysis on the six currently verified Gate D-2 shards.
This is a user-requested exploratory sidecar, not early Gate E. Its output must be labeled
`EXPLORATORY_PARTIAL_6_OF_8` and has no authority over the official Phase 6 selected window, panel,
Gate E admission, or later outcome acquisition.

The main Gate D-2 executor remains independently authorized and continues acquisition/recovery.
Do not message it for work, alter its jobs/heartbeat, or mutate its run root.

## Authority and admission

Before any input read or mutation, require:

- policy: `loopscope-tflt/AGENTS.md`, SHA-256
  `744fb7c529f713d3c13ffdb08e42483186e08045fce5bb59656cb365b0488645`;
- control: `.planning/loopscope_phase6_control.md`, SHA-256
  `dfe5c896912ff14e1c3a1ab73e6265acf363bc62a12f525526a403d8c6b17708`;
- this handoff and exact executor binding;
- repo/branch/HEAD/origin: `loopscope-tflt` / `loopscope` /
  `f759525a7a4aebbec0a101a2ebfd1018b248ae05`;
- expected local dirty state: only protected unstaged `M AGENTS.md`;
- source root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-d2-amendment3-formal8-4day-20260801T113800Z`.

Use code from exact commit `f759525a7a4aebbec0a101a2ebfd1018b248ae05`. If concurrent Gate D-2
work could move a shared checkout, use an isolated read-only/detached copy at that commit; do not
change the main D-2 checkout or repository branch. Do not stage, commit, push, or modify repository
files in this Gate.

## Exact allowed inputs

Read only the following sanitized trajectory files and their corresponding attempt/verifier
receipts. Recompute every file SHA-256 before analysis and stop on any mismatch.

| Shard | Sanitized records SHA-256 | Attempt receipt SHA-256 | Verifier receipt SHA-256 | Records | Eligible trajectories |
|---:|---|---|---|---:|---:|
| 0 | `216fd7e3f947159b1d121ac923649896867fdcf209a9068f1bb5d250887fe874` | `7d8d31c94fae543f1e471eb29515f491ca6e766c72e74c4a97f1c0498bb4bb62` | `89757384007e7f438b498a7457f3696db139858eecefebc7d8684348ef5c9967` | 1504 | 1450 |
| 2 | `979828400aa17b2f6c90d5cab81947a113871558c2707823ac8cfe75a46788a5` | `5864a31ad0972f563e6842a643e75213bf0987190a0495c126b0f8b4dd4acdc1` | `72954f429c0b8905028fb1d00827bf8fec7fac0e336115c8630a198a663220b4` | 1504 | 1433 |
| 4 | `10a77251a5d878929294e0928b64ccd543c050115d4e856ece5e02748506e2d0` | `181b4e6554b6b00a55869a5f2f9e546c427b8257224c9e083d6b7dd403651a0a` | `c15f8113d9e4e7a9c12393001cc929498cf2ffc0bf708bfd68b613672d8b004f` | 1504 | 1447 |
| 5 | `c867889502066ab543ce128b467e45284e76ae29621528f44827ce5ad6651cbe` | `6ad50545b5ff009b115e48a239e7048048fb306e42eabde41aefb9d76de9354d` | `06f61cc82b271e5bf4b64eaf1ea24dd58cdcd962176f177b6aacf3e1bc310fa2` | 1504 | 1431 |
| 6 | `a1a0cedc9158b6a3cb27857a3bc5607ae873640cd43e8ca6f673c10840428ada` | `771472cc2d16bfdd073973454f72d36aac30caa134f067dc4381f6a6d2e1965b` | `7da0552cd669b139c047cd03c6247f1b9e6ca51926eb32dd89d6e3e45db07543` | 1504 | 1444 |
| 7 | `3ad2c84a9ae6724e2e826fc54e3efa5d7de0694b2db11709e492b74007496cb2` | `b125a640f52057bceb2dbf00c9f6f4c16483d55e39e577cdbde5a588062a809c` | `37395c0397c75f39d3f63fcb977fc157b7b2acc51e901404c6f4c3332e13fd3a` | 1504 | 1436 |

For shard `NNNN`, the exact relative paths are:

```text
formal/shards/shard-NNNN/attempt-0001/sanitized_records.jsonl
formal/shards/shard-NNNN/attempt-0001/receipt.json
formal/shards/shard-NNNN/attempt-0001/verifier_receipt.json
```

Also bind:

- `formal/shard_manifest.json` file SHA-256
  `5b19044f4034a81160a5e3371cd8d5b8bf33e1804427d0c479f4666e0a1a33e5`;
- `formal/freeze_receipt.json` file SHA-256
  `b078e98e76a485bc6b1254b28fedf6d33173a3073e4bb674159af8d1c8f56fbf`;
- full frozen population 12,032, shard count 8, full ordered identity SHA-256
  `ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5`;
- sidecar subset: shard IDs `0,2,4,5,6,7`, missing `1,3`, 9,024 records and exactly
  8,641 eligible sanitized trajectories.

Each verifier receipt must independently remain `PASS` with
`opaque_payload_content_parsed=false`, `selector_executed=false`, and `outcomes_read=false`.

## Frozen selector contract

Merge only the six sanitized files, reject duplicate identities, preserve deterministic canonical
order, and project to the selector-allowed fields only: canonical identity, category, full-vocabulary
entropy trajectory `H`, and `D=KL(p_l||p_36)`.

Run the existing Phase 6 `RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE` implementation without
changing it:

- central blocks `11...24`;
- widths `3,4,5,6`, exactly 42 candidates;
- unchanged V1 `NewEligible` admission and biphasic stability rules;
- `S_RATE=sqrt(G_H*G_K)` absolute-rate ranking;
- category-stratified joint bootstrap, 2,000 replicates, seed `20260801`, `ddof=1`, epsilon
  `1e-12`, frozen linear quantile and RateStable rule;
- unchanged top-tie, selection-frequency `>=0.80`, registry exclusion, and all legitimate
  `ABSTAIN` semantics;
- hidden/geometry/angular fields remain diagnostic-only and must not enter selector scoring.

The bootstrap and point estimates are over this fixed 8,641-trajectory subset only. Do not impute,
weight, resample, or search for missing shards 1/3. Do not compare against any partial/full outcome.

## Outputs and verification

Create one new write-once sibling root named
`phase6-gate-d2p-partial-selector-<UTC>` under the LoopScope runs root. It may contain only:

- a hash-closed `partial_input_manifest.json` listing the six exact sources and missing shards;
- an exact driver/script copy if needed, with SHA-256 and exact-commit provenance;
- `partial_selector_analysis.json` with the exploratory decision/ranking and no outcome fields;
- `partial_selector_verifier_receipt.json` from a separate fresh process that reloads the six
  sources, recomputes the selector, and proves identical decision/ranking/digests;
- `partial_selector_summary_zh.md` prominently labeled `EXPLORATORY_PARTIAL_6_OF_8`, including
  shard list, missing shards, 9,024/8,641 counts, selected or `ABSTAIN` state, and a concise warning
  that this is not the official Phase 6 window.

Do not create `phase6_selector_freeze.json`, an official panel manifest, or any Gate F-consumable
artifact. Do not edit D-2 files. Preserve exact output hashes in the terminal packet.

## Information barrier and forbidden actions

Do not read, list contents of, copy, hash individual files from, or otherwise inspect sealed baseline
payloads. Do not access prompt/completion text, token IDs, answer content, gold, label, correctness,
accuracy, gain, loop outcome, Phase 4 outcome values, or unverified/live shard 1/3 data. Do not load
the model/dataset, run forward/generation/replay, use GPU/CUDA, cancel/retry/requeue D-2 jobs, modify
D-2 automation, or enter Gate E/F/G.

CPU-only execution is allowed locally or on HPC2. If an asynchronous CPU scheduler job is actually
needed, use the smallest legal resource envelope and one 30-minute probe heartbeat after a verified
job ID; otherwise create no automation.

## Agility budget and terminal event

Use the existing selector implementation directly. Run only targeted selector tests plus one input
projection/dry-run check, then execute the real partial analysis and one fresh-process verifier.
Once those pass, stop; do not add plots, broad suites, generalized frameworks, or extra robustness
variants.

Executor-owned low-risk engineering repairs are unlimited inside this exact sidecar contract. Send
exactly one `GATE_D2P_FINAL_AUDIT` or protocol-valid `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c`, with visible delivery confirmation. The planning thread alone
decides acceptance. D-2 remains active regardless of D-2P completion.
