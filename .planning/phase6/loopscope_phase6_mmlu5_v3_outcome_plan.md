# LoopScope Phase 6 Gate O — MMLU 5-shot V3 fixed-panel outcome plan

## Objective

在不再调整 selector、窗口或 loop 配方的前提下，对同一 MMLU test-14042
总体运行一个固定五 cell panel，报告 full accuracy 与相对 no-loop baseline 的
paired gain。Gate O 完成后终止本次 Phase 6 follow-up，不再自动扩展窗口或运行其他
loop 实验。

## Frozen cells

TFLT 窗口均采用 CLI inclusive 记法：

1. `no_loop`；
2. `14:16`（V3 selected，half-open `14:17`）；
3. `13:16`（half-open `13:17`）；
4. `15:18`（half-open `15:19`，既有实测参考窗）；
5. `12:16`（half-open `12:17`）。

不得新增、删除、替换、按结果重排 cell，也不得加入 `decode=bypass` 对照。

## Frozen model, evaluator, and TFLT recipe

- model: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`
- dataset: `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`
- evaluator: lm-eval `0.4.11`, standard MMLU choice-loglikelihood, `num_fewshot=5`,
  dev demonstrations, plain prompt, no chat template, no generation/CoT
- population: test 14,042 identities / 57 subjects, canonical manifest and order closed
- dtype: bfloat16
- loop: `K=3 | mode=block | strategy=euler | alpha=1 | beta=0 |
  cache=first | decode=full`
- batch size: 16 for all cells. A lower common batch size is allowed only after a real
  debug OOM, must be applied to all five cells, and requires a new fresh debug preflight;
  valid artifacts from different batch-size contracts may not be mixed.

## Decode semantics

Standard MMLU does not generate an answer string. It scores the exact one-token
continuations ` A`, ` B`, ` C`, ` D` with choice log-likelihood. Although the scientific
decision uses the first continuation token, the model request is a full prompt-plus-choice
sequence forward, not a cached token-by-token decode step. In the current TFLT wrapper,
`decode=bypass` skips looping only when the forward is incremental (`sequence_length=1`
with cache history/cache position); therefore this MMLU path is expected to execute the
loop under both `full` and `bypass`. Gate O nevertheless freezes the user-requested
`decode=full`; equivalence is a code-path statement, not an extra outcome cell.

## Execution

1. Local/HPC CPU contract and focused tests; preserve all pre-existing dirty paths.
2. On Slurm `debug`/A40, run a same-commit/launcher/env/model/data five-cell limited smoke.
   Each loop cell must prove loop effect, `operator_body_calls=3` per audited prompt,
   restore allclose, finite outputs, and no bypass; baseline must prove zero loop activity.
3. Only after debug PASS, create a fresh formal manifest and submit one five-cell A800
   array on the highest-priority legal A800 partition/QoS (`emergency_gpu` observed).
   Each task uses 1 A800, 8 CPUs, 64 GiB, and up to 6 hours.
4. Immediately after the formal job ID is verified—even while PENDING—create exactly one
   60-minute heartbeat. Each wake is one bounded read-only scheduler/artifact check.
5. Do not inspect partial accuracy. After all five cells pass scheduler and identity closure,
   perform exactly one outcome parse/analysis and one independent fresh-process verifier.

## Terminal evidence

- exact 14,042-identity join for every cell; missing/extra/duplicate = 0;
- per cell `correct_count`, `accuracy`, and accuracy delta versus no-loop in percentage points;
- paired sample-level comparison against baseline, with method/seed frozen before outcome read;
- scheduler, command, model/data/evaluator/TFLT, manifest, result, and verifier hashes;
- concise Chinese report under `资产/报告/phase6/`;
- one terminal `GATE_O_FINAL_AUDIT` or a material `BLOCK` packet to the planning task.

Engineering failures are attempt-level events: the executor must diagnose and continue with
fresh write-once attempts under the unlimited low-risk repair rule. Only scientific-contract,
information-barrier, destructive/security/external-authority, or genuine no-progress boundaries
are Gate-level BLOCKs.
