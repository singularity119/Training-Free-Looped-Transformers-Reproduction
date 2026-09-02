# GATE_M_HANDOFF

Project/phase: LoopScope Phase 6 — MMLU 5-shot Prefix Trajectory Follow-up  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor thread: `019fc180-e590-74a2-9323-f626eaf139bf`  
Authorized title: `execute-LoopScope-MMLU5-Prefix-RBRV2-第6阶段-Gate M`  
Gate: M — final validation-1531 A800 trajectory, frozen V2 selector, follow-up terminal

## Objective and terminal

Acquire the complete outcome-blind 5-shot prefix trajectory for the frozen MMLU validation
population, independently verify it, then execute the frozen V2 selector exactly once. The Gate and
follow-up terminate with either one valid selected half-open window or the selector's legal ABSTAIN.
Do not run a full loop experiment or read evaluation outcomes.

## Authoritative admission

- policy: `loopscope-tflt/AGENTS.md`, SHA-256
  `2cf835f3ab2eb4c70d29536ba267f0ebe7546af10358c3339bfcd4b828217677`
- control: `.planning/loopscope_phase6_control.md`, SHA-256
  `cc79f087c391c120c4c13fa9f4371a70fd03386006cc398bba28a73bcc1e23fb`
- plan: `.planning/loopscope_phase6_mmlu5_prefix_plan.md`, SHA-256
  `b7c9714d797d12ce1886e9146b88cdc05b0b4e3278532d72bf13d3121a3c8ff0`
- orchestration skill: SHA-256
  `73261e636b1a6cca1c123718670ac7a930af2f56f8cdfbce507868491c1ca35f`

Read all four completely before mutation. Stop on any Gate/executor/hash disagreement.

## Starting provenance and accepted inputs

- repository `loopscope-tflt`, branch `loopscope`, exact local/origin base
  `93f264440f4cc70bf44876f42d29c7b3ba2d8b86`;
- expected local dirty state: only protected planning-owned unstaged `M AGENTS.md`; no staged paths;
- protected `AGENTS.md` SHA-256:
  `2cf835f3ab2eb4c70d29536ba267f0ebe7546af10358c3339bfcd4b828217677`;
- Gate K card/prompt-schema/trajectory-schema SHA-256:
  `ea0c0c55d4b661872136e5204b58e363c646fea9c0f979fc3db88af9a919e265`,
  `668620930e9d73109d4f4d2880bef72b6fa8e8737f517d4ef940ff35f46d19c7`,
  `6d64eafd296c8abb84c72d0d490fa2c90ea19fd878e889af89a07542fd7762ec`;
- accepted Gate K projection:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-k-mmlu5-20260802T083000Z/cpu/validation_5shot_projection.jsonl`,
  SHA-256 `5ed42d8b873579aba60cf65d23dbd97d760c01a342e55ffde8c6accaee6c4b9f`;
- accepted Gate L root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-l-mmlu5-20260802T160000Z`;
- Gate L debug Job `10120369`, `debug`, A40, COMPLETED `0:0`, 21 seconds; verifier receipt
  SHA-256 `6fb5cbf6e62a53d03f61f12771a87062bb832e50d4c0d53360bff9dcc3357bfa`.

Do not stage, commit, restore, overwrite, or mutate `AGENTS.md`, `.planning/`, or any accepted
write-once root.

## Frozen scientific contract

- model/tokenizer: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`,
  36 layers, hidden size 2560, bfloat16;
- dataset: `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`;
- evaluator/rendering: lm-eval 0.4.11 standard MMLU choice-loglikelihood,
  `num_fewshot=5`, `fewshot_split=dev`, plain prompt, no chat template, no generation;
- population: exact Gate K validation 1,531 identities / 57 subjects / canonical order; every target
  has five dev demonstrations; validation gold/target and test split remain forbidden;
- exact continuations: single-token ` A`, ` B`, ` C`, ` D`;
- probe: last non-padding token of the rendered 5-shot prefix; one native no-loop forward per
  identity, `use_cache=false`, `output_hidden_states=true`;
- boundaries: B0...B36 with accepted final-norm/raw-B36 convention;
- selector metrics: four-choice entropy H and KL-to-final D;
- diagnostic-only metrics: hidden RMS-L2, hidden cosine, hidden cosine-distance, and 36 adjacent
  angular distances; they never enter candidate admission or ranking;
- selector: exact `RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE`, central blocks 11...24,
  widths 3/4/5/6, 42 candidates, subject-stratified joint bootstrap 2,000, seed 20260801,
  frequency threshold 0.80, projection exactly identity/category/H/D;
- 0-shot trajectories or selector rows must not enter this 5-shot acquisition, bootstrap, ranking,
  or terminal decision.

## Ordered execution

1. Implement only the minimal formal freeze/shard/producer/merge/verifier/selector path by reusing
   accepted Gate J and Gate K/L code where science is identical. Run focused local and HPC2 tests,
   compile/help/dry-run.
2. Freeze one fresh formal root under
   `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-m-mmlu5-*`
   with the full 1,531/57 manifest, exact Gate K order, eight deterministic shards, source/commit/
   card/schema/projection hashes, and write-once attempts.
3. Before formal submission, run a passing end-to-end preflight on Slurm `debug`, under 29 minutes,
   using the exact immutable final commit, formal launcher/environment/model/runtime producer path,
   representative 5-shot members, artifact writing, and verifier path. Gate L Job 10120369 is valid
   prior smoke evidence but may replace this final-commit preflight only if commit, launcher,
   environment, producer, hooks, and serialization are demonstrably unchanged.
4. Submit the formal workload only to the highest legally available normal-user A800
   partition/QoS. The frozen default is an eight-task array `0-7%8`, one A800 per task, 8 CPU,
   64 GiB, and up to 6 hours. A different A800 partition/QoS is allowed only to obtain the highest
   legal priority; A40 or another GPU type is not authorized for the formal producer.
5. Immediately after a verified formal job ID exists, including while PENDING, create exactly one
   executor-owned `full` heartbeat at 60-minute cadence. The same heartbeat monitors queue,
   RUNNING, startup, and terminal states and must resume this exact executor on terminal evidence.
6. Require all eight shards and fresh per-shard verification, merge exactly once, then run a separate
   fresh-process full verifier. Only after that PASS, execute the frozen V2 selector exactly once and
   independently verify the selector freeze.
7. Write a concise Chinese terminal report under `资产/报告/` labeling the 5-shot result and its
   selected window or legal ABSTAIN. Do not create a loop-outcome claim.

## Engineering and retry authority

- Allowed tracked paths: `configs/loopscope/phase6_mmlu5_*`,
  `src/tflt/loopscope/phase6_mmlu5_*`, `scripts/loopscope/*phase6_mmlu5*`,
  `tests/test_loopscope_phase6_mmlu5*`, and minimal `docs/loopscope_phase6.md`/`src/tflt/cli.py`.
- Remote writes are confined to the clean dedicated clone, fresh Gate M roots, and scheduler jobs.
- Executor-owned low-risk repair and fresh retry loops are unlimited. Preserve failed attempts and
  never overwrite or mix them. Job failure is attempt-level, not automatic Gate BLOCK.
- The executor may cancel/requeue/retry its own Gate M jobs when required for a diagnosed low-risk
  launcher, environment, resource, serialization, shard, merge, or verifier defect. Any producer-
  affecting change requires a new final-commit debug preflight before formal resubmission.
- Use up to three concurrent bounded subagents; parent alone owns Git, remote mutation, Slurm,
  integration, selector execution, heartbeat lifecycle, and terminal delivery.
- Once a valid selector decision has been produced, do not tune, rerun, or replace it. A failed
  pre-decision selector wrapper may be repaired only if no valid freeze/decision exists and all
  inputs, candidates, formulas, seed, bootstrap indices, and thresholds remain unchanged.

## Must-pass and information barrier

- formal membership exactly 1,531 identities / 57 subjects, no missing/extra/duplicate/cross-run
  records, all eight shards PASS and provenance/hash closure exact;
- 1,531 native forwards, each with 37 boundaries and 36 adjacent transitions, no generation or
  loop insertion, all persisted trajectory scalars finite and schema-valid;
- fresh full verifier PASS before selector; selector returns one legal `SELECTED_WINDOW` or legal
  `ABSTAIN`, and a fresh selector verifier reproduces decision/ranking/digests;
- persist only the closed scalar trajectory/provenance needed for verification and selection. Do
  not persist or inspect raw prompts, token IDs, validation answers/gold, correctness, accuracy,
  gain/flip/outcomes, raw logits/full-vocabulary distributions, or hidden tensors;
- no test split, no full loop experiment, and no Gate beyond M;
- final local/origin/HPC2 commit exact; local dirty state only protected unstaged `M AGENTS.md`;
  heartbeat paused/deleted before terminal delivery.

Escalate only for a genuine frozen-science choice, information unsealing, unsafe/destructive or
external authority, or repeated inability to make material progress after protocol threshold.

## Agility budget and terminal packet

Reuse the working Gate J sharded formal path and the accepted Gate K/L 5-shot renderer/runtime;
avoid new frameworks, broad historical suites, exhaustive tamper cases, or panel/outcome machinery.
The decisive path is final-commit debug preflight -> A800 full acquisition -> fresh verifier -> one
V2 selector -> verifier/report. Stop when these pass.

Send exactly one `GATE_M_FINAL_AUDIT` or genuine `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c`, request `PASS / PASS_WITH_FIXES / BLOCK`, and include the
formal job/root, debug preflight, commit, artifact hashes, selector terminal, heartbeat lifecycle,
information-barrier flags, and actions not taken. This is the Phase 6 MMLU 5-shot terminal Gate.
