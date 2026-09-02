# GATE_L_HANDOFF

Project/phase: LoopScope Phase 6 — MMLU 5-shot Prefix Trajectory Follow-up  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor thread: `019fc157-4d77-7390-b1c1-2b6d8dfcd7c4`  
Authorized title: `execute-LoopScope-MMLU5-Prefix-RBRV2-第6阶段-Gate L`  
Gate: L — CPU admission and four-identity end-to-end debug smoke

## Objective

Reuse the accepted Gate K 5-shot projection and run the smallest real no-loop GPU smoke proving that
the frozen 5-shot prefix reaches the intended last-valid-token probe path and emits valid 37-boundary
trajectories. Gate L ends at a fresh verifier PASS for exactly four identities. It does not run the
formal validation population or selector.

## Authoritative admission

- policy: `loopscope-tflt/AGENTS.md`, SHA-256
  `2cf835f3ab2eb4c70d29536ba267f0ebe7546af10358c3339bfcd4b828217677`
- control: `.planning/loopscope_phase6_control.md`, SHA-256
  `0a51b2fa24d8f86da921b35c18b0c952d8b3c721e5e84972b2a04f5799fcaf7c`
- plan: `.planning/loopscope_phase6_mmlu5_prefix_plan.md`, SHA-256
  `b7c9714d797d12ce1886e9146b88cdc05b0b4e3278532d72bf13d3121a3c8ff0`
- orchestration skill: SHA-256
  `73261e636b1a6cca1c123718670ac7a930af2f56f8cdfbce507868491c1ca35f`

Read all four completely before mutation. Stop on any Gate/executor/hash disagreement.

## Starting provenance and protected state

- repository: `loopscope-tflt`; branch `loopscope`
- exact local/origin base: `3e0a4dbace1213c4af63472bf0ed18752474a77f`
- expected local dirty state: exactly planning-owned unstaged `M AGENTS.md`; no staged paths
- protected `AGENTS.md` SHA-256:
  `2cf835f3ab2eb4c70d29536ba267f0ebe7546af10358c3339bfcd4b828217677`
- Gate K card SHA-256:
  `ea0c0c55d4b661872136e5204b58e363c646fea9c0f979fc3db88af9a919e265`
- Gate K prompt schema SHA-256:
  `668620930e9d73109d4f4d2880bef72b6fa8e8737f517d4ef940ff35f46d19c7`
- Gate K trajectory schema SHA-256:
  `6d64eafd296c8abb84c72d0d490fa2c90ea19fd878e889af89a07542fd7762ec`
- accepted Gate K CPU root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-k-mmlu5-20260802T083000Z`
- accepted projection SHA-256:
  `5ed42d8b873579aba60cf65d23dbd97d760c01a342e55ffde8c6accaee6c4b9f`
- accepted verifier receipt SHA-256:
  `35b257b9155d59ffc27472c0aeb36389c15bd0ecacf95c399f62ab22c6738d23`

Do not stage, commit, restore, overwrite, or otherwise mutate `AGENTS.md` or `.planning/`.

## Frozen science

- model: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`,
  36 layers, hidden size 2560, bfloat16;
- task: standard lm-eval 0.4.11 MMLU choice-loglikelihood, `num_fewshot=5`, dev
  demonstrations, plain prompt, no chat template, no generation;
- population contract remains validation 1,531 / 57 subjects; Gate L selects exactly four frozen
  smoke identities from the accepted Gate K projection. Prefer the same four identities as Gate I,
  but their prompts must be the fresh Gate K 5-shot prompts;
- continuations are exact single-token ` A`, ` B`, ` C`, ` D`;
- one native zero-loop forward per identity, `use_cache=false`, `output_hidden_states=true`, probe at
  the last non-padding token of the rendered 5-shot prefix;
- exact B0...B36 trajectory: choice entropy, KL-to-final, hidden RMS-L2/cosine/cosine-distance, and
  36 adjacent angular distances; hidden/angular metrics remain diagnostic-only;
- no generation, replay, loop insertion, label/gold/outcome/test read, selector, or full-population
  acquisition.

## Allowed execution

1. Revalidate exact Gate K projection, model/tokenizer/source/package/cache hashes on CPU.
2. Make only the smallest Gate L runtime/test/launcher repairs under
   `src/tflt/loopscope/phase6_mmlu5_*`, `scripts/loopscope/*phase6_mmlu5*`,
   `tests/test_loopscope_phase6_mmlu5*`, and minimally `docs/loopscope_phase6.md` or `src/tflt/cli.py`
   if required. Reuse Gate I/J runtime helpers when science is identical.
3. Run focused local and remote tests, compile/help/dry-run, then one fresh GPU smoke on Slurm
   `debug` only. Use one suitable debug GPU (A40 is acceptable), 8 CPU, 64 GiB, and time limit no
   greater than 29 minutes.
4. Use a fresh write-once root named `phase6-gate-l-mmlu5-*`; preserve failed attempts. Low-risk
   engineering diagnosis, repair, and fresh retry are unlimited while science and information
   barriers remain unchanged. A failed job is an attempt event, not an automatic Gate BLOCK.
5. Commit/push only verified authorized tracked changes and ff-only sync the clean HPC2 clone.
6. Up to three bounded subagents are allowed; parent alone owns Git, remote mutation, Slurm, and the
   terminal packet.

## Monitoring

- Debug/smoke stays on `debug` partition.
- Create the sole smoke heartbeat only after the job has actually remained RUNNING for about 60
  seconds without launcher/startup failure and a wait is still needed. Cadence: 10 minutes.
- If the job completes before that threshold, create no heartbeat. Pause/delete any created
  heartbeat before terminal delivery.

## Must-pass evidence

- exact four identities, four forwards, no duplicates/missing records;
- 37 boundaries and 36 adjacent transitions per record; final-norm/raw-B36 conventions match the
  accepted schema;
- all choice and hidden diagnostic fields finite and schema-valid;
- `loop_insertions=0`, `generation_calls=0`, `selector_executed=false`;
- validation target/gold, test split, correctness/outcome all unread; no raw prompt, token IDs,
  full-vocabulary distribution, logits, or hidden tensors persisted;
- fresh-process verifier PASS and Gate M formal runtime dry-plan importable;
- local/origin/HPC2 commits exact; final local dirty state only protected unstaged `M AGENTS.md`.

Minimum verification should stay focused: Gate K/L unit tests, compile, CLI help/dry-run, the real
four-identity debug smoke, and one fresh verifier. Do not run a broad historical suite absent a
specific regression.

## Forbidden and escalation boundary

No formal A800 job, no validation-1531 acquisition, no V2 selector, no full loop experiment, no
test split or validation gold/outcome, and no Gate M action. Do not mutate prior write-once roots or
0-shot artifacts. No stash/reset/rebase/force-push/destructive cleanup.

Continue autonomously through low-risk implementation and launcher/runtime failures. Escalate only
for a genuine scientific-contract choice, information-barrier/safety/destructive boundary, or a
repeated condition that prevents material progress after the protocol threshold.

## Terminal requirement

Send exactly one `GATE_L_FINAL_AUDIT` or genuine `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c` and request `PASS / PASS_WITH_FIXES / BLOCK`. Gate M remains
locked until planning accepts Gate L and creates a separate executor.
