# GATE_I_HANDOFF

```text
Project/phase: LoopScope Phase 6 — MMLU 0-shot Prefix Trajectory
Planning/audit thread: 019fb3de-2298-75f2-a083-0dca453ea79c
Authorized executor thread: 019fbeff-d331-78d2-b1aa-9741937b93fb
Authorized executor title: execute-LoopScope-MMLU0-Prefix-RBRV2-第6阶段-Gate I
Gate: I

Objective:
- Prove that the frozen Qwen3-4B-Instruct-2507 × standard MMLU 0-shot prefix-last-token
  producer and fresh-process verifier execute correctly on HPC2.
- Close CPU provenance/admission, then run exactly four deterministic outcome-blind validation
  identities as one real GPU smoke on the Slurm debug partition.

Authoritative sources:
- policy: loopscope-tflt/AGENTS.md
  sha256=5f8783cb9826b5f715d61b872027f49e0dc4d2f5e29a155ab9b63ebc275aae76
- control: .planning/loopscope_phase6_control.md
  sha256=5a58472b4db48da4bb936169dcbd32edae28e95bd3c46274cc1a3bdae67c4235
- plan: .planning/loopscope_phase6_mmlu0_prefix_plan.md
  sha256=bdfe9d2085824751f9d2c738020d6f7263583a8012f1bbd15856ea08a947e47a
- frozen card: loopscope-tflt/configs/loopscope/phase6_mmlu0_prefix_card.json
  sha256=a415307c6e33a5c90b2fc2072c1926af75bc6d00e2b36938c11b8dad263ff43a
- trajectory schema: loopscope-tflt/configs/loopscope/phase6_mmlu0_trajectory_schema.json
  sha256=1d6d427f9bc1f34d8af19043661f2307989e72998c461df678feeade75c4da26
- runbook: loopscope-tflt/docs/loopscope_phase6.md is procedure only and grants no live authority.

Starting provenance:
- local repo: /Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
- HPC2 clone: /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
- branch: loopscope
- base HEAD/origin: 0fda7bb191b73343d7555e44dbcd3dd67d24cb36
- required ancestor: 4f59bd93eca4da3cbf458a93508f91c5b23912bc
- expected local dirty state: exactly unstaged M AGENTS.md; staged paths empty
- AGENTS.md is planning-owned protected state: read it, do not stage/commit/restore/overwrite it.

Frozen scientific contract:
- model: Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554;
  36 decoder layers, hidden size 2560, bfloat16
- task: cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe; validation only,
  exactly 1531 identities and 57 subjects; test split is forbidden
- evaluator: lm-eval 0.4.11 standard choice loglikelihood, num_fewshot=0, plain prompt,
  apply_chat_template=false, fewshot_as_multiturn=false, generation=false
- exact continuation surfaces and token IDs: " A"=362, " B"=425, " C"=356, " D"=422
- probe: last non-padding rendered-prefix token; one native zero-loop forward; use_cache=false
- boundaries: raw B0...B36; final-normalize B0...B35 once; B36 is not double-normalized
- per-boundary choice distribution is the normalized four-choice distribution; selector fields are
  H and KL(p_l||p_36)
- diagnostics only: hidden RMS-L2-to-final, cosine-to-final, cosine-distance-to-final, and
  36 raw adjacent angular distances
- Gate I does not execute V2 selection or any loop. Gate J remains locked.

Allowed implementation and operations:
- Read current repository/history and exact authorized files; reuse Gate H, Phase 5 MMLU and
  Phase 6 boundary/final-norm helpers when semantics match.
- Add or narrowly modify only the minimal Gate I runtime/runner/verifier/tests/runbook needed for
  the normal four-identity GPU path. Do not build a general framework.
- Use up to three active bounded subagents; the executor parent owns review/integration/Git/remote.
- Run focused local and remote tests, compile/help/dry-run, commit only authorized repo files,
  push origin/loopscope once ready, verify the remote ref, and fast-forward the clean dedicated
  HPC2 clone to that exact commit.
- On HPC2 CPU, close exact cached model/tokenizer/task/dependency/source hashes and produce a
  fresh gold-free validation identity/rendered-prefix projection. Reading question and choices is
  allowed only for identity and prompt construction; do not request, index, materialize or persist
  target/gold fields. Freeze four deterministic unique identities spanning at least four subjects.
- Submit the real smoke only to Slurm partition exactly debug, wall time <=00:29:00, one technically
  suitable available debug GPU, 8 CPU and 64 GiB. Formal/emergency/priority partitions are forbidden.
- Use a fresh write-once run root. Persist only closed sanitized scalar records/receipts/manifests;
  prompt, input token IDs and source text may appear only as hashes. Do not persist raw logits,
  full-vocabulary distributions, hidden tensors, question/choice text or model generations.
- After producer terminal, run one separate fresh-process verifier over exact membership, hashes,
  schema, finite values, distribution closure, 37 boundaries, 36 adjacent angles and information flags.

Required acceptance evidence:
1. CPU admission and deterministic four-identity manifest close exact model/task/tokenizer/renderer,
   validation 1531/57, unique identity/order and four-or-more-subject membership.
2. Exactly four sanitized trajectory records, no missing/extra/duplicate identity and one native
   no-loop forward per identity; no generation and no loop insertion.
3. Every record closes B0...B36, one FinalNorm application for B0...B35, raw B36 handling,
   four-choice probabilities, H/KL, all hidden diagnostics and 36 adjacent angular distances.
4. Separate fresh-process verifier PASS and exact artifact SHA-256 evidence.
5. No test split, target/gold/label, correctness/accuracy/gain/flip/outcome, selector or Gate J access.

Engineering/retry semantics:
- Executor-owned low-risk engineering repair loops and fresh retries are unlimited while frozen
  science, information barrier, protected state and exact Gate scope remain unchanged and material
  progress continues. Preserve failed roots; use a new write-once root for each retry.
- A failed debug attempt is an attempt-level event, not Gate terminal. Diagnose, repair, run focused
  regression tests and retry in debug. Do not emit BLOCK for a first engineering hypothesis.
- BLOCK only for a required scientific-contract change, information/safety/destructive/external
  authority boundary, or proven no-progress boundary.

Heartbeat:
- Smoke cadence is 10 minutes. Create the sole executor-owned heartbeat only after the real debug
  job is RUNNING stably for about 60 seconds without launcher/startup failure and a wait remains.
  If it completes earlier, create none. On terminal event pause/delete before resuming this exact
  executor; replacement jobs reuse/update the single heartbeat definition.

Forbidden actions:
- Do not stage/commit/restore AGENTS.md or edit planning control/plan/handoff.
- Do not read the MMLU test split or any target/gold/label/correctness/outcome field.
- Do not run validation-1531 formal acquisition, formal selector, panel, loop experiment, Gate J,
  or any formal/non-debug GPU job.
- No cache/package/environment/network mutation unless an exact missing provenance dependency makes
  it essential and it remains an outcome-safe low-risk engineering action; prefer the pinned cache.
- No stash/reset/rebase/merge/force-push, destructive cleanup or mutation of historical run roots.

Agility stop rule:
- Once the four-identity debug smoke and fresh verifier pass with focused tests and exact provenance,
  stop, commit/push any final engineering repair, and send the terminal packet. Do not add optional
  hardening, full-suite testing, extra identities, performance tuning or Gate J preparation.

Terminal requirement:
- Send exactly one GATE_I_FINAL_AUDIT or genuine BLOCK to planning thread
  019fb3de-2298-75f2-a083-0dca453ea79c using send_message_to_thread with visible confirmation.
- Include branch/commit/dirty state, changed files, focused checks, remote host/job/run root,
  decision-critical hashes, observed closure, repair/retry accounting and protected-state proof.
- Request PASS / PASS_WITH_FIXES / BLOCK. Gate J remains locked.
```
