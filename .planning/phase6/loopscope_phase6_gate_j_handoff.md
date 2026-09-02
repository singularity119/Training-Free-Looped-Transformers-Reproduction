# GATE_J_HANDOFF

```text
Project/phase: LoopScope Phase 6 — MMLU 0-shot Prefix Trajectory
Planning/audit thread: 019fb3de-2298-75f2-a083-0dca453ea79c
Authorized executor thread: 019fbf4b-5d69-7931-9724-16b03e2bc3f9
Authorized executor title: execute-LoopScope-MMLU0-Prefix-RBRV2-第6阶段-Gate J
Gate: J — final validation-1531 trajectory acquisition, V2 selector, and phase terminal packet

Objective:
- Acquire one outcome-blind native no-loop prefix-last-token trajectory for every frozen MMLU
  validation identity, verify the complete 1531/57 population, and run the preregistered
  RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE selector exactly once.
- Produce the official Phase 6 MMLU0 selected half-open window or one legal ABSTAIN, plus a concise
  trajectory/diagnostic summary. This is the phase terminal; no full loop experiment follows.

Admission evidence:
- Gate H PASS at 0fda7bb191b73343d7555e44dbcd3dd67d24cb36.
- Gate I PASS at b2fc70167c7803645d442dad607945fdfacb8d0e.
- Gate I debug smoke job 10118528 completed 0:0 on partition debug in 17 seconds;
  fresh verifier receipt sha256=f09cf99d84f277ca0633fb39ec34e19afa453a6acc50d5230fdf0252f4fc5240.

Authoritative sources:
- research-gate-orchestrator skill
  sha256=73261e636b1a6cca1c123718670ac7a930af2f56f8cdfbce507868491c1ca35f
- policy: loopscope-tflt/AGENTS.md
  sha256=5f8783cb9826b5f715d61b872027f49e0dc4d2f5e29a155ab9b63ebc275aae76
- control: .planning/loopscope_phase6_control.md
  sha256=a34f88714f2596a443d6769259f1af37a78bf68e180d3e8308ea7cae1a2a3d8e
- plan: .planning/loopscope_phase6_mmlu0_prefix_plan.md
  sha256=bdfe9d2085824751f9d2c738020d6f7263583a8012f1bbd15856ea08a947e47a
- frozen card: loopscope-tflt/configs/loopscope/phase6_mmlu0_prefix_card.json
  sha256=a415307c6e33a5c90b2fc2072c1926af75bc6d00e2b36938c11b8dad263ff43a
- trajectory schema: loopscope-tflt/configs/loopscope/phase6_mmlu0_trajectory_schema.json
  sha256=1d6d427f9bc1f34d8af19043661f2307989e72998c461df678feeade75c4da26
- runbook: loopscope-tflt/docs/loopscope_phase6.md is procedure only and grants no live authority.

Starting provenance and protected state:
- local repo: /Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
- HPC2 clone: /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
- branch: loopscope
- base HEAD/origin: b2fc70167c7803645d442dad607945fdfacb8d0e
- required ancestor: 0fda7bb191b73343d7555e44dbcd3dd67d24cb36
- expected local dirty state: exactly unstaged M AGENTS.md; staged paths empty
- AGENTS.md is planning-owned protected state: read it, do not stage/commit/restore/overwrite it.

Frozen scientific contract:
- model: Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554;
  36 decoder layers, hidden size 2560, bfloat16
- task: cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe; validation only;
  exact population 1531 identities / 57 subjects
- evaluator: lm-eval 0.4.11 standard choice loglikelihood; num_fewshot=0; plain prompt;
  apply_chat_template=false; fewshot_as_multiturn=false; generation=false
- exact continuations: " A"=362, " B"=425, " C"=356, " D"=422
- probe: rendered prefix last non-padding token; exactly one native zero-loop forward per identity;
  use_cache=false; no wrapper/loop insertion
- boundaries: raw B0...B36; final-normalize B0...B35 exactly once; B36 not double-normalized
- each boundary stores normalized four-choice probabilities, H and KL(p_l||p_36)
- diagnostic only: hidden RMS-L2-to-final, hidden cosine-to-final, hidden cosine-distance-to-final,
  and 36 raw adjacent angular distances; diagnostics never enter selector decisions
- selector: RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE; central blocks 11...24;
  widths 3/4/5/6; exactly 42 half-open candidates; V1 NewEligible and V2 RateStable;
  S_RATE=sqrt(G_H*G_K); subject-stratified joint bootstrap 2000; seed 20260801;
  unique top-1 and selection frequency >=0.80
- legal terminals: selected unique half-open window, ABSTAIN_NO_RATE_STABLE_ELIGIBLE,
  ABSTAIN_NO_UNIQUE_TOP1, or ABSTAIN_RATE_RANK_UNSTABLE
- associated future loop recipe is metadata only: bfloat16, K=3, block Euler, step=1/3,
  horizon=1, cache=first, decode=full. No loop execution is authorized.

Frozen identity/projection closure from Gate I:
- validation projection file sha256=556cb1f02931da680088672c6b863398aaaebf71022d52b97bbb6cf4ad0f6280
- ordered identity sha256=20e224e90bb4db08ab18d447d017baa564b00b0a4c0080af2c28e7a029becfe2
- validation identity reuse sha256=ced8dd2ac8277a30e4d4e101f8f76c2dd125659abe35af679f60e5c6a9f076eb
- ordered zero-shot prompt projection sha256=2be66cc4966f45135075d492cbb30ae67bd148431682e8588c69cee7b1de0278
- task source manifest sha256=7e03a4bac9704839de842c099e068fb2ebba626a066adfc3499c69eb05336b68
- Gate J must freshly reproduce these semantic digests before formal submission. It may reuse the
  exact Gate I CPU admission/projection only after hash verification; never reuse Gate I smoke
  trajectories as formal scientific rows.

Execution stages:
1. Minimal implementation and verification:
   - Reuse the audited Gate I native producer and Gate H selector. Add only the formal manifest,
     deterministic sharding/attempt receipts, merge, full verifier, official selector freeze,
     and concise reporting path needed for this Gate.
   - Run focused local/remote tests, compile/help/dry-run and diff checks. Commit/push and fast-forward
     the clean dedicated HPC2 clone to one exact immutable producer commit.
2. Mandatory debug preflight before formal:
   - On Slurm partition exactly debug, run a <00:29:00 representative multi-subject cohort through
     the same immutable commit, launcher, environment, cached model/data, CUDA, native producer,
     write path and fresh verifier used by formal execution.
   - Use one technically suitable debug GPU, 8 CPU and 64 GiB. Exercise formal sharding/attempt
     layout and merge closure at small scale. Do not treat its trajectory values as selector input.
   - Any change to launcher/runtime/hook/serialization/producer after PASS requires a fresh debug
     preflight. Formal submission is forbidden until preflight PASS.
3. Formal validation-1531 acquisition:
   - Create a fresh write-once formal root and freeze the exact canonical manifest, shard count,
     membership/order, launcher hashes, producer commit, environment and chosen homogeneous GPU type.
   - Select the ordinary-user legal highest-priority partition/QoS using current scheduler evidence;
     any technically suitable homogeneous GPU type is allowed. No administrator-only priority.
   - Choose and freeze 1...8 deterministic shards/concurrency for time-to-result; resource shape must
     not change identities, prompts, dtype, forward count, metrics or selector semantics.
   - Each identity appears once and only once. Persist only sanitized closed scalar trajectories,
     hashes and receipts; no raw prompt/question/choice text, input token IDs, raw logits,
     full-vocabulary distributions or hidden tensors.
   - A shard attempt failure is not Gate terminal. Preserve it and retry only invalid/missing shards
     under the same immutable producer using fresh attempt paths. If a producer-affecting code change
     is required, rerun debug and restart all formal science in a completely fresh root.
4. Merge, verification and selector:
   - Only after all 1531 identities and 57 subjects have fresh valid shard receipts, merge once into
     canonical Gate I order and run a separate fresh-process full recomputation verifier.
   - Project exactly identity/subject/H/D into the selector; hidden diagnostics and all forbidden
     fields must be absent. Execute the frozen 42-candidate V2 selector once, then run a separate
     fresh-process selector verifier that recomputes decision, ranking, frequency and digests.
   - Freeze one official selector result and one Chinese summary. A legal ABSTAIN is a successful
     scientific terminal and must not trigger threshold/candidate/formula tuning.
5. Human-facing output after immutable result closure:
   - Write a concise Chinese result report under 资产/报告/ and outcome-blind trajectory aggregate
     source data/plots under 资产/figures/phase6_mmlu0_prefix/ without overwriting historical assets.
   - Report the official selected window or ABSTAIN prominently, population/coverage, top ranking
     evidence, and layerwise H/KL plus hidden diagnostic trajectories. Presentation cannot alter or
     rerun the selector and is not a Gate acceptance dependency beyond accurately reflecting it.

Information barrier:
- Never read, index, persist or analyze validation target/gold/label/correctness, MMLU test split,
  accuracy/gain/flip, historical outcomes or any loop outcome. Use the audited safe-column path and
  remove protected columns before record iteration.
- Selector and reports may use only sanitized outcome-blind trajectories and subject identity.
- No generation, CoT, direct-letter fallback, chat template, few-shot prompt, full loop run,
  outcome panel, Gate K or Gate L.

Acceptance evidence:
- Exact immutable commit and clean HPC2 clone; local dirty only protected unstaged AGENTS.md.
- One PASS debug-partition preflight after the final producer-affecting commit.
- Formal manifest and verifier prove 1531 unique identities, 57 subjects, exact Gate I order,
  no missing/extra/duplicate rows, 1531 native forwards, 37 boundaries and 36 transitions each.
- Finite normalized choice distributions; H/KL endpoints; hidden RMS/cosine/distance and adjacent
  angular counts; exact final-norm/raw-B36 closure; zero generation/loop insertion.
- Fresh-process full verifier PASS; official selector freeze and independent selector verifier PASS.
- Official terminal is one selected half-open window or one legal ABSTAIN, with no rule changes.
- Exact artifact hashes, Slurm job/task states/resources, write-once roots, retry accounting,
  heartbeat lifecycle and information-barrier flags are included in GATE_J_FINAL_AUDIT.

Engineering, job control and heartbeat:
- Executor-owned low-risk engineering repair loops and evidence-preserving retries are unlimited
  while frozen science, information barrier, protected state and Gate scope remain unchanged and
  material progress continues. Do not emit BLOCK for a first failed debug/formal attempt.
- The executor may cancel/requeue/replace its own pending or invalid Gate J jobs when needed for
  low-risk recovery or migration to a materially faster legal formal resource, preserving roots and
  never mixing incompatible producer commits. Do not cancel completed valid shards unnecessarily.
- Debug heartbeat: 10 minutes only after a real debug job is RUNNING stably about 60 seconds and a
  wait remains; if it finishes earlier, none.
- Formal full heartbeat: create exactly one executor-owned 60-minute heartbeat immediately after
  verified formal job ID(s), including while PENDING; monitor the whole active job set/root. Update
  it in place for replacements. On any terminal attempt, pause/delete then resume this exact
  executor for verification/repair/continuation; planning never polls.

Agility budget and stop rule:
- Smallest increment: formal runner/manifest/sharding/merge/verifier/selector freeze/report adapter
  over existing Gate I/H helpers; no general framework or historical refactor.
- Minimum checks: focused affected tests, one final-commit debug end-to-end PASS, full membership
  and fresh verifier, then exact selector recomputation verifier.
- Once the immutable formal verifier and selector verifier pass, stop engineering immediately,
  write the concise report, commit/push any authorized final code/report-reference change, and emit
  one terminal packet. Do not add robustness sweeps, new metrics, alternate selectors or loop tests.

Forbidden actions:
- Do not stage/commit/restore AGENTS.md or edit planning control/plan/handoff.
- Do not mutate/delete/overwrite Gate H/I or old Phase 6 roots and artifacts.
- No test split, target/gold/outcome access; no selector tuning after seeing results.
- No K/alpha/cache/decode/model/task/prompt/fewshot/chat/GPU-ablation experiment.
- No full loop outcome, panel, Gate K/L, stash/reset/rebase/merge/force-push or destructive cleanup.

Terminal requirement:
- Send exactly one GATE_J_FINAL_AUDIT or genuine BLOCK to planning task
  019fb3de-2298-75f2-a083-0dca453ea79c with visible delivery confirmation.
- Include final selected half-open window or exact legal ABSTAIN; branch/commit/dirty; implementation
  files/tests; debug and formal jobs/resources/roots; full manifest/merge/verifier/selector hashes;
  population/subject/forward/boundary closures; ranking/frequency summary; report/figure paths;
  repair/retry/heartbeat lifecycle; information-barrier and protected-state proof.
- Request PASS / PASS_WITH_FIXES / BLOCK. The executor does not declare Phase 6 complete; planning
  performs the concise consolidated phase audit after this terminal packet.
```
