# GATE_H_HANDOFF

```text
Project/phase: LoopScope Phase 6 — MMLU 0-shot Prefix Trajectory continuation
Planning/audit thread: 019fb3de-2298-75f2-a083-0dca453ea79c
Authorized executor thread: 019fbecf-be9a-71e3-94f0-a82356b79d29
Authorized executor title: execute-LoopScope-MMLU0-Prefix-RBRV2-第6阶段-Gate H
Gate: H

Objective:
- Freeze and implement the smallest local contract/provenance layer needed for
  Qwen3-4B-Instruct-2507 × standard lm-eval MMLU 0-shot prefix-last-token trajectories.
- Reuse audited Phase 5 MMLU choice-space and Phase 6 36-layer/V2 components where their
  semantics match; make every changed semantic explicit and testable.

Non-objectives:
- No real model forward, generation, CUDA, GPU, Slurm, formal trajectory acquisition,
  formal selector freeze, loop outcome, test split, gold/correctness/outcome read, or Gate I+ work.
- No general framework, historical Phase 5/6 rewrite, or re-audit of terminated D-2 artifacts.

Authoritative sources:
- policy: loopscope-tflt/AGENTS.md
  sha256=3e0e4425eb42ad19e2ba75f5f287ce3582d8ceefbbf2f3fe4efb6bbc7b8a14eb
- control: .planning/loopscope_phase6_control.md
  sha256=e1a42cdc62a7cadd5756679c1e969ce6a5485d026ace13dbc661e24fb99ae804
- plan: .planning/loopscope_phase6_mmlu0_prefix_plan.md
  sha256=2e16cb0c7daf7706b546dff8b28b127e0bf8e94f3403b2d34cda096002b7ec13
- runbook: loopscope-tflt/docs/loopscope_phase6.md; procedure only, executor may update
  the current-branch section without recording mutable executor/Gate state.

Starting provenance:
- local repo: /Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
- HPC2 clone: /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
- branch: loopscope
- base HEAD/origin: f759525a7a4aebbec0a101a2ebfd1018b248ae05
- required ancestor: 4f59bd93eca4da3cbf458a93508f91c5b23912bc
- expected local dirty state: exactly unstaged M AGENTS.md; staged paths empty
- AGENTS.md is planning-owned protected state: read it, do not stage/commit/restore it.

Frozen scientific contract:
- model: Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554
- tokenizer revision: same exact revision; 36 decoder layers; hidden size 2560; bfloat16
- task: cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe
- lm-eval/dependencies: 0.4.11 / transformers 4.51.3 / tokenizers 0.21.4 /
  datasets 5.0.0 / torch 2.3.1+cu121 / accelerate 1.14.0
- evaluator: standard choice loglikelihood; num_fewshot=0; plain prompt;
  apply_chat_template=false; fewshot_as_multiturn=false; no generation or direct-letter fallback
- prompt terminal: exact standard task renderer terminal; continuation surfaces exact
  " A", " B", " C", " D"; Gate H must freeze tokenizer IDs and require unique one-token surfaces
- trajectory/selector source: validation 1531 / 57 subjects; test split and loop outcome are out of scope
- probe: last non-padding rendered-prefix token; one native no-loop forward; use_cache=false
- boundaries: raw B0...B36; final-normalize B0...B35 once; never double-normalize B36
- selector fields: choice entropy H and KL(p_l||p_36)
- diagnostics only: hidden RMS-L2-to-final, cosine-to-final, cosine-distance-to-final,
  raw adjacent angular distance
- selector: exact RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE; central blocks 11...24;
  widths 3/4/5/6; 42 candidates; 2000 subject-stratified joint bootstraps;
  selector seed 20260801; selection frequency >=0.80; legal ABSTAIN preserved
- loop identity attached to the suggested window only: bfloat16, K=3, block Euler, step=1/3,
  horizon=1, cache=first, decode=full; no loop run is authorized in this Phase 6 branch

Allowed actions:
- Read repository/history/current policy-control-plan and audited Phase 3/5/6 sources.
- Add the minimal new MMLU0 Phase 6 card/schema/runtime/selector adapter/verifier/CLI/tests,
  plus update docs/loopscope_phase6.md. Prefer new phase6_mmlu0_* modules and reuse pure helpers.
- Narrowly modify an existing Phase 5/6 helper only when a focused test proves the current
  experiment normal path requires it; preserve all historical cards/artifacts.
- Use up to three active bounded subagents under the orchestrator protocol; parent owns integration.
- Bounded read-only SSH to hpc2-hkustgz with ClearAllForwardings=yes for exact cached model,
  tokenizer, lm-eval MMLU source/template, dependency and outcome-safe dataset identity/renderer
  provenance. Question/choices may be read only as needed to build gold-free identities/prompts;
  target/gold fields and the test split must not be accessed or persisted.
- Run local focused tests/compile/help/dry-run/self-test. Commit only authorized repo files,
  push origin/loopscope, and independently verify the remote ref.

Forbidden actions:
- Stage/commit/restore AGENTS.md or modify planning control/handoff/plan.
- Read target/gold/label/correctness/accuracy/gain/flip, the test split, or historical/new outcomes.
- Load model weights or run model.forward/generate; no CUDA/GPU/Slurm/scheduler query/submission.
- Run a real selector, freeze a real panel, enter Gate I+, modify/delete/move old run roots,
  stash/reset/rebase/merge/force-push, or touch protected wrapper/strategy/cache/config modules.

Required implementation and verification:
1. A machine-readable card closes exact model/task/evaluator/split/population/renderer/token
   surfaces/boundaries/metrics/V2 selector/later-loop contract and information barrier.
2. Gold-free validation/test identity and renderer provenance either reuse exact audited Phase 5
   evidence only when semantics are identical, or are freshly hash-closed for num_fewshot=0.
3. Closed validators reject fewshot !=0, chat template, generation, wrong model revision,
   multi-token/duplicate choice surfaces, wrong probe position/count, hidden diagnostics entering
   selector, non-42 candidate domain, and gold/outcome fields.
4. Pure synthetic tests recompute choice entropy/KL, RMS-L2/cosine/distance, adjacent angle,
   endpoint closures and exact half-open boundary mapping.
5. CLI help, import-only dry-run and independent self-test state explicitly that model/data
   forward, selector, test split and outcomes were not executed/read.
6. Targeted Phase 6 MMLU0 tests pass; compile and git diff checks pass. Do not run the full
   historical suite unless the executor finds a concrete broad regression risk.

Agility budget:
- Smallest increment: one card + narrow schema/helpers/verifier/CLI/tests and concise runbook update.
- Earliest evidence: synthetic self-test plus outcome-safe exact provenance/tokenizer closure.
- Minimum decisive checks: focused tests, compile, help, dry-run, verifier self-test, diff check.
- Stop rule: once these pass, commit/push and report; do not add portability, generic framework,
  exhaustive tamper matrices, duplicate receipts, or unrelated documentation.

Audit materiality:
- Blocking: wrong model/task/prompt/fewshot/chat/evaluator, wrong choice-token surface, test/gold
  access, incorrect boundary/final-norm/metric formula, selector diagnostic leakage, inability
  to execute the planned normal path, protected-state drift, or untrustworthy revision closure.
- Nonblocking: cosmetic naming, optional schema elegance, unused-path hardening, full-suite issues
  not reached by this producer path.
- Planning acceptance will sample only the commit/file scope, one contract self-test result and
  one decisive provenance/token closure; broader phase audit is deferred to Gate L.

Repair budget:
- Executor-owned low-risk engineering repair loops: unlimited while science, permissions,
  information barrier, protected state and fresh-evidence semantics stay frozen and progress is made.
- Fresh engineering retries inside this Gate are allowed; preserve evidence when a rerun could
  obscure provenance.
- Escalate only scientific/evaluator/split/metric/candidate/threshold changes, outcome access,
  protected core-module changes, destructive/external authority, or no-progress boundary.

Terminal requirement:
- Send exactly one GATE_H_FINAL_AUDIT or genuine BLOCK to planning thread
  019fb3de-2298-75f2-a083-0dca453ea79c using send_message_to_thread and retain confirmation.
- Include branch/commit/dirty, files, focused commands/tests, HPC2 provenance reads if used,
  decision-critical hashes, observed closure, deviations, repair loops and protected state.
- Request PASS / PASS_WITH_FIXES / BLOCK. Gate I/J remain locked; no Gate K/L exists.

Long-job/automation:
- None expected. Gate H has no GPU/Slurm authority and must not create a heartbeat.
```
