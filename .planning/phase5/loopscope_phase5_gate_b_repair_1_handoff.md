# GATE_B_REPAIR_1_HANDOFF

~~~text
Project/phase=LoopScope Phase 5
Hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f864e-9b57-73e1-b56c-e7e40e7ad28d
Executor title=execute-LoopScope-Trajectory-Selection-第5阶段-Gate B
Gate=B
Planning decision=PASS_WITH_FIXES
Repair cycle=1 of 1 planning-audit-returned cycle
State=AUTHORIZED_AFTER_VISIBLE_DELIVERY
Terminal event=GATE_B_FINAL_AUDIT or BLOCK
~~~

## 1. Audit finding and repair boundary

Formal array `10030284` is terminal. Shards 0000 and 0002 stopped after partial native forwards
because the producer required a sliced four-row bfloat16 GEMM to be within fixed tolerance of the
model's native full-vocabulary B36 LM-head output. Shards 0001 and 0003 completed. The planning
audit verified that Qwen3's native logits already come from the frozen full LM head applied to the
same final-normalized hidden state. A four-row GEMM may select a different bfloat16 kernel and is
not the authoritative final-logit computation.

This is a narrow engineering false-negative, not a change to model, data, renderer, probability
space, trajectories, selector, panel, or outcome boundary. Gate B may resume only under this
supplement. All unchanged clauses of `.planning/loopscope_phase5_gate_b_handoff.md` remain binding.

## 2. Immutable invalid attempt

The following root is permanently preserved as a failed, non-canonical engineering attempt:

~~~text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-20260721T201330Z
formal_job=10030284
formal_membership_sha256=e874e1a896eadc0819858bc4cc465ced6008b6945adba67a016c4dffa23a3f63
persisted_records=1126
missing_records=405
canonical_selector_input=false
merge_or_salvage=false
~~~

Do not overwrite, delete, append to, merge, or select from this root. Its 1126 records are audit
evidence only. The successful shards may not be combined with repaired output. Record the old root,
job, implementation commit, failure receipts, and invalidation reason in the new repair admission
receipt.

## 3. Authorized implementation repair

Starting from clean pushed commit `9431688b8e1f704837b22e1370978dea59d528c0`, make one narrow
repair commit:

1. For B0..B35, retain the exact frozen-final-norm plus tied LM-head four-choice-row projection.
2. For B36, use `outputs.logits[0, 0, choice_ids]` as the authoritative recorded choice logits.
3. Independently apply the same complete frozen LM head to the native final-normalized B36 hidden
   state, compare the resulting full-vocabulary logits with `outputs.logits`, and fail closed if that
   same-shape closure does not pass. Full-vocabulary logits are temporary and must never be persisted.
4. The sliced four-row B36 projection may be retained only as a non-gating diagnostic; it must not
   determine acceptance, probabilities, entropy, KL, selector evidence, or stored B36 choice logits.
5. Add a targeted test proving that B36 probabilities are sourced from native logits and that a
   simulated sliced-bfloat16 mismatch does not abort when the full-head/native closure is valid.

Default authorized edit scope:

~~~text
src/tflt/loopscope/phase5_acquisition.py
tests/test_loopscope_phase5_gate_b.py
~~~

A minimal launcher or Gate B runbook update is allowed only if required to bind the superseding root.
Do not modify Gate A card/schema/registry, selector mathematics, protected implementation paths, or
Phase 1-4 artifacts. Run the original targeted Gate A/Gate B checks, the new regression, py_compile,
`git diff --check`, and protected-path diff. Create one purpose-specific commit and ordinary push;
remote sync must be clean fast-forward only.

## 4. Repair smoke and canonical reacquisition

Use a fresh write-once repair root; never reuse the invalid root. A recommended name is:

~~~text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-<UTC>Z
~~~

Before formal launch, run one exact two-identity GPU repair smoke using the two identities that
triggered the false-negative:

~~~text
mmlu_marketing:validation:24
mmlu_human_aging:validation:18
~~~

They are chosen solely as engineering regression triggers, not from trajectory values. The smoke
must prove native B36 source, same-shape full-head closure, five-trajectory final closure, no loop,
no forbidden persistence, and exactly two forwards. Smoke artifacts are excluded from selector.

If smoke passes, perform one fresh canonical acquisition of all 1531 validation identities under
the repaired commit. This explicit superseding reacquisition is authorized even though the invalid
root contains earlier physical forwards: every canonical identity must have exactly one successful
formal forward in the new root, and only the new root may feed merge/selector. Preserve the same
frozen global identity manifest, prompt hashes, dtype, renderer, fields, sharding semantics, and
selector contract. Scheduler partition/concurrency may be chosen operationally as before.

Do not salvage suffixes or mix producer commits. If any new formal shard writes partial records and
then fails, stop and send `BLOCK`; this supplement does not authorize another audit-returned repair
cycle. A zero-forward/zero-record scheduler failure retains the original narrow retry rule.

## 5. Completion and terminal evidence

After exactly 1531 repaired formal records close, run the original B4 merge, independent verifier,
outcome-blind selector, eligibility/ABSTAIN, and High3/Low3 panel freeze. Gate C remains locked and
no MMLU test/outcome access is authorized.

The next terminal packet must include all fields required by the original Gate B handoff plus:

- repair commit and exact two-identity smoke job/result;
- old invalid root/job/record counts and proof it was not mutated or consumed;
- new canonical root/job IDs and exact 1531 identity/forward closure;
- B36 source contract and full-head/native closure extrema;
- confirmation that sliced B36 mismatch is non-gating and full-vocabulary tensors were not persisted;
- repair-cycle accounting=`2 executor-owned repairs + 1 planning-audit-returned repair`;
- no test/outcome access and no Gate C work.

Send exactly one `GATE_B_FINAL_AUDIT` or `BLOCK` to the planning thread with visible delivery
confirmation, then stop.
