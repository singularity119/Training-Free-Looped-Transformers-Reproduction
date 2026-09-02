# GATE_H_AUDIT_RETURNED_REPAIR_1

```text
Project/phase: LoopScope Phase 6 — MMLU 0-shot Prefix Trajectory
Planning/audit thread: 019fb3de-2298-75f2-a083-0dca453ea79c
Authorized executor: 019fbecf-be9a-71e3-94f0-a82356b79d29
Gate: H
Decision: PASS_WITH_FIXES

Authoritative state:
- policy sha256: 3e0e4425eb42ad19e2ba75f5f287ce3582d8ceefbbf2f3fe4efb6bbc7b8a14eb
- live control sha256: cccc00bc3466435817b089b7920e7309fd499f1b017a5aa0bf48725dfd6d63ab
- original handoff sha256: f5c7d4772183c60dc16d19ed0bfdce396b0e544ee19d48ed69dd1e5d34722eea
- repair base HEAD/origin: 4fd4c15baae2316ae4ac189a3dfc0be564c07883
- expected dirty state: exactly unstaged M AGENTS.md; staged paths empty

Material findings:
1. configs/loopscope/phase6_mmlu0_prefix_card.json freezes config_sha256 as
   a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3,
   which is tokenizer_config.json. Planning independently hash-checked the exact pinned snapshot:
   config.json=5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba;
   tokenizer_config.json=a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3;
   tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4.
2. The current user-frozen branch ends at Gate J and forbids reading/persisting the test split or
   running full loop outcomes. The card/schema copied test_split, 14042 count, Phase 5 test identity
   manifests/order/disjointness fields. They are not needed for the validation-1531 selector and
   conflict with the current minimal information boundary.

Exact repair:
- Correct only the model config hash to the exact config.json hash above everywhere it is frozen,
  validated, tested or documented.
- Remove test split/count and every copied test identity/order/disjointness field from the new
  phase6_mmlu0 card, validator/schema, tests, CLI/self-test output and runbook statements.
- Retain validation-1531 / 57-subject identity-only provenance and fresh-required zero-shot prompt
  projection. Do not read any dataset record, test split, gold, label, correctness or outcome.
- Keep model/task/evaluator/probe/metric/V2 candidates/bootstrap/thresholds and Gate terminal
  unchanged. Do not add new features or hardening.

Validation and delivery:
- Run the same focused unittest, compile, help/dry-run/verify/self-test and diff checks affected by
  the repair; no full suite and no model/data/GPU/Slurm action.
- Commit only the narrow repair files with one purpose, push origin/loopscope and verify exact ref.
- Return one concise GATE_H_FINAL_AUDIT_REPAIR_1 to planning with the new commit, changed paths,
  focused checks, new card hash and protected-state confirmation.

Repair accounting:
- This is audit-returned repair cycle 1. Executor-owned low-risk iterations within this exact
  repair remain unlimited. Gate I/J stay locked.
```

