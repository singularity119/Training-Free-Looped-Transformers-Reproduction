# GATE_D2_OPERATIONAL_SUPPLEMENT_1

Project/phase: LoopScope Phase 6  
Gate: D-2  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb8f3-8cb7-7c93-a8a0-bae811735601`  
Decision: `PASS_WITH_FIXES`; resume the same Gate D-2 executor  

## Admission

Re-read and require exact hashes before mutation:

- `loopscope-tflt/AGENTS.md`:
  `e4373fd0513a312026004d807869a59b8857b032e06abed5b81a93ad96a5c628`;
- `.planning/loopscope_phase6_control.md`:
  `2ebadfc1fa2f8101eb6e931f151693c65ae5f8566cd68bb49172e386e7560399`;
- original Gate D-2 handoff:
  `83e4a144838137911070b232f5f76f0058a6c1f32a68881c8cdc6b3bdbe789b4`.

Local HEAD and `origin/loopscope` must be
`b4eda56b9acf77a62140aa0d235de6c04ca446f2`; local dirty state remains exactly protected
unstaged `M AGENTS.md`; HPC2 clone must be clean at the same commit. Jobs `10107105` and
`10107145` and their write-once roots remain immutable failed evidence. The paused debug heartbeat
must not be duplicated or reactivated for those terminal jobs.

## Planning disposition

The debug-only assertion that a named historical identity must deterministically reproduce
`ANCHOR_NOT_EXPRESSED` is removed from Gate D-2 acceptance. Scientific Amendment 2 permits any
fresh run to observe zero or more `ANCHOR_NOT_EXPRESSED` identities; it freezes how the state is
handled and the population/category coverage floors, not a permanent per-identity format class.

This supplement does not change regex, ordinal, anchor semantics, eligibility states, overall
coverage floor `0.995`, per-category floor `0.98`, model/data/generation/TFLT, metrics, selector,
panel, information barrier, or formal population. It changes only the debug characterization
requirement.

## Authorized repair and continuation

1. Remove only the debug verifier requirement that the actual cohort contain a predetermined or
   nonzero count of `ANCHOR_NOT_EXPRESSED`. The verifier must validate the observed partition,
   including the legal all-eligible case.
2. Add focused regression tests proving both:
   - an all-eligible debug cohort closes and verifies;
   - a synthetic mixed eligible/not-expressed cohort closes and verifies with no replay or
     trajectory for not-expressed rows.
3. Retain all existing zero-match masking, no-fallback, canonical identity, sealed membership,
   alignment fail-closed, and coverage-floor tests. Do not search model generations for a new
   real zero-match identity and do not target/retry nodes to manufacture one.
4. Commit/push only the minimal affected Gate D runner/verifier/tests/runbook paths; fast-forward
   the clean HPC2 clone; rerun focused local and remote tests.
5. Submit one fresh `<30min` debug A40 preflight using the existing frozen cohort/runtime and a new
   write-once root. It may observe zero or more not-expressed identities. It must complete the real
   producer, eligibility receipt, namespace separation, trajectory writer, and fresh verifier.
6. If that debug passes, immediately continue the original Gate D-2 handoff: freeze a new formal
   root, submit the exact test-12032 array, create the single 60-minute full heartbeat immediately
   after the verified job ID including PENDING, then merge and fresh-verify on terminal success.
7. If debug exposes a different reproducible engineering failure, unlimited executor-owned
   low-risk repair remains available under the original handoff. Preserve each failed root and
   rerun on a fresh path. A scientific-contract or coverage-floor failure remains `BLOCK`.

No sealed completion content, prompt/answer/token IDs, gold/label/correctness/outcome, selector,
or Gate E-G action is authorized. Send exactly one final `GATE_D2_FINAL_AUDIT` or `BLOCK` to the
planning thread with visible delivery confirmation.
