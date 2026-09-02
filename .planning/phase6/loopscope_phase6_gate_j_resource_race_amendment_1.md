# GATE_J_RESOURCE_RACE_AMENDMENT_1

```text
Project/phase: LoopScope Phase 6 — MMLU 0-shot Prefix Trajectory
Gate: J
Planning/audit task: 019fb3de-2298-75f2-a083-0dca453ea79c
Authorized executor task: 019fbf4b-5d69-7931-9724-16b03e2bc3f9
Authority: USER_EXPLICIT_A40_A800_QUEUE_RACE
Decision: CONTINUE_SAME_GATE_SAME_EXECUTOR

Purpose:
- Keep the already-submitted A40 formal array and submit one scientifically equivalent A800 formal
  array. Retain only the leg whose first real array task starts earliest; cancel the other leg.
- This user-authorized amendment changes only queue/resource routing. The final scientific dataset,
  producer, metrics, population, sharding, selector and information barrier remain frozen.

Authoritative state:
- policy sha256=5f8783cb9826b5f715d61b872027f49e0dc4d2f5e29a155ab9b63ebc275aae76
- live control sha256=fb30abcd98a900c8f16008d501d1b3b8c119d6ced2e5e63894e34a225ced2b11
- original Gate J handoff sha256=fd0a9ac2a2d7d212d4eb24d4244ca2d18469c5fac0965c3e4c7890f65829a1cb
- connectivity supplement sha256=73adda7c6b8cc07eebaf33de10ad5990f401e447ac31c24b9b2c20ea925eaa0c
- immutable producer commit=37b454bca209b97512325475587cc919c13a4040
- local/origin and clean HPC2 clone must remain at that exact commit during both race legs
- local dirty state remains exactly protected unstaged M AGENTS.md

Existing A40 leg:
- job=10118660, array tasks 0-7, throttle 8
- partition/QoS/GPU=emergency_gpua40 / emergency_gpua40 / a40:1 per task
- state at amendment=all tasks PENDING(Priority), no task Start timestamp, no shard receipt
- scheduler prediction observed=2026-08-04T10:41:14 cluster time
- root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-j-formal-validation-1531-20260801T231500Z
- existing sole heartbeat automation id/name=loopscope-p6-gate-j-full-job-10118660

Authorized A800 leg:
- Before submission, make one fresh bounded scheduler check. If any A40 array task has already
  started, A40 has won: do not submit A800 and continue the A40 leg.
- Otherwise create one fresh write-once sibling root named
  phase6-gate-j-formal-validation-1531-a800-race-<UTC>.
- Freeze/copy the exact same canonical 1531/57 manifest, ordered identity/prompt digests, eight shard
  memberships, producer commit, card/schema, environment and scientific launcher arguments. Prove
  semantic manifest/shard digests are identical to the A40 leg before submission.
- Submit exactly one successful A800 array job with tasks 0-7, throttle 8, partition emergency_gpu,
  QoS emergency_gpu, gres gpu:a800:1 per task, and the same CPU, memory and wall-time envelope as the
  A40 array. Only job/resource/log/root fields may differ.
- A client-side or scheduler rejection before a real job ID may be repaired and resubmitted; once one
  real A800 array ID exists, do not create another competing A800 job unless ordinary Gate J failure
  recovery later requires it after the race has ended.

Race decision rule:
- The winner is the job set with the earliest actual non-empty Slurm Start timestamp among any raw
  array task, not submission time, predicted start, parent state or queue reason.
- If both legs start between observations, compare the recorded earliest raw-task Start timestamps.
  If timestamps are exactly tied at scheduler resolution, A800 wins.
- If one leg reaches a pre-start terminal rejection/failure while the other remains viable, the viable
  leg wins. If both fail before start, resume the exact executor for normal low-risk diagnosis.
- Immediately after the winner is independently confirmed, the exact executor must scancel the entire
  losing array, including any later-started loser task, and verify all loser tasks are terminal.
- Preserve both roots and scheduler evidence. Never delete or overwrite the loser root.
- Only the winner root may supply valid shards, merge input, verifier input or selector input. Never
  mix, salvage or compare trajectory values across A40 and A800 legs.
- Once a winner is frozen, all eight scientific shards must come from that one homogeneous GPU type.
  If winner shards later fail, use ordinary same-leg Gate J repair/retry rules; do not reactivate the
  cancelled race loser as a source of science.

Heartbeat amendment:
- Do not create a second automation. Update the existing sole heartbeat in place after A800 submission
  to monitor both exact array IDs and both roots.
- During the unresolved race, use a temporary 10-minute cadence to reduce simultaneous-start waste.
  The heartbeat remains read-only: on first-start/terminal evidence it must pause itself and send an
  AUTOMATION_TERMINAL_RESUME packet to exact executor 019fbf4b-5d69-7931-9724-16b03e2bc3f9;
  it must not cancel jobs itself.
- The resumed executor applies the race rule, cancels the loser, records the winner and then retargets
  the same sole heartbeat to the winning array/root at the normal formal-full 60-minute cadence.
- Healthy dual-PENDING state remains quiet after one bounded check.

Unchanged science and safety:
- The existing final-commit debug preflight remains the admission evidence because code, launcher,
  environment, producer and verifier are unchanged; no new engineering/smoke job is authorized.
- No test split, target/gold/outcome access, selector tuning, loop execution, Gate K/L, destructive
  cleanup, mixed-GPU scientific merge or code change is authorized by this amendment.
- Formal acquisition still terminates in one verified 1531/57 winner-root merge and the exact frozen
  V2 selected window/legal ABSTAIN.

Terminal requirement:
- Continue the same Gate J until GATE_J_FINAL_AUDIT or a genuinely new material BLOCK. The final packet
  must include both race job IDs, roots, earliest Start evidence, winner/loser decision, loser
  cancellation closure, heartbeat lifecycle and explicit proof that only winner artifacts entered
  verification and selection.
```
