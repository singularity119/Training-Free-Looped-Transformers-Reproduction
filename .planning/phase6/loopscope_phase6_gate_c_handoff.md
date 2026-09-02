# GATE_C_HANDOFF

Project/phase: LoopScope Phase 6  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb452-e2af-7bf2-a411-612c81971245`  
Title: `execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate C`  
Gate: C — four-identity end-to-end GPU smoke  
State: AUTHORIZED only after this file, live control, executor identity, and activation hashes agree.

## Objective and stop rule

Implement the smallest real Qwen3 two-pass runtime and prove it on exactly the four identities
frozen by Gate B:

1. native no-loop deterministic CoT generation;
2. unique answer-span/token anchor resolution;
3. exact anchor-prefix no-loop replay with raw `B0...B36`;
4. full-vocabulary entropy/KL plus all diagnostic hidden trajectories;
5. duplicate determinism and independent sanitized-record verification.

Once all four identities pass, stop and return Gate C. Do not start test-12032 or add hardening.

## Admission and provenance

- Local repo/branch: `loopscope-tflt` / `loopscope`
- Exact base: `404c1b5118aec9f8478eae3ef7a9a4a575042c5b`
- Expected local dirty state: only protected unstaged ` M AGENTS.md`
- Gate B run root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-b-provenance-20260730T183107Z`
- Gate B manifest SHA-256:
  `ccb148bd61971d24ad81b240cbe1b5f0810e06e620fb568473a61b2147b44a1d`
- Gate B receipt SHA-256:
  `e60d10ad355504a0436b1fe3dcdc9dce2ca10a5b0d342bdcc016a431887aeedf`
- Gate B verifier SHA-256:
  `aad451fb9ed5f5e2a37311a97ff7cd02c28ef961753d67d4b617d3c80c6d7c31`
- Ordered identity SHA-256:
  `ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5`
- Ordered safe token closure SHA-256:
  `0d7e9ef3139f687fd3f15d825e883e52d1f84a504cae134ad4b17ebd6ad82108`
- Gate D...G remain locked.

Read completely before mutation:

- `loopscope-tflt/AGENTS.md`
- `.planning/loopscope_phase6_control.md`
- this handoff
- `loopscope-tflt/docs/loopscope_phase6.md`
- `configs/loopscope/phase6_pre_answer_v2_card.json`

Stop if binding/base/dirty state or Gate B receipt bytes differ.

## Frozen smoke membership

Use exactly these Gate B verifier rows, including the frozen rendered-prefix/token-ID hashes:

1. ordinal 0, business, qid 70  
   `70:business:ori_mmlu-business_ethics:7967613730e252a49314cfb3dfb30134f09b9e55c0b8404c6a3f6a51d51ac5f7`
2. ordinal 789, law, qid 866  
   `866:law:ori_mmlu-professional_law:4ef6e81c98e41ee505138e6df2df7a15b1014209cac8fcc126bf1370fd90db88`
3. ordinal 1890, psychology, qid 1986  
   `1986:psychology:ori_mmlu-professional_psychology:2478839201a23311fb6076ddb3e6b47d8e8391fd636f2ef83e8defcdb6da4e0c`
4. ordinal 2688, biology, qid 2804  
   `2804:biology:ori_mmlu-high_school_biology:1f07729da6ee685072170f005de3debca8aa86842b2927df6fb8d935cdca4bf8`

No replacement, drop, extra sample, or target-label access is permitted.

## Frozen scientific/runtime contract

- `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`
- bfloat16, 36 layers, hidden size 2560; local files only
- MMLU-Pro test, validation 5-shot renderer, exact Gate B safe prefix bytes
- greedy, `do_sample=false`, `temperature=0`, `max_gen_toks=2048`, stop string `Question:`
- generation once and replay once per replicate; zero loop insertions
- extractor regex exactly `answer is \(?([ABCDEFGHIJ])\)?`, case-sensitive, capture group 1,
  collect all spans and require exactly one
- probe is the generated token immediately before the first answer-content token
- replay IDs are prompt IDs plus generated IDs strictly before that answer token
- replay: `use_cache=false`, `output_hidden_states=true`
- raw `B36` is the single FinalNorm pre-hook input; post-norm hidden state cannot replace it
- selector inputs are only full-vocabulary `H_l` and `KL(p_l||p_36)`
- hidden RMS-L2/cosine/cosine-distance and raw adjacent angular distance remain diagnostic-only
- no selector or outcome computation.

Run two deterministic replicates per identity. Generated/replay ID hashes, anchor indices,
sanitized scalar records, and final next-token closure must agree across replicates.

## Smallest implementation

Allow only:

- `src/tflt/loopscope/phase6_runtime.py`
- `scripts/loopscope/run_qwen4_phase6_gate_c.py`
- `tests/test_loopscope_phase6_gate_c.py`

Reuse Gate A/B and existing probe/runtime helpers. Do not modify frozen card, schemas, anchor,
acquisition, selector, Gate B artifacts, Phase 4/5 files, or build a general hook framework.

The runtime script needs only focused CPU fixtures, `dry-run`, one GPU smoke mode, and one
fresh-process verifier mode. Parent executor alone commits/pushes, syncs remote Git, submits or
controls Slurm, integrates delegated work, and sends the terminal event.

## HPC2/GPU authority

- SSH:
  `ssh -o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes hpc2-hkustgz`
- Dedicated clone:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`
- Venv:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711`
- Fresh root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-c-smoke-<UTC>`

The executor may inspect `squeue/sinfo` read-only, then choose one available single-GPU path:
prefer `emergency_gpua40` with `gpu:a40:1`; an available A800-equivalent path is allowed when it
materially shortens queue/runtime. Record exact partition/QoS/GRES, node, job ID, GPU model, and
elapsed time. Use 8 CPU, at most 64 GB RAM, and a walltime sufficient for the exact four-sample
smoke only.

One primary job is authorized. At most one fresh-root retry is allowed only for a preserved
scheduler/transient infrastructure failure or GPU OOM before a valid smoke receipt exists. It may
switch A40/A800 but may not change samples, generation limits, dtype, metrics, or code after seeing
scientific outputs. Never overwrite a prior attempt. Any anchor/replay/numeric failure is a Gate C
`BLOCK`, not a retry target.

If the job outlives the executor turn, attach exactly one executor-owned smoke monitor at a
10-minute interval after the real job exists. Each healthy wake performs one bounded read-only
status check and remains quiet. On terminal or a material blocker, pause or delete the monitor
first, then visibly trigger this exact executor with `AUTOMATION_TERMINAL_RESUME`; fallback is
`AUTOMATION_RELAY_REQUIRED` to planning. The cadence grants no retry, cancellation, resubmission,
scientific, outcome, or next-Gate authority. Planning does not poll or create a duplicate monitor.

## Must-pass evidence

For all 4 identities and both replicates:

- unique span, unique token mapping, probe exists;
- rendered prompt/token closure equals Gate B;
- generation/replay ID hashes and duplicate results agree;
- replay final argmax equals Pass 1 answer-first token;
- `B0...B36` exactly 37 raw boundaries; FinalNorm pre-hook count exactly 1;
- FinalNorm(raw B36) closes the post-norm final hidden within frozen numeric tolerance;
- H/D and hidden geometry each length 37; angular length 36; all finite;
- `D36<=1e-6`; hidden-to-final endpoints close; cosine distance is exactly `1-cosine`;
- adjacent angular uses raw boundaries and remains `[0,1]`;
- generation_count=1, replay_count=1, loop_insertions=0 per replicate;
- sanitized records contain no prompt/generated text, token IDs, answer content/hash, prediction,
  target label/gold, correctness, accuracy, gain, outcome, logits, probabilities, or tensors;
- independent verifier returns `PASS`.

Persist one smoke manifest, sanitized smoke records, duplicate-closure receipt, and independent
verifier receipt with SHA-256. Generated text/IDs and answer content may exist in memory only and
must never be printed or persisted. Do not access target `answer` or `cot_content`.

## Validation and agility budget

Run only:

1. focused local Gate C tests, compile/help/dry-run, `git diff --check`;
2. one explicit-path commit/push and remote ff-only sync;
3. one exact four-identity GPU smoke job;
4. one fresh-process verifier.

No full suite, test-12032 dry acquisition, extra determinism matrix, plotting, selector, or report.
At most two executor-owned low-risk code repair loops before the first valid GPU result; after real
smoke evidence, do not change scientific code to make the smoke pass.

## Forbidden actions

- Gate D or any sample beyond the frozen four.
- Any loop insertion or loop outcome cell.
- Target labels/cot_content, correctness, accuracy, gain, Phase 4/5 outcomes, selector/panel.
- Network/model/data download; package/cache/environment mutation.
- Stash/reset/rebase/clean/force push, overwrite/delete, or modifying protected `AGENTS.md`.
- Cancellation/resubmission beyond the exact retry envelope above.

## Terminal

Send exactly one visible `GATE_C_FINAL_AUDIT` or `BLOCK` to planning
`019fb3de-2298-75f2-a083-0dca453ea79c`. Include local/remote commit and state, changed files,
focused tests, job/resource/terminal evidence, run root, artifact hashes, 4x2 membership and
anchor/replay/numeric closures, repair/retry accounting, information-barrier confirmation,
actions not taken, and requested decision `PASS / PASS_WITH_FIXES / BLOCK`.
