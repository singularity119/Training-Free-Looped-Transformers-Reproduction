# GATE_B_HANDOFF

Project/phase: LoopScope Phase 6  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor thread: `019fb439-6a3a-71d1-a917-0bca66701cf9`  
Executor title: `execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate B`  
Gate: B — HPC2 provenance and CPU admission  
State: AUTHORIZED after this file, live control, executor identity, and activation-packet hashes agree.

## Objective

Close only the remote facts needed to admit the first GPU smoke:

1. fast-forward the dedicated HPC2 clone to the single Gate B implementation commit;
2. verify the pinned environment, model/tokenizer metadata, dataset bytes, MMLU-Pro
   renderer/filter bundle, generation contract, 36-layer architecture, and Phase 4 known-window
   registry identity without model forward or outcome reads;
3. materialize one write-once, outcome-free Gate B provenance receipt containing the exact
   canonical test-12032 safe identity/order hash and four frozen smoke identities;
4. verify that the Gate C runtime entrypoint can import and plan the frozen two-pass path.

Non-objectives: no model-weight load, generation, replay, selector, panel, accuracy/outcome parsing,
GPU/CUDA/Slurm, general provenance framework, portability work, or Gate C execution.

## Admission and protected state

- Local repo: `loopscope-tflt`
- Branch: `loopscope`
- Exact base: `177cf1f7a960319c3a86a392319702f22c113ba1`
- Required ancestor: `43d57e95ccbb96535a2be5b94938396ec82fbabb`
- Expected local dirty state: only protected unstaged ` M AGENTS.md`
- Gate A: `PASS`; Gate C...G: `LOCKED`
- Never stage, edit, commit, stash, reset, or clean `AGENTS.md`.
- Stop before mutation if repo/branch/base/dirty state, control, handoff, or executor binding differs.

Authoritative sources:

- stable policy: `loopscope-tflt/AGENTS.md`
- mutable state: `.planning/loopscope_phase6_control.md`
- this handoff: `.planning/loopscope_phase6_gate_b_handoff.md`
- runbook: `loopscope-tflt/docs/loopscope_phase6.md`
- frozen card: `configs/loopscope/phase6_pre_answer_v2_card.json`

## Frozen remote envelope

- SSH: `ssh -o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes hpc2-hkustgz`
- Dedicated clone:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`
- Venv:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711`
- Write-once root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-b-provenance-<UTC>`
- Model snapshot:
  `/hpc2hdd/home/xhuang225/shared/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Dataset snapshot:
  `/hpc2hdd/home/xhuang225/shared/datasets/TIGER-Lab___mmlu-pro/default/0.0.0/b189ec765aa7ed75c8acfea42df31fdae71f97be`

The executor may use read-only SSH inspection throughout these project-scoped roots. Remote writes
are limited to one ff-only update of the dedicated clone and one fresh write-once Gate B root.

## Smallest implementation increment

Implement only what Gate C needs:

- `scripts/loopscope/run_qwen4_phase6_gate_b.py`
- `tests/test_loopscope_phase6_gate_b.py`
- optionally `src/tflt/loopscope/phase6_runtime.py` only if a small importable runtime-plan/helper
  is necessary for the Gate C entrypoint.

The script must have one acquisition mode and one fresh-process verify mode. Reuse existing
Phase 4 safe projection/renderer/tokenization helpers where possible. Do not copy the Phase 4
control-matrix framework or create new schemas unless a normal-path failure proves necessary.

The receipt may contain hashes, counts, versions, safe identities, category, question ID, source,
question/options-derived content hashes, prefix/token-ID hashes, sequence lengths, and four smoke
memberships. It must not persist target `answer`, `cot_content`, gold/label, generated text/tokens,
prediction, correctness, accuracy, gain, outcome values, logits, probabilities, or hidden tensors.
Validation answers used by the frozen 5-shot renderer may exist only inside rendered demonstrations;
they are not target labels and must not be exposed as separate receipt fields.

## Exact closures

Fail closed unless all of these match the card and live bytes:

- model/tokenizer revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`;
- model config hash and tokenizer hashes from the pinned snapshot; config reports exactly
  36 decoder layers and the live hidden size;
- dataset revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`;
- dataset component SHA-256:
  `dataset_info.json=ce81f7ec00c6701dc737889a3171a9dd32d2521914bc1e5029a3f8bb28df9aab`,
  `test arrow=4c741944e3b53719b433be6e7916e4ab9e8ff33f70e9c365614b4e317c0476bb`,
  `validation arrow=5730c2a9cd07a1ee5f70f4cd1940e43da1df7602c21977b7dc7ef51d496bdf00`;
- cached split counts test=`12032`, validation=`70`;
- environment Python 3.11.9, lm-eval 0.4.11, transformers 4.51.3,
  tokenizers 0.21.4, datasets 5.0.0, torch 2.3.1+cu121, accelerate 1.14.0;
- renderer `utils.py=74ab409c...`, group YAML `0271e3fd...`, default template
  `356e937a...`, literal case-sensitive regex `answer is \(?([ABCDEFGHIJ])\)?`,
  generation `until=Question:`, `max_gen_toks=2048`, greedy temperature 0;
- Phase 6 card SHA-256
  `31bd12dc3b58cb7df92c7617047cd86c290c678381ec744b9af73d01945e6f38`;
- known registry names exactly
  `no-loop,15:18,6:9,10:13,25:28,4:7,5:8,22:25`.

For the known registry, use only identity/path/hash evidence. The Phase 4 frozen panel manifest
file hash `02148127ef7b634106b1ee1044f591012d3543563e94344583480a7c3061eb79`
and selector-freeze file hash
`a4cc48a9b82b3d89b962cdefb55b6e4133c08a0aff3431f0de72c3750fc062c8`
must be checked at these exact outcome-safe paths:

- `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4b-20260716T204154Z/freeze/phase4_outcome_panel_manifest.json`
- `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4b-20260716T204154Z/freeze/phase4_selector_freeze.json`

Planning independently verified both live SHA-256 values after the initial Gate B blocker. The
earlier P4-C-root location was a provenance-pointer error only. Use `stat`/`sha256sum`; do not open
either file or any results file, and do not display outcome values.

Canonical target identities use only
`question_id,category,src,question,ordered_options`, ordered by numeric question ID then safe
content hash. Require count 12,032, unique count 12,032, missing/extra/duplicate/reorder zero.
Freeze four smoke identities deterministically as the first four distinct categories in canonical
order. Gate C must consume these exact bytes.

## Allowed actions

- Local focused implementation/tests within the three allowlisted paths.
- Local explicit-path commit and push to `origin/loopscope`.
- HPC2 read-only source/env/cache/config/dataset inspection.
- `git fetch origin loopscope` and `git merge --ff-only origin/loopscope` in the clean dedicated
  clone only; no merge commit.
- Offline CPU imports, tokenizer load, cached dataset load/render/hash, CLI help/dry-run/verifier.
- Create exactly one fresh Gate B root after all in-memory checks pass.
- At most two executor-owned low-risk repair loops that preserve this contract.
- Up to three concurrent, non-nested subagents for bounded read-only inspection or isolated local
  implementation/testing; parent alone owns Git, remote writes, integration, and terminal delivery.

## Forbidden actions

- Model weight loading or any forward/generate/replay call.
- CUDA/GPU/Slurm, scheduler queries/submission, or job retry.
- Network downloads, package/environment mutation, cache mutation.
- Reading target labels/answers/cot_content or any Phase 4/5/6 accuracy, correctness, gain,
  prediction, flip, or results JSON content.
- Editing the frozen card/formulas/candidates/anchor semantics or known registry membership.
- Remote destructive operations, checkout/reset/rebase/stash/clean, force push, or overwriting a
  prior run root.
- Gate C or later work.

## Decisive validation and stop rule

Run only:

1. focused local Gate B tests plus `git diff --check` and compile/help;
2. one offline remote acquisition producing the write-once receipt;
3. one fresh-process remote verification of that receipt;
4. one import/dry-plan check for the Gate C runtime entrypoint.

Do not run the full historical test suite. Once the receipt verifies and the Gate C entrypoint
imports without model/data execution, stop hardening and send the terminal packet.

Must-pass evidence:

- local and remote exact commit/clean provenance;
- receipt and manifest SHA-256, 12,032 identity/order closure, four smoke identities;
- exact environment/model/tokenizer/dataset/renderer/filter/generation/registry closure;
- explicit `model_weights_loaded=false`, `model_forward_executed=false`,
  `outcomes_read=false`, `target_labels_accessed=false`;
- verifier `PASS`;
- protected `AGENTS.md` unchanged and unstaged.

Audit-returned repair cap: one bundled cycle. A change to frozen science, identity definition,
renderer, registry, or information barrier is not an automatic repair; report `BLOCK`.

## Terminal requirement

Send exactly one visible `GATE_B_FINAL_AUDIT` or `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c`, retaining delivery confirmation. Include exact local and
remote commits/status, changed files, commands/exit codes, run root, receipt/manifest hashes,
safe identity counts/hash, four smoke identities, observed closure, repair loops, protected-state
confirmation, actions not taken, and requested decision `PASS / PASS_WITH_FIXES / BLOCK`.

No long-job monitor is needed for Gate B.
