# LoopScope Phase 5 runbook

> `PHASE5_ACTIVE / GATE_D_AUTHORIZED`. This runbook is procedure, not execution authority.
> The controlling planning file is `../../.planning/loopscope_phase5_control.md` at the
> parent workspace level. Only the executor and Gate named there may act; later Gates remain
> locked until the planning/audit task issues their exact authorization.

## 1. Experiment identity

```text
hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
model=Qwen/Qwen3-4B-Base
task=mmlu
dataset=cais/mmlu
num_fewshot=5
trajectory=validation 1531 / 57 subjects / one native no-loop forward
outcome=test 14042 / exact logged-sample join
selector=choice-space entropy-KL CONSENSUS width4
panel=baseline + fixed 15:18 + blind high3 + blind low3
loop=K3 Euler(alias=damped_euler), alpha1, step1/3, horizon1, block, cache-first
```

The formal contract lives at `configs/loopscope/phase5_card.json`; its narrow validator is
`src/tflt/loopscope/phase5_schema.py`. The card does not authorize a model forward, trajectory
acquisition, selector freeze, test outcome, GPU, or scheduler action by itself.

## 2. Planning preflight and Gate A provenance

The planning task is allowed to cache a missing `Qwen/Qwen3-4B-Base` snapshot from a domestic
Hugging Face mirror on the HPC2 login node and inspect config/tokenizer metadata only. The approved
cache root is:

```text
/hpc2hdd/home/xhuang225/shared/hf_home
```

Preferred environment:

```text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
HF_HUB_DISABLE_XET=1
```

This exception does not authorize model forward, CUDA, Slurm, test outcome, or later-Gate work.

Planning preflight resolved the following snapshot metadata without opening the tensor weights:

```text
commit=906bfd4b4dc7f14ee4320094d8b41684abff8539
architecture=Qwen3ForCausalLM
layers=36
hidden_size=2560
vocab_size=151936
dtype=bfloat16
tie_word_embeddings=true
final_norm_weight=model.norm.weight
tied_output_weight=model.embed_tokens.weight
spaced_choice_ids=A:362,B:425,C:356,D:422
plain_choice_ids=A:32,B:33,C:34,D:35
safetensors_total_size=8045591552
snapshot_cache_complete=true
weight_shards=3
incomplete_files=0
```

Gate A hash-binds these values, the runtime module classes/source, the reused exact MMLU renderer
anchor, and gold-free validation/test identity evidence. Native final logits still require the
later authorized Gate B numerical closure; Gate A only freezes its fail-closed contract.

## 3. Gate A: contract and provenance

### Admission

- the user explicitly activates Phase 5;
- planning resolves every material choice in the control;
- `AGENTS.md` is updated to the activated Phase 5 control only after that authorization;
- planning creates exactly one Gate A executor.

### Required artifacts

Gate A uses one minimal contract set:

```text
configs/loopscope/phase5_card.json
configs/loopscope/phase5_card_schema.json
configs/loopscope/phase5_known_outcome_registry.json
configs/loopscope/phase5_gate_a_receipt.json
src/tflt/loopscope/phase5_schema.py
scripts/loopscope/verify_phase5_gate_a.py
tests/test_loopscope_phase5_gate_a.py
```

The card must bind exact model/tokenizer/manifest commit, dtype, layer count, hidden size, module
paths, choice token surfaces and IDs, dataset revision/fingerprints, lm-eval version, all MMLU task
YAML/source hashes, renderer and prompt contract, validation/test identity namespaces, loop kernel,
bootstrap streams, selector, registry and panel constructor.

### Minimum tests

1. General boundary formulas produce `B_0...B_L`, starts `0...L-4`, SHIFT `1...L-5` and
   CONSENSUS `4...L-8`. Reproduce the resolved `L=36`, starts `0...32` and CONSENSUS `4...28`.
2. Inclusive `15:18` maps exactly to `B_15 -> B_19`.
3. `strategy=euler`, K=3 and alpha=1 invoke the damped-Euler kernel exactly three times with
   step `1/3` and horizon `1`.
4. Cache-first/block wrapper restores the model and does not contaminate the next cell.
5. All four spaced choice surfaces are unique one-token continuations. Any multi-token result
   blocks the Gate; no first-token approximation.
6. Synthetic trajectory records close entropy, KL, RMS-L2, cosine, derived cosine distance and
   final-boundary invariants.
7. Selector recomputation is deterministic and rejects gold, correctness and outcome fields.
8. Historical baseline/15:18 reuse requires exact per-sample closure; otherwise both remain fresh.

Forbidden: formal model trajectory, MMLU test outcome, selector tuning, or a future executor.

## 4. Gate B: native trajectory and selector freeze

### Smoke

Run exactly four validation identities spanning more than one subject. Verify:

- one native forward per identity; `use_cache=false`; loop insertion count zero;
- exact last non-padding answer position and renderer prompt hash;
- `B_0...B_L` and final norm/LM-head closure;
- A/B/C/D choice distributions and all five finite trajectories;
- `KL(B_L)=0`, L2(B_L)=0, cosine(B_L)=1, cosine-distance(B_L)=0;
- full logits, vocabulary probabilities and hidden tensors are absent from persisted artifacts.

### Formal acquisition

Acquire all 1,531 validation identities and 57 subjects once. Choice/output and hidden geometry
must come from the same forward; a separate B-2 rerun is forbidden. Then run the outcome-blind
analyzer and freeze:

```text
all supported-window scores
eligibility and failures
published ranking
selected_window or ABSTAIN reason
selection_frequency
raw_high3
panel_high3 after the user-approved collision rule
blind_low3
fixed comparator 15:18
complete panel identities
card/source/trajectory/analysis/panel hashes
```

Gate B must not import or open test logged samples, target gold, correctness, or any current-cell
window outcome.

The authorized Gate B entry point is:

```text
scripts/loopscope/run_qwen4base_phase5_gate_b.py
```

Its ordered commands are `admission`, `freeze-smoke`, `write-launcher --mode smoke`,
`acquire-smoke`, `freeze-formal`, `write-launcher --mode formal`, `acquire-shard`, `merge`,
`select`, and `verify`. Every command remains bound to the exact implementation commit and the
write-once run root declared by the Phase 5 control and Gate B handoff.

## 5. Gate C: sealed full outcomes

### Panel

At most eight unique cells:

```text
baseline
fixed_15_18
blind_high_1
blind_high_2
blind_high_3
blind_low_1
blind_low_2
blind_low_3
```

Deduplicate identical windows. A selected window is not an extra cell. If selected equals 15:18,
it remains a known-comparator hit. If the selector abstains, keep the frozen high/low panel.

### Execution policy

1. Freeze command argv, environment, snapshot, card, renderer, test identity and panel hashes.
2. Run a limit smoke for each distinct recipe class before full acquisition.
3. Run every formal cell on the same 14,042 identities with `log_samples=true`.
4. Baseline has no wrapper. Every loop cell must prove K3, step 1/3, three body calls, block mode,
   cache-first and model restoration.
5. Write every cell to a new directory. Never overwrite a complete or partial prior attempt.
6. Seal results and expose only scheduler state, completion count, identity/hash checks and errors.
7. Do not expose partial accuracy, gain, flip counts, prediction distributions or any scientific
   summary before the complete panel is sealed.

Gate C terminal evidence is one `GATE_C_FINAL_AUDIT` or one `BLOCK`.

## 6. Gate D: one-shot unseal

Admit only when every required unique cell is terminal, complete, hash-closed and joined to the
same canonical test manifest. Unseal all cells once, then compute in one analysis family:

- per-window paired accuracy gain and subject-stratified paired-bootstrap 95% CI;
- High3 mean absolute gain and High3-Low3 enrichment;
- selected vs baseline and selected vs 15:18;
- panel-local regret;
- correct-to-wrong, wrong-to-correct, exact two-sided McNemar;
- secondary score/gain Spearman correlation;
- eligible count, selection frequency and ABSTAIN reason;
- secondary geometry/output and geometry/gain diagnostics.

An independent verifier must reconstruct identities, cell metrics, bootstrap streams and terminal
label from raw sealed samples. The final report must keep ranking, absolute gain, comparator
competitiveness and ABSTAIN distinct.

Do not repair the selector, change the panel, rerun an old Gate for a more favorable result, or
extend to another K, width, model or intervention after unseal.

The authorized entry point is:

```text
scripts/loopscope/run_qwen4base_phase5_gate_d.py preflight --expected-commit <implementation-commit>
scripts/loopscope/run_qwen4base_phase5_gate_d.py execute --expected-commit <implementation-commit>
```

`preflight` hashes and validates the card, selector, panel, test manifest, Gate C completion
receipt, and all eight sealed result files without parsing the result payloads. `execute` first
repeats that closure, creates one write-once unseal marker, then performs exactly one analysis
pass and one independent raw-payload verifier pass on the login CPU. Once the marker exists,
the command refuses every second unseal or analysis attempt.

### Gate D terminal result

The one-shot HPC2 login-CPU execution completed with `unseal_count=1`, `analysis_count=1`, and an
independent verifier `PASS`. The frozen selector remained the legal
`ABSTAIN_NO_POINT_ELIGIBLE` (`eligible_count=0`, no selected window), yielding terminal label
`H5_ABSTAIN_WITH_RANKING_RESULT`.

High3-Low3 enrichment was `+4.156578 pp` with subject-stratified paired-bootstrap 95% CI
`[+3.695995, +4.609980] pp`, but High3 absolute mean gain was `-12.958743 pp` with 95% CI
`[-13.507216, -12.427005] pp`. The blind-six selector score-gain association was
`rho=-0.314286`, exact two-sided `p=0.563889`. The fixed `15:18` comparator was the panel best at
`73.401225%`, a paired `+0.384561 pp` over the `73.016664%` baseline, with 95% CI
`[-0.021364, +0.797607] pp` and exact McNemar `p=0.071967`.

These results preserve the distinction between relative ranking enrichment and positive absolute
gain: Phase 5 does not claim a selected success or a universal cross-model automatic window
selector. The canonical Chinese report is
`../../资产/报告/LoopScope_第五阶段实验结果与结论.md`; write-once
analysis artifacts remain under the Gate C run root's `gate_d/` directory.

## 7. Gate terminal packet

Every executor sends exactly one terminal packet:

```text
GATE_X_FINAL_AUDIT
gate=X
executor_task_id=<id>
base_commit=<sha>
final_commit=<sha-or-NONE>
scope=<authorized scope>
artifacts=<paths and sha256>
must_pass=<itemized evidence>
forbidden_checks=<proof no forbidden action occurred>
known_limitations=<itemized>
recommended_planning_verdict=PASS|PASS_WITH_FIXES|BLOCK
```

If blocked, replace the header with `BLOCK` and include the smallest material blocker, attempted
safe checks, and the exact new authority or external state needed. Planning alone issues the actual
`PASS / PASS_WITH_FIXES / BLOCK` decision and creates the next executor.

## 8. Terminal stop

After Gate D is audited, Phase 5 becomes terminal. Preserve the card, selector, ABSTAIN decision,
panel, sealed raw outcomes, verifier receipt and Chinese summary. Do not start a K sweep,
variable-width study, new model, representation intervention or Phase 6 without a new user-approved
hypothesis and control.
