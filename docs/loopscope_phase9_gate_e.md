Gate E completed 2026-09-24: planning PASS. The 12 new Lag1 cells and 47 reused raw-score cells closed; no Holm-adjusted primary contrast passed. See `.planning/phase9/loopscope_phase9_gate_e_acceptance.md` and the outer `资产/报告/phase9/LoopScope_第九阶段GateE逐轮方向实验报告.md`. No new compute is authorized.

# Phase 9 Gate E: switchable direction policy, Lag1 formal panel

Gate E is a post-completion exploratory extension to the closed Phase 9 A–D panel. The exact frozen contract is `.planning/phase9/loopscope_phase9_gate_e_lag1_contract.md`; executor and operational authority are in the Phase 9 control and Gate E handoff.

## Frozen panel

The shared Gate E runtime and scoring adapter require two independent explicit selectors: `direction_policy=fixed_t0|lag1` and `intervention_mode=spectral|matched_norm`. All four combinations use the same prefix mask, native residual, FP32 reduced SVD, and answer-position update path. Missing or unknown selector values fail; neither is inferred from an arm name. The formal manifest contains only twelve `direction_policy=lag1` cells: the retained Qwen3-4B-Base windows `13:16` and `15:18`, plus Qwen3-1.7B-Base `12:15`; K=3 and K=4; `spectral` (`Lag1-Online`) and `matched_norm` (`Lag1-Matched-norm`). Every cell scores the existing complete MMLU test pool of 14,042 identities. The test set has been seen in Phase 9, so this is an explicitly post-outcome exploratory comparison. Full fixed-t0 raw scores are reused from the verified Phase 9 panel and are not rerun.

For each prompt, t0 is unchanged and fits D0 from all valid retained prompt-token residual rows. `fixed_t0` uses v0 at every t>=1. Under formal `lag1`, t1 uses v0 while fitting D1 from the current native residual before the transform; t2 uses D1 and, for K4, fits D2 for t3. The final round never fits an unused direction. SVD is uncentered and unnormalized over the complete effective prefix rows, with FP32 reduced SVD and CUDA `gesvd`; direction sign is canonicalized and the vector normalized. `lambda=0.5`, `alpha=1`, and Euler step `1/K` are frozen. Bounded records label each round with `applied_t`, `direction_used_fit_t`, and `direction_fit_t`.

`matched_norm` uses the selected direction on its own trajectory, then uniformly scales the answer-position residual to the norm produced by directional damping. Its trajectory can diverge from `spectral` for K>2; describe it as the rule's own equal-norm control. At K2, `fixed_t0` and `lag1` both match the corresponding original Phase 9 arm.

## Local contract checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase9_gate_e_*.py'
python3 -m py_compile src/tflt/loopscope/phase9_gate_e_runtime.py scripts/loopscope/run_phase9_gate_e_accuracy.py scripts/loopscope/analyze_phase9_gate_e.py
```

The real debug job runs on the cluster `debug` partition, under 30 minutes per job. Its CLI explicitly selects both direction policies and both intervention modes. One job per model covers both windows, K3/K4, an ordinary and upper-tail prompt, policy agreement through t1 and divergence from t2, K4 lag1 t3 sourcing D2, prompt-state release, and K2 score-plus-answer-residual equivalence against the frozen Phase 9 `Online-t0` and `Matched-norm` implementations. It records no correctness or accuracy. The predeclared absolute tolerances are exact equality (`0.0`) for FP32 raw choice scores and returned answer-residual rows, matching the frozen Phase 9 real-debug's zero-delta checks for zero-strength and candidate-order equivalence. Both scores and transient residual rows are compared, while residual vectors are not written to artifacts.

## Acquisition

The Gate E manifest and canary IDs are write-once, label-free files:

```bash
python scripts/loopscope/build_phase9_gate_e_manifest.py \
  --pool "$POOL" --scope FORMAL_TEST --output "$RUN/manifest/score_manifest.json"
python scripts/loopscope/build_phase9_gate_e_canary_indices.py \
  --pool "$POOL" --output "$RUN/manifest/canary_indices.json"
```

Each score worker invocation must explicitly pass `--direction-policy lag1` and an `--intervention-mode` matching its manifest cell (`spectral` or `matched_norm`). A bounded preflight may use `--direction-policy fixed_t0`; full formal selection rejects it.

Run one formal canary per model on A800 with the mixed canonical/long-prefix 512 identities. The score shard is preserved and counts toward the 14,042 identities. Measure SVD time, throughput, peak allocated/reserved memory, and the actual loading/concurrent-process shape before selecting packing. Packing 2 requires a two-process same-card run with the frozen batch-one, long-prefix workload and at least 15% memory headroom; a single-process canary is insufficient.

Full acquisition is 12 × 14,042 scores (168,504 new sample-cell records), in fresh paths. Keep no more than two GPU jobs active. Retry only invalid or missing identities after diagnosing a low-risk engineering failure, and preserve every failed attempt. The total 160 GPU-hour ceiling includes debug, canary, failures, and retries.

Each worker is invoked with `scripts/loopscope/phase9_gate_e_accuracy.sbatch`. Submit one bounded wave at a time and keep the executor's single heartbeat at the cadence for the active job class. Do not let two submitted/running A800 jobs exceed the two-GPU limit.

## Closure and analysis

Verify all twelve cells cover each canonical identity exactly once, with finite raw A/B/C/D log-likelihood scores, 57 subjects, fixed revisions, and no gold field. Independently reverify the original Phase 9 18 new score cells and 29 historical score cells from raw per-identity files. No partial accuracy or correctness is read before all three closures pass.

After closure, run `scripts/loopscope/analyze_phase9_gate_e.py` once. Its twelve-item primary family compares Lag1-Online with original Online-t0 and Lag1-Matched-norm for each of six model/window/K combinations. It uses exact two-sided McNemar tests with one Holm adjustment at 0.05, plus subject-stratified paired bootstrap intervals (10,000 replicates, seed `20260923`). Lag1-Online versus Loop, original Matched-norm, and Native are descriptive only. The two primary comparisons for a cell must both pass the twelve-item Holm family to support exploratory direction-update value.

Outputs are a new extension `analysis.json`, `cells.csv`, `primary_contrasts.csv`, `descriptive_contrasts.csv`, and the separate human report under the outer `资产/报告/phase9/` directory. The original 47-cell report and Phase 9 A–D decisions are not revised by this extension.
