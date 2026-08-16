# LoopScope Phase 7 Gate A sidecar

This sidecar closes the producer, verifier, and depth-general V3 contracts for
the three frozen Base repositories. It is outcome-blind: the persisted
trajectory contains only model/revision metadata, safe identity/subject,
zero-loop admission counters, scalar trajectories, and raw-boundary diagnostic
scalars. Prompt text, targets, correctness, outcomes, token IDs, input IDs,
full logits/probabilities, hidden tensors, and loop residuals are rejected.

The Phase 7 selector receives only `identity`, `subject`, `H`, and `D`, where
`H` is choice entropy and `D` is `KL(p_l || p_L)`. Candidate domains are
generated from the live layer count by `ceil(0.30 L)`, central inclusive block
bounds, widths 3/4/5/6, and width/start order. No fixed 36-layer table is
embedded in the producer.

## Gate B producer/verifier contract

Freeze the single common smoke manifest before any model forward.  The
manifest builder reads only the validation `question/choices/subject` safe
projection, follows the canonical task/validation order, and selects two rows
from each of the first two subjects:

```bash
PYTHONPATH=src python3 scripts/loopscope/run_phase7_build_smoke_manifest.py \
  --cache-dir "$HF_DATASETS_CACHE" \
  --output <fresh-gate-b-input>/phase7_smoke_manifest.json
```

The producer renderer entrypoint is
`tflt.loopscope.phase7_renderer:render_mmlu_smoke`.  It uses the existing
lm-eval 0.4.11 `ConfigurableTask.fewshot_context` path with five dev
demonstrations, creates an in-memory empty final-turn sentinel, and returns
only ephemeral tokenizer tensors.  The manifest persists only
`canonical_identity`, `subject`, `task_name`, and `validation_index`; it never
persists validation targets, prompt text, or token IDs.

After live read-only admission closes model and tokenizer revisions, the
launcher is:

```bash
PYTHONPATH=src python3 scripts/loopscope/run_phase7_base_trajectory.py \
  --model-key <qwen25_3b|llama32_3b|gemma2_2b> \
  --model-repo <exact frozen repository> \
  --revision <resolved local model revision> \
  --tokenizer-revision <resolved local tokenizer revision> \
  --identity-manifest <safe identity/subject manifest> \
  --renderer-entrypoint <audited_module>:<function> \
  --cache-dir <audited local HF cache> \
  --output <write-once sanitized JSONL>
```

The launcher requires local cache access, `trust_remote_code=false`, one
native `use_cache=false` forward per identity, `batch_size=1`, bfloat16, and
the standard plain non-chat MMLU renderer's full `Answer:` prefix. It admits
the four exact continuation surfaces in memory and does not persist their
token IDs. Runtime loading propagates the audited `--cache-dir` to both
tokenizer and model loads, binds the model and every tensor input explicitly
to `cuda:0`, and fails closed if CUDA is unavailable or devices differ. The
verifier is:

```bash
PYTHONPATH=src python3 scripts/loopscope/verify_phase7_base_trajectory.py \
  --input <sanitized JSONL> \
  --model-key <model key> \
  --layer-count <live L> \
  --expected-count 1531 \
  --expected-subjects 57 \
  --receipt <write-once verification receipt>
```

The four-identity Gate B smoke needs, for each of the three models: exact
resolved model and tokenizer revisions, verified local cache paths, live `L`,
the decoder/embedding/FinalNorm/lm-head path closure, choice-surface closure,
four safe canonical identities with subjects, the audited renderer entrypoint,
one output root, and enough GPU memory for one bfloat16 batch of size one. A
debug preflight is limited to one identity per model: load from the audited
local cache, perform the one native zero-loop forward, verify the scalar record
in a fresh process, and finish within 30 minutes. It must not sample validation
rows beyond the four smoke identities, access gold/outcome channels, or create
formal scoring artifacts.

## V3 analyzer/verifier contract

Gate A synthetic checks use small bootstrap counts only. Formal scoring remains
locked to Gate D. The callable analyzer and fresh-process verifier are:

```bash
PYTHONPATH=src python3 scripts/loopscope/run_phase7_v3.py \
  --input <verified sanitized JSONL> \
  --layer-count <live L> \
  --output <write-once V3 JSON>

PYTHONPATH=src python3 scripts/loopscope/verify_phase7_v3.py \
  --input <verified sanitized JSONL> \
  --analysis <V3 JSON> \
  --layer-count <live L> \
  --receipt <write-once V3 receipt>
```

The verifier starts a fresh Python process and rechecks record membership,
candidate enumeration, formulas, terminal state, and the selector projection.
Phase 7 deliberately persists no historical selector digest; the frozen
method/version, seed, and candidate/ranking fields are sufficient for this
Gate A sidecar contract.

## Current Gate A admission state

The repair closes the causal-LM output path without requiring
`last_hidden_state`: the producer captures the native FinalNorm output from the
single forward, closes its lm-head logits against the native logits (including
Gemma 2 final-logit softcapping), and never normalizes raw `B_L` twice.

Metadata/tokenizer-only acquisition used the authorized route order and the
shared cache at `/hpc2hdd/home/xhuang225/shared/hf_home/hub`:

| model | cache / source route | live revision and depth | choice closure | candidate domain |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-3B` | cache miss; `hf-mirror.com` success; no official fallback | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, `L=36`; all four module paths present | PASS: lengths `1,1,1,1`, mutually distinct | central `[11,24]`; widths `3/4/5/6`; 42 windows |
| `meta-llama/Llama-3.2-3B` | cache miss; token-backed `hf-mirror.com` success; official fallback not used | `13afe5124825b4f3751f836b40dafda64c1ed062`, `L=28`; all four module paths present | PASS: lengths `1,1,1,1`, mutually distinct | central `[9,18]`; widths `3/4/5/6`; 26 windows |
| `google/gemma-2-2b` | cache miss; token-backed `hf-mirror.com` success on the user-requested recheck; official fallback not used | `c5ebcd40d208330abc697524c919956e692655cf`, `L=26`; all four module paths present | PASS: lengths `1,1,1,1`, mutually distinct | central `[8,17]`; widths `3/4/5/6`; 26 windows |

The cache audit found no model weights or weight-index files for any target.
No credential content, model weights, datasets, model forward, CUDA, GPU, or
Slurm were used. All three frozen Base repositories now pass Gate A admission;
the complete machine-readable admission table is
`configs/loopscope/phase7_model_admission.json`.

## Gate E fixed 3/3/2 outcome panel

Gate E is an independent outcome path.  Its frozen card is
`configs/loopscope/phase7_gate_e_outcome_card.json`: Qwen and Llama each have
one no-loop baseline plus three width-four windows across `k={2,3}` and
`cache={first,last}`; Gemma has one baseline plus its two finite windows, for
13/13/9 and 35 cells total.  The runner keeps raw lm-eval samples only in
memory and writes only canonical test identity, subject, cell configuration,
and the paired-analysis correctness bit.

On HPC2, prepare a fresh formal root before any test forward, then use a debug
root referring to that frozen identity manifest:

```bash
PYTHONPATH=src python scripts/loopscope/run_phase7_gate_e_panel.py prepare \
  --mode formal --run-root <fresh-formal-root> --expected-commit <commit> \
  --cache-dir "$HF_DATASETS_CACHE"

PYTHONPATH=src python scripts/loopscope/run_phase7_gate_e_panel.py prepare \
  --mode debug --run-root <fresh-debug-root> --expected-commit <commit> \
  --formal-identity-manifest <formal-root>/inputs/canonical_test_manifest.json
```

`build-launch` freezes a bounded Slurm worker-pool plan.  The debug path first
measures one-process peak memory/throughput and sampled whole-GPU utilization,
then validates increasing independent-process packing levels with about 10%
headroom.  Only the highest passing bounded concurrency may be used by the
formal worker pools.  All children retain independent model instances, logs,
outputs, and write-once cell paths; `batch_size=16` never changes.

If a formal worker-pool attempt has an ordinary resource or launcher failure,
preserve that root and never rerun a completed cell.  A fresh `formal_retry`
root may reference only the completed cell artifact roots from the preserved
attempt, while scheduling only the remaining invalid/missing indices:

```bash
PYTHONPATH=src python scripts/loopscope/run_phase7_gate_e_panel.py prepare \
  --mode formal_retry --run-root <fresh-retry-root> --expected-commit <commit> \
  --retry-source-run-root <preserved-formal-root> \
  --retained-cell-indices <comma-separated-completed-indices>
```

The retry root rechecks the frozen card and canonical membership, retains
read-only references rather than copying outcome rows, and permits the
pre-outcome verifier and one later paired analysis only after all 35 logical
cells are complete.  Its generated Slurm launcher tolerates compute nodes
where the optional modules profile is absent, while still loading the audited
venv and remaining offline.  Provenance closure resolves `/usr/bin/git` when
available, so a node-local module `PATH` cannot prevent the worker from
checking the frozen commit before its model forward.

After the formal worker pools are complete, run the fresh verifier without
printing or aggregating partial outcome values:

```bash
PYTHONPATH=src python scripts/loopscope/verify_phase7_gate_e_panel.py \
  preoutcome --run-root <formal-root>
```

Only a `PASS` pre-outcome receipt permits the single combined paired analysis
and its separate fresh-process verifier.  Gate E does not re-score V3, add a
window, produce plots, or update the final Chinese report.

## Gate F V3 point-rank Top-2 outcome panel

Gate F reuses the verified Gate E no-loop baselines and executes only 24 new
width-three loop cells: the two frozen model-local V3.1 point-rank windows for
each model, crossed with `k={2,3}` and `cache={first,last}`.  Its card is
`configs/loopscope/phase7_gate_f_outcome_card.json`; the three retained
baselines plus those new cells form the 27-cell analysis panel.  Before the
single analysis/verifier pass, use the Gate F entrypoints and the same
pre-outcome completeness barrier as Gate E:

```bash
PYTHONPATH=src python scripts/loopscope/run_phase7_gate_f_panel.py prepare \
  --mode formal --run-root <fresh-formal-root> --expected-commit <commit> \
  --cache-dir "$HF_DATASETS_CACHE"

PYTHONPATH=src python scripts/loopscope/verify_phase7_gate_f_panel.py \
  preoutcome --run-root <formal-root>
```

The Gate F card is post-terminal exploratory.  It must not rerun a baseline,
alter the frozen point-rank panel, inspect partial outcomes, or create plots.
