# LoopScope Phase 3 producer contract

This runbook describes the local, outcome-blind contracts implemented in Gate P3-A for
`H3_BASELINE_LOCAL_REVERSAL_WINDOW_RECOVERY_V1`. It does not authorize P3-B acquisition,
dataset/model access, SSH, HPC2, GPU, Slurm, or any historical/blind/full outcome read.

## Frozen inputs and artifact roles

- `configs/loopscope/phase3_card.json` is byte-frozen at
  `5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825`.
- `phase3_source_manifest.json` is a **P3-A declarative source contract**. Its live-source
  bindings are intentionally null; it is not a fabricated validation-1531 identity manifest.
- `phase3_pool_schema.json` is the closed-world JSON Schema for one gold-free validation
  pool record.
- `phase3_trajectory_schema.json` is the closed-world JSON Schema for one selector-safe
  no-loop trajectory record.
- `phase3_card_verifier_receipt.json` binds the card, the three contract files, the Phase 3
  implementation sources, a synthetic 1,531-record/57-subject closure fixture, raw trajectory
  recomputation, and two deliberately rejected decision-critical tampers.

The JSON Schemas document interchange shapes. The standard-library Python validators in
`phase3_schema.py` and `phase3_pool.py` are authoritative for finite-number checks, exact
identity/source closure, canonical order, self-hashes, and recomputed metrics.

## Source and pool closure

The future P3-B source producer must supply exactly 1,531 unique MMLU validation identities
covering 57 subjects. Records are ordered lexicographically by:

```text
subject, task, doc_id, doc_hash
```

`doc_id` ordering is ordinary string ordering. Each identity is the normalized tuple
`(task, doc_id, lowercase doc_hash)`. A pool is accepted only when its complete source
projection—identity, subject, split, prompt hash, sanitized-content hash, and renderer
provenance—exactly equals the planning-signed source manifest. A count match alone is not
source closure.

Pool records may contain question, ordered A/B/C/D choices, and a rendered prompt so the
producer can be verified. They may not contain target gold, correctness, loop residuals,
accuracy, answer flips, or unknown fields. Those raw text fields are not copied into selector
trajectory records.

The sanitized-content digest is SHA256 over canonical JSON containing only the NFC-normalized,
whitespace-collapsed question and the four ordered choices. Validation and test metadata must
have both zero canonical-identity intersection and zero sanitized-content-hash intersection.

## Trajectory contract

Each formal primary identity produces one `native_no_loop` record with zero loop insertions and
exactly these ordered boundaries:

```text
B_0, B_1, ..., B_28
```

`B_0` is the embedding output and `B_j` is the output after decoder layer `j`. For each boundary,
the producer stores only the four-choice probability distribution and recomputable scalar fields:

```text
choice_entropy
kl_to_final
cross_entropy_to_final
top1_to_final_agreement
```

Raw logits are producer-only inputs and are removed before the selector artifact is emitted.
Probabilities are computed by a stable float64 four-choice softmax. They must be finite, strictly
positive, and normalized within `1e-8`; values are never clamped or silently renormalized.
Validators recompute all scalar fields within absolute and relative tolerance `1e-10`, including
`C = H + D`, `D(B_28)=0`, and `C(B_28)=H(B_28)`. Top-1 ties use A, then B, then C, then D.

For inclusive layer window `s:s+3`, all window metrics use boundary entry `B_s` and exit
`B_(s+4)`. For example, `12:15` uses `B_12` and `B_16`.

## Baseline-only analysis

Per sample and width-4 start:

```text
E_i(s)      = H_i(B_s) - H_i(B_(s+4))
K_i(s)      = D_i(B_(s+4)) - D_i(B_s)
CEdrop_i(s) = E_i(s) - K_i(s)
```

Point estimates are arithmetic means over all 1,531 records. A single
`random.Random(20260716)` stream generates 2,000 subject-stratified bootstrap replicates in
replicate-major, canonical-subject, draw-major order. The identical index map is reused for all
windows, metrics, contrasts, and variants. Standard errors use `ddof=1`; percentile endpoints
linearly interpolate at `q*(R-1)`.

The analyzer publishes SHIFT, FLANK, and CONSENSUS independently. It does not choose a variant
or consume outcomes in P3-A. Scorability, point eligibility, ranking, and the per-variant window
decision remain separate:

- SHIFT supports starts `1..23` and compares `s-1,s+1`.
- FLANK supports starts `4..20` and compares `s-4,s+4`.
- CONSENSUS supports starts `4..20` and requires both comparison families.
- Support-exterior starts are `UNSCORABLE_ABSTAIN`; no zero fill or interpolation is allowed.
- All supported starts receive a score/rank even when point-ineligible.
- Eligibility uses strict signs and strictly positive contrast CI lower bounds; equality at zero
  fails.
- Selection frequency uses the fixed point-eligible candidate set and original full-sample SE
  denominators. Point ties use `1e-12` relative/absolute tolerance; replicate ties use ascending
  start. Frequency exactly `0.80` passes, while values below it abstain.

Historical, blind, and deployment scopes only filter the same frozen scores/bootstrap evidence;
they do not rerun the bootstrap. P3-A performs no historical recovery or variant choice.
Production orchestration enters through `analyze_manifested_phase3_trajectories`, which closes
every trajectory against the signed source and pool manifests before computing any statistic.

## Write once versus verify

The prepare script has an explicit one-time materialization mode:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/loopscope/prepare_qwen17_phase3.py \
  write --card configs/loopscope/phase3_card.json
```

It refuses to replace an existing contract artifact. Routine verification is read-only:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/loopscope/prepare_qwen17_phase3.py \
  verify --card configs/loopscope/phase3_card.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tflt.loopscope.phase3_verifier \
  --card configs/loopscope/phase3_card.json \
  --receipt configs/loopscope/phase3_card_verifier_receipt.json
```

The verifier contains no timestamp, absolute path, Git state, Python version, or other dynamic
receipt field. It reconstructs the synthetic population, raw trajectory, baseline-only analysis,
and implementation hashes in memory and exact-compares the committed receipt.

## Local P3-A validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests \
  -p 'test_loopscope_phase3*.py'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tflt.cli --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tflt.loopscope.phase3_verifier \
  --card configs/loopscope/phase3_card.json \
  --receipt configs/loopscope/phase3_card_verifier_receipt.json
git diff --check
```

The repository's accepted pre-existing full-suite deviation is the Phase 2 immutable hash mismatch
for `phase2_analysis.py`. P3-A must not modify that historical card or source and must introduce no
additional failure.

## P3-B admission boundary

After an independent P3-A audit returns `PASS`, a new P3-B executor may instantiate these schemas
with a planning-signed renderer/source manifest and a write-once validation pool. It should not
need to change identity, metric, support, bootstrap, eligibility, ranking, or abstention semantics.
Actual renderer/tokenizer provenance and all 1,531 identities are bound only in P3-B artifacts.
P3-A's synthetic receipt is never scientific evidence and must never be mixed into the primary
population.
