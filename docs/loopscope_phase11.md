# Phase 11 generation adapters

Authority: `.planning/phase11/loopscope_phase11_contract_v1.md` and the live Gate control. Gate A implements CPU preparation and adapters. The commands below describe later authorized work; they do not authorize scheduling.

## Frozen recipe

Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554, BF16, inclusive15:18, K2/K3, block damped Euler, cache first, full decode, batch1. Each task has 18 independent configurations and 21 display rows; K2 lag1 is a display alias pending real equivalence checks. Strengths are 0.1/0.5/0.9. Directions use each configuration's own prefill, FP32 exact uncentered rank1 SVD, then remain frozen during single-token cached decode. The bank is released after every question. ARC scores full candidates and intervenes only at the common prompt end; it does not generate CoT.

MMLU-Pro uses five native first_n validation demonstrations of the same category. GPQA uses zero demonstrations, one seeded option-shuffle RNG in canonical source order, and a separate sealed mapping. Generation is greedy with native EOS/EOT and a 2048-token cap. Only exactly one valid standalone `Final answer: (X)` line is accepted. Extraction failures remain in the denominator; a valid truncated answer still counts. Runtime failures invalidate the attempt.

## Inputs and source binding

`data_binding.json` contains metadata and explicit existing source paths. Builders load only cached tokenizers. ARC/MMLU target gold is projected away at Arrow read. The isolated GPQA builder necessarily reads answer-role fields to shuffle options and seal the mapping; model pools contain no roles or gold. Real prompts, CoT, raw outputs and gold stay outside Git.

GPQA acquisition is bound to the author original GitHub archive at commit56686c06f5e19865c153de0fdb11be3890014df7, member`dataset/gpqa_main.csv`, logical split`main_csv`, canonical CSV row order. Its distinct source kind is`author_github_archive`; this commit is not an HF revision. Planning accepted this official source in`loopscope_phase11_gpqa_source_binding.md`. Content and order equality against the inaccessible HF candidate are unverified. The builder uses only the five specified current question/answer fields, excludes revision/extra explanation/validator fields, and writes the shuffled mapping separately.

```bash
PYTHONPATH=src python scripts/loopscope/build_phase11_inputs.py \
  --binding-file .planning/phase11/data_binding.json \
  --task mmlu_pro --output-dir <fresh-remote-input-dir>
PYTHONPATH=src python scripts/loopscope/build_phase11_engineering_inputs.py \
  --output-dir <fresh-remote-synthetic-dir>
PYTHONPATH=src python scripts/loopscope/build_phase11_manifest.py \
  --dataset mmlu_pro --scope PREFLIGHT_ONLY \
  --pool <synthetic-dir>/mmlu_pro-pool.json --output <fresh-manifest.json>
```

Use the pinned remote Python recorded in Gate A evidence. The old remote source checkout is read-only and differs from the new local implementation. Gate A uses an explicit CPU staging overlay. Later GPU jobs must use a separately authorized source preparation whose actual HEAD matches the supplied audited commit; do not point GPU commands at the old source or bypass the check.

## Gate B synthetic checks

On a separately authorized debug allocation of less than 30 minutes, one GPU and at most eight CPUs:

```bash
bash scripts/loopscope/phase11_preflight.sbatch <audited-source> <pinned-python> \
  check_phase11_gpu.py --run-root <fresh-gpu-check-root> --commit <audited-commit> \
  --config configs/loopscope/phase11_panel.json
bash scripts/loopscope/phase11_preflight.sbatch <audited-source> <pinned-python> \
  run_phase11_accuracy.py --dataset mmlu_pro --manifest <synthetic-manifest.json> \
  --pool <synthetic-dir>/mmlu_pro-pool.json --cell-id mmlu_pro-q4-w15-18-k3-online-current-t-lambda0.5 \
  --run-root <fresh-acquisition-root> --commit <audited-commit>
PYTHONPATH=src python scripts/loopscope/verify_phase11_scores.py \
  --manifest <synthetic-manifest.json> --pool <synthetic-dir>/mmlu_pro-pool.json \
  --scope PREFLIGHT_ONLY --attempt <fresh-acquisition-root> --output-dir <fresh-verification-root>
```

The launcher fixes three generated tokens only for synthetic engineering checks, forcing prefill plus at least two decode forwards. It is distinct from the formal 2048-token recipe. Run ARC through the same acquisition/verification path, and use the checker for Native interface, Loop versus lambda0, K2 alias, K3 direction freezing, all-layer cache growth, K body calls and one stash pass. Outputs store scalar comparisons and synthetic IDs, never hidden/logit/direction tensors. These checks do not establish formal packing or memory needs.

Pending real CUDA checks: numerical equivalence, exact SVD on the installed CUDA runtime, incremental cache behavior and measured throughput/memory at representative prompt/decode lengths. Gate A fake-tensor tests establish control flow only.

Gate B resource measurements use `phase11_resources.sbatch` with one selected
`--dataset {mmlu_pro,gpqa_main,arc_challenge}` and
`--family {Native,LoopK3,OnlineK3}` per independent process. `OnlineK3` is
current_t with strength 0.9. Synthetic generation inputs have 2860/2819 ordinary
tokens and force 2048 new tokens; the 2048 actual forwards comprise one prefill
and 2047 incremental decodes, so the final KV length is prompt+2047. This
forced-length engineering control does not alter the formal greedy EOS recipe.
ARC measures four complete synthetic candidates with 1242 prompt and 46
continuation tokens, an upper envelope rather than a real test question.
Scalar artifacts contain load/prefill/decode memory, synchronized timings,
throughput, GPU/allocation identity and every layer's cache length. They cannot
size packing on a different GPU type without direct evidence there.

```bash
sbatch <authorized-debug-resource-flags> scripts/loopscope/phase11_resources.sbatch \
  <audited-source> <pinned-python> --dataset mmlu_pro --family OnlineK3 \
  --run-root <fresh-resource-root> --commit <audited-commit>
```

## Formal closure and analysis

Prepare a `FORMAL_TEST` manifest from the accepted bound config and complete pool. Each acquisition runs one independent cell, optionally disjoint modulo shards via `--shard INDEX/COUNT`. Every saved attempt includes command metadata, canonical raw records, telemetry and a completion summary. Fresh paths are mandatory. There is no retry/recovery answer generation; preserve failed attempts and create a fresh authorized attempt.

```bash
PYTHONPATH=src python scripts/loopscope/run_phase11_accuracy.py \
  --dataset mmlu_pro --scope FORMAL_TEST --manifest <formal-manifest.json> \
  --pool <formal-pool.json> --cell-id <independent-cell-id> \
  --run-root <fresh-run-root> --commit <audited-commit>
PYTHONPATH=src python scripts/loopscope/verify_phase11_scores.py \
  --manifest <formal-manifest.json> --cell-roots <cell_id-to-root-lists.json> \
  --output-dir <fresh-closure-root>
PYTHONPATH=src python scripts/loopscope/analyze_phase11.py \
  --manifest <formal-manifest.json> --cell-roots <cell_id-to-root-lists.json> \
  --gold <sealed-identity-gold.jsonl> --output-dir <fresh-analysis-root>
```

Gold sidecars use exact canonical identity plus `label_index` and optional category; GPQA's sealed `gold_letter` mapping is accepted directly. ARC label indices must follow the original actual choice labels; MMLU uses the official `answer_index`. Source-to-gold conversion is permitted only after full task closure under the later Gate's authority. The analyzer independently re-closes all 18 complete cells before its first gold read. Missing/duplicate/out-of-order identities, overlapping shards or runtime failures reject closure.

All 18 results and 21 display rows are published. Per-task exact two-sided McNemar/Holm families have sizes30/12/2; paired bootstrap10000/seed20261008 yields nominal95%CI, category-stratified for MMLU-Pro and item-level otherwise. No repeated test of aliases. The panel is exploratory, has no matched-norm control and does not establish direction specificity or transfer. All-task acquisition is245736 records, including224640 generations; no GPU cost estimate is implied by that count.
