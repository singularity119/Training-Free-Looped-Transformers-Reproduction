# Phase11 Gate A implementation evidence

Executor: `01a11ac8-ca15-7573-8ff6-c1dc0892faab`, host`local`, title`Gate A-execute-LoopScope-Generation-Adapters-第11阶段`. Planning: `01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f`, host`local`. Only Gate A is authorized. Executor does not decide PASS.

## Source and permissions

Local real repository is the nested `loopscope-tflt`, branch`loopscope`, starting HEAD`81f873b38942474a012e2648db3e81d3085abdd6`. Fixed ancestor`4f59bd93eca4da3cbf458a93508f91c5b23912bc` check exit0. Initial dirty state contained only the allowed Phase11 planning directory. Branch/HEAD/root/ancestor were rechecked before grouped writes, remote preparation and final Git operations. Protected wrapper/strategies/cache/config paths and Phase10 control remain unchanged.

HPC2 SSH used `BatchMode=yes`, `ConnectTimeout=10`, `ClearAllForwardings=yes`. Read-only old source remains clean branch`phase9-gate-c-20260922` at`a2c9e58a32163e1ed1527d3ab57798bc7378f486`. No remote source Git mutation or environment installation. New CPU writes are confined to dedicated Phase11 input/staging roots in `remote_preparation.json`. Local staging overlays copy only small source/config metadata files; no local model/data downloads or model forwards. Actual CUDA, Slurm submission, formal generation, model outcomes and real gold scoring were not performed.

Pinned remote Python: `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python`. Installed torch2.3.1+cu121, transformers4.51.3, datasets5.0.0 and lm_eval0.4.11. Qwen3-4B-Instruct cached snapshot`cdbee75f17c01a7cc42f958dc650907174af0554`: 36 decoder layers, hidden2560, model context262144, native EOS/EOT `[151645,151643]`. Tokenizer is loaded local-only. Native generation config's sampling setting is overridden by the frozen greedy recipe.

## Implemented increment

- `phase11_panel.py`/config and manifest entry: 18 independent cells per task, 54 total, 21 display rows per task with three K2 aliases; 245736 sample records, including224640 generations. Aliases still require real Gate B equivalence.
- `phase11_data.py`/builders: exact source binding, gold-free target projection, native ARC25 train demonstrations with seed20261002, MMLU five category validation first_n demonstrations, native chat rendering, full token lengths, isolated GPQA shuffle and sealed mapping. Synthetic engineering pools are separately marked PREFLIGHT_ONLY.
- `phase11_runtime.py`/adapter: each Online configuration fits its own prefill bank; ordered K callbacks reset every decode forward; decode fits no direction and changes only the current token. Bank released per question. ARC common-prefix fitting and prompt-end intervention preserve full candidates.
- Acquisition entry: batch1 Native/Loop/Online, allocated CUDA only, actual loopscope HEAD check before weights, greedy2048/native EOS, strict extraction, truncation and full denominator records; failed attempts preserve raw failure/FAILED summary and cannot close. Formal modulo shards preserve canonical indices.
- Closure/analysis: all18 complete task cells with one source commit, exact identities/candidate order/categories, no overlapping/missing shards. Analysis freshly closes before reading gold. Frozen exact two-sided McNemar/Holm families30/12/2; paired bootstrap10000/seed20261008 and nominal95%CI, MMLU category strata. No actual gold analysis was run.
- Gate B checker and debug launcher: synthetic prefill plus at least two incremental decode steps, K body calls/one stash, all-layer cache growth, Native interface/Loop-lambda0/K2 alias/K3 direction bank checks. Source HEAD must match audited commit. Scalar evidence only; no hidden/logit/direction tensors saved.

## Actual input preparation

Metadata-only artifacts: `data_binding.json`, `input_token_facts.json`, `remote_preparation.json`. Real prompts and mappings remain exclusively remote. Source raw ARC files and MMLU cache were reused read-only; fresh CPU builders used the instruct tokenizer rather than historical Phase10 length facts.

| task | source population | max prompt/context tokens | max continuation | max actual input | truncated |
|---|---:|---:|---:|---:|---:|
| ARC-Challenge |1172|1242|46|1243|0|
| MMLU-Pro |12032|2860|n/a|2860|0|
| GPQA-Main (author archive) |448|2819|n/a|2819|0|

ARC candidate-prefix inconsistency count0. MMLU min prompt866; prompt+2048 cap max4908 fits262144. MMLU validation70 rows in14 categories, five canonical indices each; exact demonstration identities are in token facts. ARC train1119/test1172/validation299, gold projected at read for targets; only train labels allowed for demos. Builders completed exit0. These are renderer/tokenizer facts and do not establish GPU accuracy or resource packing.

GPQA official source inspection and accepted binding are distinct. HF public metadata returned200/gatedauto; unauth exact-file HEAD401; remote HF API reset had no HTTPstatus, so the remote request is a visibility failure. Allowed fixed-root source search found no prior raw GPQA. The author README exposes its original official archive as an alternative distribution. Fixed GitHub commit`56686c06f5e19865c153de0fdb11be3890014df7`, archive2348038bytes, remote HEAD200/download succeeded; one `dataset/gpqa_main.csv` member,448 rows and required current fields. Candidate inspection metadata remains unchanged. Planning separately accepted this author source in `loopscope_phase11_gpqa_source_binding.md`; logical split`main_csv`, explicit `author_github_archive` provenance and canonical CSV order. Equality with inaccessible HF content/order is unverified. No mirror, credentials, login or terms acceptance was used. GPQA final builder completed exit0, min/max prompt89/2819 and max prompt+2048=4867, no truncation. Canonical identities run from archive index0 to447; shuffle uses one RNG(seed20261008) and sealed mapping is inaccessible to acquisition/closure until task-level release. The builder has limited authorized answer-role access; `target_gold_loaded=false` describes the model pool, not a claim that this isolated builder never touches labels. All three complete formal pools and all three two-item synthetic pools passed the final remote `load_pool` checks (exit0), recorded in `pool_validation.json`. Runtime Python3.11.9 and exact resolved staging module paths are retained there. No model weights were loaded or gold scored.

## Checks and boundary

Before accepted GPQA source adaptation: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase11*.py'` ran44 tests, exit0; all seven new Python entries `--help` and seven `--dry-run` paths exit0; `bash -n scripts/loopscope/phase11_preflight.sbatch`, Phase11 `py_compile` and `git diff --check` exit0. Exact CLI metadata/commands are retained in `cli_verification.json`; dry runs used only synthetic local metadata and never loaded weights or gold. Final source/report adaptation: the same targeted command ran45 tests, exit0; all seven help entries and the three changed source/config dry-run paths were repeated exit0. Final `git diff --check` and shell syntax exit0. No historical suites were run.

Remaining Gate B facts: true BF16/CUDA numerics, exact SVD/backend behavior, all-layer incremental cache, actual K2 alias equality, ARC full scores and representative GPU memory/throughput. No fake test is claimed as real model equivalence. No Gate B authorization, GPU job or automation was created by this executor.

## Ready paths and final delivery

`remote_ready_paths.json` binds all three formal and three synthetic manifest/pool pairs. Remote CPU manifest creation completed exit0; manifests contain only frozen membership and paths, and authorize no acquisition. Data/artifact root is`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z`; final CPU source overlay is`staging/phase11-gate-a-20261008T095224Z`. The CPU overlay is an uncommitted Gate A snapshot paired with the preserved read-only old source, and is not an audited GPU source checkout. Later jobs must use the final implementation commit under their own Gate authorization.

`docs/loopscope_phase11.md` and`phase11_preflight.sbatch` provide exact later commands, synthetic inputs and remaining real CUDA checks. The task analysis report carries the dataset recipe, including the distinct author source; GPQA reports explicitly retain the unverified HF-equivalence boundary.

Allowed final Git operation: stage only this Phase11 package, single-purpose implementation commit and normal push`origin/loopscope`. Exact resulting commit, push verification and dirty state are provided in the unique`GATE_A_FINAL_AUDIT` tool-delivered to planning, avoiding a self-referential commit field inside this file. Planning owns acceptance and all Gate crossing.
