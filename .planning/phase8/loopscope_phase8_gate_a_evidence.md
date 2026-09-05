# Phase 8 Gate A executor evidence

Executor: 01a07030-6413-7660-b712-c18d9e93d26e. Planning recipient:
01a06fe9-c7bb-7d72-a906-234f301de317. Date: 2026-09-05.
This is engineering evidence requesting planning acceptance, not a self-issued PASS.

## Local identity and scope

Dedicated `loopscope-tflt` checkout, branch `loopscope`, protected ancestor
4f59bd93eca4da3cbf458a93508f91c5b23912bc verified (exit 0).
Initial clean HEAD 072d89a28b0cb03cd21bec01b127fb7bf5818751, implementation parent
b53ff067992e41b66a8df01794dfb106c61b29d0. Initial planning diff contains only
Phase 8 documentation and entry updates. During execution planning added
c118b86e30cd74ad295b4c04e280205106b3cf54 and the fixed-horizon supplement;
retained. Planning then added AGENTS-only 2f26c0e1548d1414ed6960f4251aed32c19ef77b, the final authorized implementation parent. The Gate skill and terminal protocol reference were read. Only these four new files belong to A:

- `src/tflt/loopscope/phase8_spectral_core.py`
- `tests/test_loopscope_phase8_spectral_core.py`
- `configs/loopscope/phase8_model_windows.json`
- `.planning/phase8/loopscope_phase8_gate_a_evidence.md`

Final implementation revision and push status are delivered in the terminal packet
(the evidence file cannot name its own containing commit in advance).

## Decisive mathematics and configuration

Pure standard-library references implement unit-direction rank-1 projection,
soft damping and per-vector matched-norm uniform scaling, including explicit zero
residual behavior and finite/unit checks. For unit v, a=v^T delta:
`||delta'||^2 = ||delta||^2 - (2 lambda-lambda^2) a^2`;
parallel energy scales by `(1-lambda)^2`, orthogonal component stays unchanged.
Uniform uses `||delta'||/||delta||` on the same original vector, not a global alpha.

Six tests cover lambda 0/1/0.5, orthogonal preservation, sign invariance,
matched norm/direction/zero and fully parallel removal, invalid inputs, all four
exact inclusive/boundary/cache mappings and model layer bounds.
Both models/all K freeze alpha=1, h=1/K, horizon=1. K set, split, SFA confirmation,
basis-fit K and application K remain pending; no fixed-h prefix reuse is valid.

## Live read-only HPC evidence

SSH alias `hpc2-hkustgz`, user xhuang225, observed login mgmt-6;
all calls used BatchMode=yes, ConnectTimeout=10, ClearAllForwardings=yes.
Dedicated remote checkout `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`
is clean on loopscope at 08da7733a1e8402677184c860d0ec35f041335aa.
It has NOT been synchronized. Gate B must authorize synchronization and recheck.
Existing interpreter `<checkout>/.venv-loopscope-cu121-20260711/bin/python`:
Python 3.11.9, torch 2.3.1+cu121, transformers 4.51.3, lm_eval 0.4.11
(read using importlib.metadata, exit 0, no torch/model forward).

Cache root `/hpc2hdd/home/xhuang225/shared/hf_home/hub`:

| Model snapshot subpath | layers | hidden | config dtype |
| --- | --- | --- | --- |
| models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 | 36 | 2560 | bfloat16 |
| models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1 | 28 | 2048 | bfloat16 |

Both config.json and tokenizer_config.json read successfully; vocab=151936,
Qwen2Tokenizer, eos=<|endoftext|>, no explicit padding_side. These checks establish
config availability, not a full weight-load test. Four windows fit these layer counts.

4B historical runtime BF16 is live-verified from
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-c-repair-identity-20260722T072256Z/formal/cell-05-blind_low1_8_11/eval/command_args.json`
(and cell-06-blind_low2_9_12): exact 4B revision, bfloat16, mmlu, five shots.
1.7B historical runtime FP16 is supported by versioned Phase 1 control line 31;
canonical historic root is
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022`.
Its existence and parameter-file names were checked; file contents/results were not read.
This directory-only check reached the historic root outside the handoff's enumerated
new workspace roots while following its exact-historical-path requirement; no further
reads there were performed. Thus 1.7B historical dtype remains documentary evidence,
not newly reread runtime parameters. Do not infer FP16 from the model's BF16 config.

Initial bare squeue/sinfo failed (exit 127, binary absent from PATH). Resolved using
`timeout 8s /opt/slurm/bin/squeue -u xhuang225 -o "%.18i %.9P %.30j %.2t"`
and `timeout 8s /opt/slurm/bin/sinfo -p debug -o "%P %a %l %G"` (exit 0):
no user jobs; debug up, 30:00 maximum, GPU entry gpu:a40:8(S:0).
Gate B may request a <30-minute debug GPU job after its authorization; availability
must be refreshed then. No Slurm submissions/cancellations or remote writes occurred.

## Minimum proposed Gate B integration (not implemented)

Read local `strategies.py` run_loop/_damped_euler, `wrapper.py`
LoopBlockEntryWrapper.forward/_audit_tensor_diff and eval_runner.py HFLM path.
Current update is `x = x + (alpha/k) * (operator(x)-x)`; audit callbacks discard returns.
Body calls snapshot/crop cache, then a separate stash pass runs on original input
(cache first) or looped input (cache last), replacing returned hidden with looped.
Stash is not another t step and must not fit/apply the intervention.

Smallest proposed protected diff: `config.py` adds an optional residual-transform
callback default None; `strategies.py` branches only when callback is present and
Euler is selected, computes the original native-dtype residual once, invokes the
callback with t and residual, then applies the unchanged h update. When None, retain
the existing run_loop and _damped_euler path literally, including K=1 shortcut.
Reject unsupported opt-in strategies. New opt-in adapter outside protected runtime
holds basis, exact pre-answer position and t>=1 rule; it returns a residual copy
with only that token altered. Do not modify the operator return to simulate an
update: subtract/add rounding would differ. Wrapper/cache need no change for the
fixed batch-one full-sequence path if token position is supplied by the scoring
adapter; its existing cache/stash semantics remain intact.

Gate B must separately specify projection arithmetic dtype/cast placement.
`(out-x).float()` measures the actually used native residual, whereas
`out.float()-x.float()` is an FP32 measurement difference and generally differs.
Store these names separately; never silently substitute the latter into the update.

Remote evaluator source is `<venv>/lib/python3.11/site-packages/lm_eval/models/huggingface.py`:
lines 1233-1238 construct `(context+continuation)[-(max_length+1):][:-1]`;
1282 right-pads loglikelihood inputs, 1301 computes log_softmax, 1321 selects
continuation logits, 1350 gathers candidate-token probabilities (summed afterward).
The MMLU source `tasks/mmlu/default/_default_template_yaml` uses cais/mmlu,
test split, dev first_n fewshots, plain A/B/C/D prompt ending `Answer:`, and
multiple_choice acc. Reading YAML reads schema, not target labels.
Do not use generation's left-padding default to locate the scoring token.

For one-token choices, the scored pre-answer token is the final nonpadding input
position. For multi-token choices it is the last context token, NOT the last token
of context+continuation[:-1]. The adapter must use exact evaluator tokenization,
left-truncation offsets and right-padding lengths, and verify the four continuations
are single-token for these frozen prompts before using a simpler batch-one index.
Preserve HFLM choice-loglikelihood scoring; Gate B must test prompt/choice alignment
and disabled-path equality. Cache first/last may not affect full-sequence scores;
measure/report that instead of switching to cached generation.

## Commands and validation

From repository root:
`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loopscope_phase8_spectral_core.py'`
passed, 6 tests, exit 0. Two invocations accidentally used the outer workspace
and failed with tests not importable (exit 1 each); corrected working directory, no
code workaround. `git diff --check` passed (exit 0); final scope/status contains exactly the four authorized new files. Final commit/push results accompany terminal delivery. No full suite,
CLI/framework expansion, fitted basis, model forward, target outcome/labels,
installation/download, GPU use, heartbeat, or Gate B execution.
