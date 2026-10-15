# Phase 11 D-6 GPQA prompt and damping scope controls

Authorization and exact identities live in
`.planning/phase11/loopscope_phase11_gate_d6_handoff.md` and the Phase 11 control.
The twelve fresh cells use the same 448 GPQA identities, original option order,
input seed 20261008, and runtime seed 20261015. Historical outputs are read-only.

`phase11_d6.py --prepare-inputs --original-pool POOL --output-dir NEWDIR --commit COMMIT`
creates the CoT/direct pools with the cached tokenizer. CoT rows remain identical;
direct replaces only the exact frozen instruction. All other rendered text,
including the historical `Answer: Let's think step by step.` prefix, remains.
This tests that instruction change; it does not establish absence of internal reasoning.
Use `--synthetic` for the separate gold-free engineering pool.

`build_phase11_d6_plan.py` freezes twelve cells and eight disjoint canonical
shards. Each child process owns its model, direction state, and output root;
bounded batch packing shares only the allocated GPU. `phase11_d6.sbatch` runs
the plan with the existing pinned Python. Debug uses synthetic three-token
outputs and then `check_phase11_d6_gpu.py` to verify identical prefill logits,
actual K Loop calls, first-KV cache growth, and zero decode fits.

`damping_scope=prefill_only` suppresses only decode residual damping. The
ordinary Loop continues with `decode_mode=full` and `cache_strategy=first`.
The first output-token logits still reflect the damped prefill.
The default scope is `full`; historical configurations retain that behavior.

Each completed question is atomically checkpointed. A restart validates and
retains successful records, quarantines incomplete serialization, and performs
only pending work. An execution attempt never overwrites a successful output.

After all 5,376 records and runtime facts close, freeze S2 predictions with
`analyze_phase11_d6.py --close --plan PLAN --output-dir CLOSURE`.
Only then run `--closure CLOSURE/closure.json --gold GOLD --output-dir ANALYSIS`.
The analysis rereads and verifies all frozen predictions before opening gold.
S2 is unchanged. The primary family has four tests; the auxiliary family ten.
Both use Holm independently; paired bootstrap CIs are nominal. The prompt
interaction uses 100,000 sign flips and an exploratory exchangeability assumption.
All comparisons use the full 448-item denominator. Same-question selection of
the methods makes this an exploratory mechanism test, not independent confirmation.

`observe_phase11_d6.py --jobset JOBSET` makes one bounded read-only observation.
The single executor heartbeat owns long waits and exact terminal self-wake.
The executor reports evidence; planning alone accepts the Gate.
