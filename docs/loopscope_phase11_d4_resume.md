# D-4 consolidated acquisition

The 2026-10-13 user amendment retains the highest-priority `emergency_gpu`
partition and a total maximum of eight A800 GPUs. Existing jobs stay in place.
The executor verified submitted worklists against remote submission receipts:
5499 unique pairs were already submitted, leaving exactly 2565 of the frozen
448 × 18 GPQA panel. The unused small-batch plan is retained as history and must
not be submitted.

Producer commit `ba218fc60cc7eb3bd0e45a7e30002549b4404357` adds persistence around
the existing `run_phase11_accuracy.acquire_row` and `load_runtime` functions.
The model, prompt, greedy native EOS, 8192 token limit, batch size, cache and
direction fitting code remain unchanged. Eight tasks each run five independent
processes, with disjoint per-cell shards and no shared model or tensor state.
Each task requests one A800, eight CPUs and 96 GiB, with the legal partition
maximum of fourteen days. QoS limits all retained and new jobs together to
eight GPUs. Actual runtime and failures remain accounted separately from that
requested upper bound.

Each question commits its complete record and telemetry to
`complete/<canonical_index>.json` through a flushed, fsynced temporary file and
atomic replacement, before updating `checkpoint.json`. Restart validates these
bundles against the same cell, input identity and lifecycle, then reconstructs
done/pending state. Valid successes are never overwritten. Uncommitted or
incomplete serialization is moved to `quarantine/` with a reason; scientific
contract errors raise. Interrupted questions are recomputed at a question
boundary; no token-level KV recovery is introduced. A single-writer lock
prevents two attempts from writing the same root concurrently.

Complete shards materialize the conventional `records.jsonl`, `telemetry.jsonl`
and complete `summary.json`, preserving compatibility with full-panel closure.
Every process and parent attempt gets a separate attempt directory. Slurm logs
append across restarts. Parent progress contains counts only; the observer reads
scheduler, checkpoint and summary metadata, never raw records or gold.

Five CPU interruption/recovery tests passed locally and in the pinned remote
Python. Same-path array launcher debug `12997841_0` completed with exit `0:0`
in 51 seconds, exercising Native and Online K3 acquisition and persistence.
The retained A800 packing5 resource evidence remains applicable because the
numerical model path is unchanged.

Formal array `12997851_0` through `12997851_7` covers all 2565 remaining pairs.
The replacement `cell-roots.json` contains only submitted historical roots and
the new long roots. Closure may explicitly approve the original producer
`5fc00f067773b2a20e355bf4def462781214b99f` and the above persistence producer,
after verifying their common numerical path. Full 8064 identity/cell closure
and a saved gold-unread barrier still precede S1 primary/S0 auxiliary statistics.
MMLU remains paused. Planning alone decides Gate acceptance.

The sole observer uses the exact current jobset in
`.planning/phase11/gate_d4_monitor.json`. Its 4539-record prior baseline excludes
every monitored wave220 job, including partial children, preventing double
counting. Pending Slurm arrays are queried by parent and compact accounting
rows are expanded only onto requested child IDs. Failure or all-terminal state
pauses the observer and triggers the exact executor; no work remains unsubmitted.
