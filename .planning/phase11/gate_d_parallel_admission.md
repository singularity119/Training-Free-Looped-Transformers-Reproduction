# Gate D parallel operational admission

2026-10-09 latest local control and D handoff permit ARC/GPQA parallel with shared max8A800, first-round reserved<=16GPUh, total128actual+reserved. Same exact executor, scientific recipe and immutable compute f9845ea86e3deaae4145fd383d1f316a70d7dfe3; scheduling-only authorization change does not require re-running ARC debug.

ARC debug12954680 COMPLETED0:0,61GPU seconds; both Native/K3-current_t-.9 workers processed original synthetic indices0,1 with one model load. Both RAW_COMPLETE/count2 and ATTEMPT_RAW_VERIFIED, producer and verifier exits0, target_gold_loaded=false. Device sampler valid. Monitor paused before next submissions.

ARC retainable trial-p2 job12954694:1A800, emergency_gpu/emergency_gpu,8CPU/128G/30min,2 independent workers x24 records, long inputs interleaved by frozen length for same K3-current_t-.9 arm. GPQA debug12954695:1A40,debug/debuglimit,8CPU/96G/29min,2 workers x2 synthetic questions,3 generated tokens to verify incremental decode. Source/launcher/pinned environment match ARC debug and subsequent formal producer. GPQA formal still waits its own debug; neither task waits the other task's completion.

Actual D GPUh=61/3600=0.016944444; current requested reservation=(1800+1740)/3600=0.983333333; total1.000277778<16 and128. Snapshot ledger002 is canonical remote safety record. The same monitor is now ACTIVE/full60min across both new IDs, including PENDING. No duplicate automation, no cancellation, no gold/outcome read, no new MMLU-Pro work.
