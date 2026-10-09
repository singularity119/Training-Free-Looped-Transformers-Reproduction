# Gate D first three full batches verified and backfilled

ARCfull b000/b001/b002 (12955297/12955298/12955299) completed0:0, allocated513/1341/697seconds, each512records. Executor independently verified all12workers against exact worklist original indices, full population/manifest/cell/source, telemetry identity, Online per-question released directions and decode-fit0. Source f9845ea86e3deaae4145fd383d1f316a70d7dfe3, Python/launcher unchanged. Device peaks46064/47186/43947MiB of81920, highest allocator12046MiB per worker. Resource report gate_d_full_wave1_resource_verification.json preserves measurements without reading records/gold/outcome.

C+D retained ARC1752/21096 arm-index records (C72+D1680); GPQA144/8064 (C72+D72). These are collection counts, not accuracy or Gate acceptance. Four ARC jobs12955300–12955303 remain RUNNING at last snapshot, GPQA12955304 PENDING Priority. Preserved all five untouched.

Submitted exactly three fresh ARCfull p4 batches007/008/009 (12955541/12955542/12955543),512records each,1A800/8CPU/128G/2h emergency_gpu/emergency_gpu, same compute. Active/submitted set totals8single-GPU jobs. Full plan batches0..9 now submitted, next unsubmitted10; never repeat completed/active work.

Completed GPU allocation cost1.963055556GPUh plus observed running0.993055556 gives latest actual snapshot2.956111111GPUh; requested reservation16GPUh. Accounting now explicitly includes latest observed running allocation without double counting terminal jobs. No formal GPUh stop cap. Pending GPQA p6 still needs actual capacity/throughput evidence before full packing selection.

Unique monitor paused after terminal check, AUTOMATION_TERMINAL_RESUME delivery to exactexecutor confirmed, then retargeted ACTIVE/full60min across all eight IDs with tool confirmation. No duplicate automation, cancellation, runtime/scientific change, gold/outcome or MMLU-Pro work. Continue bounded backfill on terminal; full18arm raw identity closure required before task-level outcome analysis.
