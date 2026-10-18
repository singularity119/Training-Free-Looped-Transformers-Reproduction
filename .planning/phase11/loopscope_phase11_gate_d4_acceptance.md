# Gate D-4 planning acceptance — PASS

Date: 2026-10-14 (Asia/Shanghai).
Decision: PASS — full GPQA8192 acquisition and frozen S2 analysis delivered; not a claim of significant method benefit or whole-phase terminal acceptance.
Planning: 01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local.
Audited executor: 01a12f40-cbc9-7790-ac6e-f3fe1c454ab0/local.
Audited delivery commit: 7e404bcf4b89e45e9a13642035be87e2c37d2b2a; subsequent receipt-only commit 462539bfde14d0bc66d175c7f3d92280c65902a3. Analysis source f40cd50247dae1d4c196e9252acb43c6119e70e9, remote clean loopscope checkout. Producer sources 5fc00f067773b2a20e355bf4def462781214b99f and ba218fc60cc7eb3bd0e45a7e30002549b4404357. Baseline 4f59bd93eca4da3cbf458a93508f91c5b23912bc remains an ancestor.

## Three decisive checks

1. **Full membership and frozen method.** A fresh read-only close_raw_panel on the immutable final roots verified all 448 canonical identities × 18 cells = 8064 records, exact no-overlap/no-missing membership, approved source set, input/cell bindings, generated-token/native-EOS-or-8192 consistency, cache growth, prefill/decode counts, Online decode_fit_count=0 and direction release. Final saved S2 predictions were independently reproduced from raw text for all 8064 records with the frozen whole-line syntax. No model/GPU execution or remote writes occurred.
2. **Order and statistics.** Original raw closure 2026-10-13T14:20:20.384576Z, S2 specification 14:20:20.413150Z, all predictions barrier 14:20:25.646889Z, final gold-read 14:20:25.741138Z are in required order. Earlier user-authorized interim scoring is a retained historical exception; this is not claimed to be the project's first gold access. Planning then independently recounted all 18 S2 cells and all 44 paired contingency tables from the saved predictions and canonical gold, recomputed exact two-sided McNemar with SciPy, original 30/12/2 Holm families, and 10000-draw seed20261008 paired-bootstrap intervals. Maximum exact-p discrepancy 4.43e-14; CI discrepancy 0 pp. All counts/deltas agree; 0 Holm rejections, minimum adjusted p=0.0658472672.
3. **Scheduler and resource settlement.** Fresh exact-job sacct and squeue checks verified all 290 ledger jobs (288 formal + 2 debug) COMPLETED/0:0 with matching allocations, times, partition/QoS and no inflight Gate job. Recomputed formal204.685833 GPUh, debug0.020833 GPUh, total204.706667 GPUh; formal allocation peak8 A800. Observer automation.toml is PAUSED, terminal self-resume and executor-to-planning receipt are confirmed. The 14-day TimeLimit was an upper bound, not actual duration.

Machine evidence: gate_d4_planning_verification_20261014.json; reproducible read-only checker gate_d4_planning_verify_20261014.py. Existing final analysis, execution evidence, job ledger and delivery receipt remain intact. One initial audit invocation reached the scheduler check after data/statistics verification but failed because sacct was absent from the non-login PATH; the checker was rerun with the observed /opt/slurm/bin binaries and passed. No experiment output was changed.

## Interpretation and accepted measurement boundary

S2 ties: K2 current_t lambda0.5 and K3 fixed_t0 lambda0.9 both244/448=54.46%; Native225/448=50.22%, LoopK2 226/448=50.45%, LoopK3 215/448=47.99%. All44 Holm comparisons fail alpha0.05. Length and parser were selected after diagnostics; these remain exploratory results. No matched-norm control or direction-specific causal conclusion.

S2 is a frozen syntactic rule, not semantic final-intent adjudication. Full-population boundary screening found 0 accepted lines in Markdown fences, 64 accepted records with nondecorative text after the last legal line, and 1 with a later different-label Final-answer surface ignored by the whole-line grammar. The historical 7032-record manual audit identified that actual final-intent mismatch. These limits are disclosed; they do not violate the user-frozen S2 metric, and this acceptance does not modify any prediction or authorize another parser. 83 length-limit records have accepted answers under the same rule; all448 records/cell stay in the denominator.

## Disposition and next admission

D4 execution/GPU/scheduler/gold-analysis authority is closed. Monitor remains PAUSED and must not resume the closed panel. No further raw acquisition, rescoring or new Gate is authorized by this decision. MMLU-Pro remains PAUSED_BY_USER_UNTIL_ARC_GPQA_COMPLETE_AND_DISCUSSION; phase-wide terminal acceptance is not claimed. Planning owns updating the Phase11 report and weekly report to full8192/S2 results, preserving interim artifacts as history. External Notion delivery status is tracked separately from scientific Gate PASS.

## Report delivery

2026-10-14: Notion authentication restored. Phase11 page 3f87d1c8d46f80f5a73bfe34487fe060 and weekly page 3f87d1c8d46f803fb171c4aa3b1ad918 updated to full 8064/448 S2 results. Read-back verified every cell in the 18-configuration accuracy table, 18-configuration output diagnostics, four key comparisons and all 44 paired comparisons; all three full aggregate attachments are present. Existing Phase11 methods/ARC table and weekly MMLU/ARC tables and task overview preserved. Old GPQA attachments remain explicitly historical. Evidence: outer workspace 资产/报告/phase11/notion_full_20261014/notion_delivery_verification.json and page readbacks.
