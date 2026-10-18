# Gate D-5 planning acceptance — PASS

2026-10-15. Planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local; executor01a1385a-fdfa-7d43-a6b8-1405a585b8b9/local. Audited delivery0b39d676c8529d31b27a765b34225ca058e79d99. This is execution/reproducibility acceptance, not independent confirmation of scientific efficacy.

Three decisive checks:
1. Fresh remote exact-job accounting: original13030531_0..7 allCOMPLETED0:0, no missing jobs; retired squeue ID accepted only with complete accounting. All224 checkpoints complete,102796 records; formal allocation114.805833GPUh. Existing224child exit0/health evidence and monitorPAUSED receipt inspected.
2. Fresh remote saved-prediction readback independently counted102796 unique family/cell/repeat/index entries over28cells, asserted actual saved seeds20261014/20261015 and cross-repeat identities/predictions equal for every entry. Frozen closure metadata and source were inspected; saved closure mtime precedes final gold receipt02:24:12Z. Existing full raw identity/input/telemetry closure reused rather than duplicated. Observer's hardcoded gold=false field was explicitly NOT used as final gold-state evidence.
3. All30 aggregate contrast deltas, exact two-sided binomial McNemar p and separate MAIN20/SECONDARY10 Holm were independently recalculated, discrepancy<1e-10. Reviewed CPUrepair diff: JSON container and exact source binding correction, no producer/parser/statistical change; first failure beforegold preserved.

Results: both new rounds have identical14cell predictions; source analysis also finds9/9likelihood scorecells and5/5GPQA token cells identical. Original reference13/14 predictioncells reproduce. Only BaseMMLU Native changes14predictions,8wrong-to-right/6right-to-wrong,10262→10264correct. Diagnostic differences all located in originalA40 512subset, remaining13530A800 scorevectors identical. Consistent with hardware numerics, not causal proof; accepted disclosed boundary, no rerun to force agreement.

MAIN20 rejects4 entries (2 comparisons repeated in2 rounds): BaseMMLU damped-Native andGPQA K3fixed0.9-Loop. GPQA old full-grid Holm.065847 vs new.039508 is changed family over identical outputs; no independent evidence gain. Other damped-Loop comparisons remain nonsignificant. Fixed inputs, greedy/likelihood and postselected candidates remain exploratory.

DecisionPASS; D5GPU/scheduler/newanalysis authority closed, monitor remainsPAUSED. MMLU-Propaused; no automatic nextGate. Human report: outer资产/报告/phase11/gate-d5-repeat-20261015/执行与复现结论.md. Machine acceptance evidence: gate_d5_planning_verification_20261015.json.
