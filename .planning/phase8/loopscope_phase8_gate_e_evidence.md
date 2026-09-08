# Gate E execution evidence — 4B K3 supplement

Executor `01a08031-a85e-7101-b150-60916abdc4ac`; planning `01a06fe9-c7bb-7d72-a906-234f301de317`.
Scope and scientific authority: current control, Gate E handoff and amendment. Original A-D evidence remains frozen. This is a post-outcome exploratory supplement, not a replacement primary analysis.

## Admission and environment

2026-09-08: local dedicated clone `loopscope`, clean starting HEAD `f97b4ae5c28e70f127bbcb9adb70e5b756cab95d`; fixed-base ancestry check exit 0. Read skill/protocol, AGENTS, project memory, planning index, contract_v1, original terminal audit, current control and Gate E handoff/amendment.

Strict forwarding-disabled SSH succeeded on mgmt-3; dedicated remote clone clean at `2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0`; account squeue empty. Existing Phase8 B/C/D monitors all PAUSED. No old jobs or outputs were changed.

Existing environment: torch `2.3.1+cu121`, transformers `4.51.3`, lm_eval `0.4.11`. Original calibration/test pool paths exist. Normal-user association includes debug/emergency_gpu; emergency_gpu is A800, RootOnly=NO, PriorityTier=300. Debug GPU is A40. Frozen resource cap: 24 GPUh total, at most 2 GPUs, 6h per formal job; batch1 unchanged.

Initial planning acknowledgement failed because recipient was archived. Dispatching task restored it; one retry succeeded to the exact unchanged planning ID. No authority/recipient was silently changed.

## Implementation and validation

Bounded delegated engineering: calibration agent owns K3 runtime admission and E calibration producer/verifier/tests; accuracy agent owns E four-cell manifest, scorer, closure and two-item K3 statistics/tests. Executor owns launcher integration, all Git/remote mutation, scheduler actions and terminal delivery.

Preflight estimate: under 29 minutes for two small calibration cells (first two plus upper-tail rows from original512), then four upper-tail test score cells. Serial A40 processes avoid importing A800 packing assumptions. The same calibration/accuracy launchers are used for preflight and formal scopes. Formal A800 starts with packing2; first512 scores per configuration are retainable canaries before remaining work.

Targeted validation before debug: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_e*.py'` passed 8 tests at initial integration; historical `test_phase8_accuracy*.py` passed 17. Calibration worker additionally ran 8 runtime and 9 historical calibration tests successfully. E runner/verifier/analyzer CLI `--help`, all three E launchers `bash -n`, and `git diff --check` passed. Protected wrapper/strategies/cache/config and original phase8_model_windows.json have no diff. K3 uses explicit opt-in shared accuracy entrypoints, leaving default Gate D behavior unchanged.
