# Gate F execution evidence

Executor `01a08053-81a4-70b1-a306-a721189018a2`; planning `01a08024-3556-75c1-8209-64e89345b940`.
Current authority: F handoff superseding concurrent execution clause, control `ACTIVE_GATE=E,F`, user parallel exception. Initial observed clean local base `3d1cd8012ff4daafe5540c9cdb6beb140ee29dc8`; implementation starts after planning `208ab8284547beb672047e7fb102593e62202819`.

## Source isolation and implementation

E confirmed its queued launchers read mutable dedicated HPC source at `a818dbb61ac82879858c34eb74b7ee152ad17bdc`. Planning explicitly authorized F fixed source snapshots under dedicated W/staging; F never fast-forwards E source. Original environment remains `.venv-loopscope-cu121-20260711/bin/python` under dedicated source, with F snapshot/src explicitly first in PYTHONPATH. E owns its evidence edits; F stages only named F files. Ordinary push includes the two already-committed planning updates.

All implementation is new F modules/scripts/tests; original runtime, adapter, wrapper, cache, strategies and original t1 fitter remain unchanged. Three bounded helpers implemented basis checks, panel/statistics, and calibration runner; executor integrates, tests and owns all Git/remote/scheduler actions.

Four t0 bases use retained C K2 residuals, source scope/identities/native FP32 residual matrices and positions checked; original SVD arithmetic reused through a t0 matrix adapter. Explicit fit_t=0/source_k=2/applies_to_k=[2,3,4]; single basis file shared per model/window. Two q17 K3 t1 bases remain K-specific. Intervention still original runtime t>=1 only.

## Initial validation and remote read-only check

`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_f*.py'`: 16 tests, exit0. Tests cover shared basis provenance, actual tensor fit/save/load when torch available, 36 logical cells and separate12+12 Holm families, 16-new/224672 score closure, missing/duplicate scores and intervention timing. `git diff --check` and all three F launcher `bash -n` checks exit0. No full historical test suite rerun.

Strict key-only forwarding-disabled SSH reached mgmt-3; dedicated HPC source clean `a818dbb`. C cell-0 residuals and metadata exist with512 validation source. Current emergency_gpu is A800, PriorityTier300, RootOnly=NO; user association account root includes emergency_gpu/debug QoS (account name does not imply administrator privileges). Debug provides A40, 30min limit. F bounds48GPUh including debug/retries, <=2GPUs, <=6h perjob; no changes to E jobs.

Planned debug: two independent model jobs, each 1A40/8CPU/128G/29min, expected <10min from four preflight calibration identities and six test identities per cell. Both windows exercise exact t0 equality K2/3/4, shared loading and real intervention, saved-C fitting path, raw standard scores and verifier; q17 covers new K3 t1 too. Formal blocked until both pass. A800 canary uses retainable first512 before expanding packing from actual memory/throughput.
