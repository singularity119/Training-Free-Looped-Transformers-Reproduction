# Gate G execution evidence

Executor `01a08443-1e4f-7623-aa1b-69d3dbae406e`; planning `01a08024-3556-75c1-8209-64e89345b940`.
Authorization: committed G handoff/control `377e1b791b28dac971a47e1fc97f6816dc5c61ce`, after clean base `46d71b07c2f56162aec989792639672ad4780ec9`. Read AGENTS, PROJECT_MEMORY, planning README/control, research-gate-orchestrator skill and protocol. Local dedicated clone/loopscope/protected ancestor verified. G is an exploratory user-added 4B inclusive15:18 / B15→B19 window; K2/3/4×Loop/t1/t0, nine newly scored cells, three K-specific t1 bases plus one K2-source t0 shared across K. No old direction reuse.

## Implementation and first contact

Two bounded local helpers own new G calibration and score/verification/statistical modules; executor owns launchers, integration, Git, SSH/Slurm and terminal delivery. No protected runtime/default or historical model_windows edits planned. All new source is G-specific and reuses unchanged runtime/scoring primitives.

Strict BatchMode/StrictHostKeyChecking=yes/UpdateHostKeys=no/ClearAllForwardings=yes SSH reached mgmt-6. Original dedicated HPC source clean at a818dbb61ac82879858c34eb74b7ee152ad17bdc; no source mutation. squeue returned no account jobs. Original C calibration512 and D test14042 pools exist at handoff paths. Current debug is A40 with30min limit; emergency_gpu provides A800, PriorityTier300, RootOnly=NO, user account root has debug/emergency_gpu QoS (account name is not elevated privilege). A combined scontrol query was rejected for multiple arguments; separate queries succeeded. No job submitted before implementation/preflight checks.

Resource envelope:24GPUh including debug/retry,<=2simultaneous GPUs,<=6h perjob,8CPU/128G perGPU initially. Debug planned1A40/29min, one small new-window acquisition across3K and4basis, six test identities including four longest prompts for each of9arms, actual intervention and output verifier. Estimate under10min from previous runtime evidence; passing this exact producer/launcher/environment is required before formal. Formal A800 begins with retainable0:512 percell at packing1; remainder512:14042 only after canary closure and new-window resource check. Old E/F jobs, monitors, source and evidence remain owned by their prior tasks.

## Local validation

Initial integrated G-only unittest run:13 tests,1.426s,exit0. Seven G Python CLI help commands and three launcher bash-n checks exit0; git diff --check exit0. Tests cover new window and fourbasis mapping, K-specific fit_t1 versus shared fit_t0, nine unique cells and three independent3-comparison Holm families, missing/duplicate identity closure and analysis entry at the correct G closure state. Local environment has no torch, so SVD/CUDA/actual tensor intervention checks run in the mandatory debug job. Integration corrected runtime_checks versus verifier output filename collision and aligned the existing top-level basis fit_t contract before submission; no protected helper changes.

Directly read old D aggregate analysis at W/artifacts/phase8-gate-d-20260906T131600Z-analysis-a1/analysis.json: q4-native10262/14042,0.7308075772681955. This is descriptive context only, not new test gold access or new native acquisition.

## Committed source and deployment

Final integrated targeted run14tests/1.388s/exit0 (includes added analysis-admission regression). Producer commit `cffc8d3902c92f0c0cc3107b060aefe18c28a9b6`. Ordinary authorized push failed because github.com:22 timed out; no authentication/key/network change or alternate GitHub route attempted. The shell proceeded to date/revision output, so its final status0 is not push success. Authorized `git archive <producer> | strict SSH mkdir-and-tar` deployed to W/staging/phase8-gate-g-20260909T035003Z-source-a1, exit0. This snapshot is fixed; original shared source remains unchanged. Both endpoints use existing credentials only.

## Debug submitted

HPC snapshot targeted unittest14tests/1.788s/exit0. `sbatch --parsable --partition=debug --qos=debug --account=root --job-name=p8-g-debug --output=W/staging/phase8-gate-g-20260909T035003Z-debug-a1/slurm-%j.out --error=<same>.err <snapshot>/scripts/loopscope/phase8_gate_g_debug.sbatch <snapshot> W/inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json W/inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json W/runs/phase8-gate-g-20260909T035003Z-debug-a1 cffc8d3902c92f0c0cc3107b060aefe18c28a9b6` returned12692339/exit0. A40x1/8CPU/128G/29min. sacct/scontrol confirm real PENDING/MaxJobsPerAccount, StartTimeUnknown; no startup logs expected before running. No failure inferred or resubmission. Requested planning's minimum queue-observation exception; original stable-running monitor rule otherwise remains until planning responds.
