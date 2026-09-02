# LoopScope 第三阶段全局计划与控制

最后更新：2026-07-17

## 1. 文件职责与权威优先级

本文件是 LoopScope 第三阶段唯一的可变控制面。它记录：

- 当前 Gate、executor 与授权状态；
- 第三阶段冻结的科学合同；
- 可写路径、远程/GPU/Slurm 权限和保护边界；
- 已审计 commit、run、jobs、决定、blocker 与下一 admission；
- repair、retry 和 terminal packet 规则。

信息源分工：

| 层 | 权威文件 | 职责 |
|---|---|---|
| 稳定政策 | `../loopscope-tflt/AGENTS.md` | 长期 Git、数据、HPC2、保护路径和 Gate 审计规则 |
| 当前控制 | 本文件 | Phase 3 动态状态、授权、冻结卡、Gate 决定 |
| 科学报告 | `../报告/LoopScope_第三阶段总体目标与Gate计划.md` | 假设动机、公式解释、实验设计与结果叙事 |
| Runbook | Gate handoff 指定的 `docs/` 文件 | 命令、CLI、schema 与运行步骤 |
| Evidence | Git、Slurm、write-once artifacts | 实际发生了什么 |
| 历史 | Phase 1/2 controls、报告、线程、memory | 背景与 provenance，不是当前授权 |

权威优先级：

```text
用户最新明确指令
> live Git / Slurm / 不可变工件
> loopscope-tflt/AGENTS.md
> 本 Phase 3 control
> runbook
> 报告与历史材料
```

本文件只在 Gate/decision、executor、审计对象、科学合同、权限边界、blocker 或下一 admission 变化时更新。不得追加逐命令日志、轮询记录或普通运行状态。

## 2. 当前身份与权威状态

```text
phase=LoopScope Phase 3
planning/audit thread=019f670d-24ca-7ad0-b92e-64438aa04ef1
phase2 planning/audit predecessor=019f6634-0369-7832-a617-d9f7e225345f; historical authority only

control status=PHASE3_COMPLETE_WITH_POSTHOC_X4_PASS
current state=PHASE3_TERMINAL_COMPLETE; POSTPHASE_X1_PASS; POSTPHASE_X2_PASS; POSTPHASE_X3_PASS; POSTPHASE_X4_PASS
current Gate=none
next planned Gate=none; Post-P3-X4 terminal follow-up is complete
P3-A state=PASS; authorization=false
P3-B state=PASS; authorization=false
P3-C state=PASS; authorization=false
P3-D state=PASS; authorization=false
P3-E state=PASS; authorization=false
P3-F state=PASS; authorization=false
completed P3-F executor=019f6aea-2b1d-7103-9bef-8365eb15640b; visible project thread `execute-LoopScope-H3V1-第三阶段-Gate F`; retained unarchived
P3-G state=PASS; authorization=false
completed P3-G executor=019f6b0a-2979-7662-b380-10bd3ed4b352; visible project thread `execute-LoopScope-H3V1-第三阶段-Gate G`; retained unarchived
void P3-F internal executor=/root/execute_loopscope_h3v1_phase3_gate_f; interrupted; no self-report, child review, or terminal packet admissible
completed P3-E executor=019f6a4f-e4c6-7170-add5-1c513c19aa16; retained in project; authorization=false
completed P3-D executor=019f6a24-06eb-7320-8405-4f4758ce7095; retained in project; authorization=false
completed P3-C executor=019f69ae-1ce3-7303-9221-c7779d57c8c5; retained in project; authorization=false
completed P3-B executor=019f67e9-4e17-7b53-a1db-b0d673a15bff; retained in project; authorization=false
completed P3-A executor=019f67bd-c4b6-72e1-a3a8-0aad8ec60879; retained in project; authorization=false
void executor=019f67b6-d57d-7203-822a-dc1dd58af603; authority revoked by user; no evidence admissible
completed Post-P3-X4 executor=019f6bb3-8f65-7ea1-a2f9-425a7959680f; retained visible unarchived; authorization=false
active authority=none; Phase 3 and authorized post-Phase3 follow-ups are complete

hypothesis card=H3_BASELINE_LOCAL_REVERSAL_WINDOW_RECOVERY_V1
scientific contract status=P3G_VARIABLE_WIDTH_TARGETED_FULL_CARD_BYTE_FROZEN
machine-readable card=../loopscope-tflt/configs/loopscope/phase3_card.json
machine-readable card SHA256=5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825
variable-width card=../loopscope-tflt/configs/loopscope/phase3_variable_width_card.json
variable-width card SHA256=a793a320830c5527823ca18fa3798d54cda691c530c8358adfc88a536696a490
active variable-width dual card=../loopscope-tflt/configs/loopscope/phase3_variable_width_dual_card.json
active variable-width dual card SHA256=b99352c415d9aa041f90d6d8c5b7cffd3771c549b348db36f10740998ec23612
active P3-G targeted-full card=../loopscope-tflt/configs/loopscope/phase3_variable_width_targeted_full_card.json
active P3-G targeted-full card SHA256=ccc4a285388966dba1043cd78028c535d8dba45e88fb99d9d535135ed8022acf
prior variable-width card execution status=SUPERSEDED_PRESERVED_NOT_EXECUTABLE
blind-12 metadata receipt SHA256=05e80be4c3576b768111418084d9a955d0187283eaab1aedcd5308a6f46a54f4
science report=../报告/LoopScope_第三阶段总体目标与Gate计划.md
science report SHA256=a5190377cfc19a21b498f3937bbd437a9b42a0fcdcadfc89c819d11145f0a47f

dedicated repo=../loopscope-tflt
branch=loopscope
pre-governance science baseline=1465c8019867560eeccc0f7e2845b36b50979ef7
governance baseline commit=58188777855ae6f8e71c1cccd06cf550c912841f
card freeze/base commit=ed9b31b48068decfcc6040021053d32e21f3c0fd
origin/loopscope current=684aff7a58bd62b6dd893e24689f3bfb4833f075
working tree current=clean
current AGENTS.md working-content SHA256=bd72c67140ebe13a17dfac3426479523fcb9c18868bdeafec06320db04fe06eb
P3-A authorized base commit=ed9b31b48068decfcc6040021053d32e21f3c0fd
P3-A audited final commit=5c7b45f7f48005f7e8539d8394a4b1a2e39e4218
P3-A audit decision=PASS
executor-retention governance commit=338809de322a73a4c2a0a1c4d987ae2a6db0dd7b
P3-B original authorized base commit=338809de322a73a4c2a0a1c4d987ae2a6db0dd7b
long-wait automation governance commit=07a10de3e09a1e8568529c20aad2a16dd6a2dbac
P3-B prospective base commit=07a10de3e09a1e8568529c20aad2a16dd6a2dbac
P3-B authorized base commit=07a10de3e09a1e8568529c20aad2a16dd6a2dbac
P3-B authorized run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z
P3-B audited interim commit=a519091f83a2b1ee4556a7761415905b958ca05d
P3-B audited final commit=3e40c36700005bf6a53b66100300f505f5fbceed
P3-B resource accounting file SHA256=e6be5b6b1ad26bd11f5b9de326a42cdd267c86ecc5e21fd69e7fe2253b350174
P3-B resource accounting internal SHA256=e89abcc704b337c833d749b1c4d3aa1b9df0664e89b1ddffcb1c59788b373a0a
confirmed-automation-handoff governance commit=ef783c313298145d809bf6adafc2a0108c5639f6
P3-C prospective base commit=ef783c313298145d809bf6adafc2a0108c5639f6
P3-C authorized base commit=ef783c313298145d809bf6adafc2a0108c5639f6
P3-C authorized run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3c-20260716T064601Z
P3-C implementation commit=8e9c5b135bd8cdf469691a3e5ee8c1c47602f269
P3-C first bounded repair commit=1cc93f3631891c43fda40346996a4fa791126377
P3-C content-overlap policy repair commit=b43d4b41cca599ecb4c451d8bae81c1237ad6e6b
P3-C continuation base commit=b43d4b41cca599ecb4c451d8bae81c1237ad6e6b
P3-C additional runtime repair allowance=1 single-purpose commit for list-valued unrelated command_args.window; explicitly supersedes the exhausted generic pre-outcome allowance
P3-C content-overlap policy=P3C_CONTENT_OVERLAP_POLICY_V2
P3-C audited final commit=1793df195518b823a0886249050a204bc17dd31e
P3-C audit decision=PASS
P3-C science label=RETROSPECTIVE_TOP1_RECOVERY
P3-C selected variant=CONSENSUS
P3-C selector freeze SHA256=8e0737d68cdf3c4cdb59a6f28df1e25932a156d0c944c2caa2e33ce63f8b2eef
P3-C retrospective adjudication SHA256=ef91c08f085ce89489515bc7c35eaa8075fc48384d9fd76c06fefb36dd97c147
P3-C missing12 predictions SHA256=0b3f12c6beb8070f70d9ab6def1918fa1aa35e8ff26303273d322f440aae25f0
P3-D authorized base commit=1793df195518b823a0886249050a204bc17dd31e
P3-D authorized run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3d-20260716T085629Z
P3-D audit decision=PASS
P3-D D1 job=9989896; 12/12 COMPLETED/0:0
P3-D D2 job=9989919; 12/12 COMPLETED/0:0
P3-D engineering summary SHA256=8cd9ab9f96044eae9ed93ec5004cf56c34e6ff0adc7b124ffe5a673ce0f5eb18
P3-D accepted deviation=D2 frozen verifier false negative because producer env snapshot omits offline keys; immutable launchers prove all three offline exports; structural fallback PASS
P3-E authorized base commit=1793df195518b823a0886249050a204bc17dd31e
P3-E authorized run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3e-20260716T094417Z
P3-E authorized implementation paths=src/tflt/loopscope/phase3_p3e.py;scripts/loopscope/run_qwen17_phase3_p3e.py;tests/test_loopscope_phase3_p3e.py
P3-E audited final commit=2f96124b2ea998ff2efcc5fa042149d69f3a6ced
P3-E audit decision=PASS
P3-E full job=9990349_[0-11]; 12/12 COMPLETED/0:0; each cell 57 tasks and 14042 ordered identities
P3-E all-25 analysis SHA256=41b502dfdbca0ba2a66d0d6231e5aeb7948e647b4d27b1b723ded59a550855f3
P3-E all-25 analysis internal SHA256=afb813a2aeb07785a3d551ea54dc4b82f44d62066521036df09af947d9a486e8
P3-E verifier SHA256=654fb13916eba29be16d6ee5c77d8e21c00c7596cec7051e31c733ff5d627438
P3-E verifier internal SHA256=2d4c52a6e282a9cc54ec85079d17026ce8d46a4050f9b0f23479bb7eeec64354
P3-E science label=WIDTH4_RANKING_SIGNAL_ONLY
P3-E observed all25 oracle=12:15; selector rank=1; outcome rank=1; delta=+0.4130465746 pp; regret=0 pp
P3-E blind primary=G_blind +0.0142429853 pp; CI95 [-0.3169954422,+0.3525138869] pp; INCONCLUSIVE
P3-F original card commit=d3666dd5853cdf378ea287d8f1e16aa37383bd8e
P3-F superseding authorized base commit=7303f289b90ee8e235e614855b0f2f8e2977978b
P3-F authorized executor=019f6aea-2b1d-7103-9bef-8365eb15640b
P3-F visible executor title=execute-LoopScope-H3V1-第三阶段-Gate F
P3-F void internal executor=/root/execute_loopscope_h3v1_phase3_gate_f; interrupted; evidence inadmissible
P3-F inherited implementation commit=41306450623f93d43eb36c90720854f9c0a8af4a; clean current baseline for visible executor; must be independently checked
P3-F audited final commit=41306450623f93d43eb36c90720854f9c0a8af4a
P3-F audit decision=PASS
P3-F strict98 artifact SHA256=74332060f0b1da99202a7e647bc5b18a6733a992ed152889043c77b5c3cffb00
P3-F edge168 artifact SHA256=d15127c2f0432008e1c6802eaabb59db82e08b5a088cc20db4e1466f24449a35
P3-F dual ranking SHA256=b118bac946bdf6a262fab0c52394ad0b777bb58438843b8314d999ea829c6739
P3-F verifier SHA256=23551a389ce60091ff615b378b809a35bd58620106a96e68784e99d9686dc045; internal=0afaef483ff194e249316f9fdbadc1c803a8b7c05b73930d0a9ebc57dc3de9dd; PASS
P3-F strict score top1=7:9 width3 score 54.982423273300824; point_eligible=false
P3-F edge-aware score top1=2:9 width8 score 88.00188385374891; point_eligible=false
P3-F unique eligible=12:15 under both rules; width4 compatibility PASS
P3-G frozen proposed four-cell shortlist=7:9,2:9,13:15,3:7; outcome-blind selection rationale=cross-rule top candidate, edge top1, second cross-rule high candidate, edge-only boundary candidate
P3-G authorized base commit=b7e28ea1ab9ec3221bad909f4346963d7e91153f
P3-G card freeze commit=b7e28ea1ab9ec3221bad909f4346963d7e91153f
P3-G authorized executor=019f6b0a-2979-7662-b380-10bd3ed4b352
P3-G authorized run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3g-20260716T130132Z
P3-G authorized implementation paths=src/tflt/loopscope/phase3_p3g.py;scripts/loopscope/run_qwen17_phase3_p3g.py;tests/test_loopscope_phase3_p3g.py
P3-G exact full cells=7:9,2:9,13:15,3:7
P3-G primary comparator=12:15; reused read-only, not rerun
P3-G secondary comparator=ordinary no-loop baseline; reused read-only, not rerun
P3-G full recipe=Qwen3-1.7B-Base MMLU test 14042 5-shot float16 K=2 fixed-horizon block damped_euler alpha1 beta0 cache-last decode-bypass
P3-G monitoring=executor-owned low-frequency automation after one bounded post-submit check; terminal wake must delete and resume exact executor
P3-G audited final commit=684aff7a58bd62b6dd893e24689f3bfb4833f075
P3-G audit decision=PASS
P3-G smoke job=9990929_[0-3]; 4/4 COMPLETED/0:0; structural only
P3-G full job=9990976_[0-3]; 4/4 COMPLETED/0:0; each cell 57 tasks and 14042 ordered identities
P3-G manifest SHA256=ccf59db567a94253b6ba68e850395f947ce776df1c08ebab9fe947ad7c3a3bb9; internal=fddfcbfe3be59dc6bc4a96a266288ef01e0ad2c9cdb4e7a755cffe161b36489b
P3-G analysis SHA256=fd952d4ed3c39c04ac226e5133e4e8d04cbac59ba5e4cee34c59a203076f9568; internal=b78af38c2c28ccc7e4fa6ef77e92a9cf20ce33202f95858a7e802163141f6e1f
P3-G verifier SHA256=07b218e376fb1ddfe3f2b9546774e093d5637c476f5d520c5c52eb08fa33e20b; internal=29bfa2d8fbebd40e6361923c547390b0b5b05d8371c4c697b184f7ea6f4c2bdb; PASS
P3-G science label=TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT
P3-G observed candidate ranking=7:9,13:15,3:7,2:9
P3-G best candidate=7:9; accuracy 62.3913972369%; delta vs baseline -0.3204671699 pp; delta vs 12:15 -0.7335137445 pp; CI95 vs 12:15 [-1.2321962683,-0.2278877653]; Holm p=0.0049252413
Phase 3 consolidated science conclusion=no-loop local reversal recovers and safely retains 12:15 within this cell, but blind enrichment is inconclusive and targeted variable-width high-score candidates do not improve on 12:15
P3-F original run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3f-20260716T115623Z; must remain absent/unmodified
P3-F active run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3f-dual-20260716T121430Z
P3-F authorized implementation paths=src/tflt/loopscope/phase3_variable_width.py;scripts/loopscope/run_qwen17_phase3_p3f.py;tests/test_loopscope_phase3_variable_width.py
P3-F exact widths=2,3,4,5,6,7,8
P3-F strict output=98 original-rule scored rows; unsupported rows omitted, no fill
P3-F edge-aware output=168 finite-score rows using available reference set R(s,w)
P3-F external compute=CPU-only local/HPC2; no GPU/Slurm/model/dataset forward
P3-F GitHub sync note=card commit pushed and origin verified from clean HPC2 clone; local origin ref updated after bounded local GitHub SSH timeout

current blocker=none
next admission=Post-P3-X4 may execute only the frozen single-layer-15 K=2/3/4 smoke and full experiment; no other follow-up is admitted
```

P3-G 已 `PASS`，第三阶段 P3-A 至 P3-G 全部工程闭合。全部 executor 权限已撤销并按用户要求继续保留为可见 provenance。当前没有 active executor、GPU/Slurm 或 outcome 授权。

### 2.1 Post-P3-X1：探索层 relaxed eligibility（用户于 2026-07-16 明确授权）

```text
status=PASS; authorization=false
executor=/root/execute_loopscope_h3v1_postphase_gate_x1r
visible title intent=execute-LoopScope-H3V1-第三阶段后分析-Gate X1-R
void executor=/root/execute_loopscope_h3v1_postphase_gate_x1; interrupted after repeated missing status/terminal delivery; its self-report and any unverified artifact are inadmissible
planning/audit recipient=019f670d-24ca-7ad0-b92e-64438aa04ef1
repo/branch/base=loopscope-tflt / loopscope / 684aff7a58bd62b6dd893e24689f3bfb4833f075
superseded run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-posthoc-x1-relaxed-eligibility-20260716T144456Z; replacement must only inspect existence/membership and must not reuse or modify it
run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-posthoc-x1r-relaxed-eligibility-20260716T145532Z

scientific status=post-hoc descriptive exploration; must not alter the frozen Phase 3 conclusion
selector inputs=P3-F strict98 and edge168 no-loop trajectory tables only
outcome access=forbidden; do not read P3-E/P3-G accuracy, correctness, answers, flips, or labels
GPU/Slurm/model forward=forbidden
Git/code mutation=not authorized by default; use a bounded read-only computation and write-once result artifacts

strict relaxed eligibility:
  center E_rate>0 and K_rate>0
  lower95CI>0 for each of O_H,O_K,F_H,F_K
  delete only the per-reference E_rate<0 and K_rate<0 requirements

edge-aware relaxed eligibility:
  center E_rate>0 and K_rate>0
  lower95CI>0 for each of A_H,A_K
  delete only the per-reference E_rate<0 and K_rate<0 requirements

equality at zero=fail
score/rank=preserve frozen P3-F score and rank; never compare strict and edge-aware raw scores numerically
bootstrap=no recomputation or tuning required when frozen per-row CI fields are sufficient; if recomputation is necessary, reuse the exact P3-F 2000-replicate subject-stratified joint stream and report why

required outputs:
  strict98 relaxed eligible count/rate and complete score-descending eligible table
  edge168 relaxed eligible count/rate and complete score-descending eligible table
  width-4 subsets for both rules
  original-eligible intersection and newly admitted windows
  one compact JSON summary plus CSV/JSON tables under the exact write-once run root

acceptance:
  exact P3-F input hashes match the frozen control values
  source row counts remain 98 and 168
  every relaxed decision follows only the frozen conditions above
  output contains no outcome/accuracy/correctness fields
  terminal packet is delivered once to the planning thread

agility budget=one bounded analysis, targeted internal consistency checks only, no full suite, no old-Gate re-audit, no receipt tree, no GPU or scheduler work
repair budget=one narrow executor-owned repair only if the normal artifact field/schema differs; scientific rule changes require planning decision

final result:
  strict relaxed eligible=28/98 (28.5714%); width4=12:15,6:9,7:10,14:17,4:7
  edge relaxed eligible=55/168 (32.7381%); width4=12:15,6:9,7:10,14:17,4:7,1:4
  original eligible intersection=12:15 only under both rules
  newly admitted=strict 27; edge 54
  summary SHA256=14f16d7eddc2fa16e4b121a37247660900ad05e4026e7ae88d4c7849089858ec
  decision=PASS for descriptive computation only; does not validate newly admitted windows as beneficial
```

### 2.2 Post-P3-X2：Strict 两阶段 mean-z 重排（用户于 2026-07-16 明确授权）

```text
status=PASS; authorization=false
executor=/root/execute_loopscope_h3v1_postphase_gate_x2
visible title intent=execute-LoopScope-H3V1-第三阶段后分析-Gate X2
planning/audit recipient=019f670d-24ca-7ad0-b92e-64438aa04ef1
repo/branch/base=loopscope-tflt / loopscope / 684aff7a58bd62b6dd893e24689f3bfb4833f075
run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-posthoc-x2-strict-mean-z-20260716T151936Z

scientific status=post-hoc outcome-blind descriptive reranking; Phase 3 conclusion remains frozen
source=P3-F strict_consensus_98_table.json SHA256 74332060f0b1da99202a7e647bc5b18a6733a992ed152889043c77b5c3cffb00
source rows=98
outcome access=forbidden; no P3-E/P3-G/X1 outcome or accuracy/correctness/gold/answer/flip fields
GPU/Slurm/model forward=forbidden
Git/code mutation=not authorized

stage-1 admission:
  E_rate>0 and K_rate>0
  lower95CI>0 for each of O_H,O_K,F_H,F_K
  equality at zero fails

stage-2 score:
  mean_z=(z_O_H+z_O_K+z_F_H+z_F_K)/4
  rank admitted rows by mean_z descending, then width ascending, then start ascending
  preserve original min_z score/rank and compute rank delta for comparison
  also publish mean_z for all 98 rows with admitted flag, but selection candidates remain admitted rows only

required outputs:
  strict_mean_z_all98.json and .csv
  strict_mean_z_admitted.json and .csv
  strict_mean_z_summary.json containing counts, complete admitted ranking, top15, width4 subset, and min-z versus mean-z rank changes

acceptance:
  source hash/count exact
  admitted set exactly follows frozen stage-1 rule and is expected to match X1 strict relaxed set without reading X1 result content as authority
  mean_z arithmetic and ordering exact
  no outcome fields
  exactly one terminal packet delivered to planning

agility budget=one bounded read/filter/arithmetic/write pass plus one internal consistency check; no framework, tests, full suite, receipt tree, old-Gate audit, GPU or scheduler work
repair budget=one narrow schema-field adaptation only; score/admission changes require planning decision

final result:
  admitted=28/98 (28.5714%)
  mean-z top6=4:7,3:4,7:9,14:15,6:9,12:15
  width4 mean-z order=4:7,6:9,12:15,7:10,14:17
  12:15 mean-z rank=6; original min-z admitted rank=2
  summary SHA256=0e710d60335b02ff9f29f058a2a78e205718e0cd3b468dd44d29a9419ad2cb74
  decision=PASS for descriptive arithmetic only; no claim that mean-z predicts loop outcome
```

### 2.3 Post-P3-X3：Entropy-z 与 KL-z 分离排名（用户于 2026-07-16 明确授权）

```text
status=PASS; authorization=false
executor thread=019f6b98-88bc-7e21-9ea7-19c3a3bc5a56
executor title=execute-LoopScope-H3V1-第三阶段后分析-Gate X3
executor model/reasoning=gpt-5.6-sol/max
planning/audit recipient=019f670d-24ca-7ad0-b92e-64438aa04ef1
repo/branch/base=loopscope-tflt / loopscope / 684aff7a58bd62b6dd893e24689f3bfb4833f075
run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-posthoc-x3-split-z-20260716T153114Z

source=P3-F strict_consensus_98_table.json SHA256 74332060f0b1da99202a7e647bc5b18a6733a992ed152889043c77b5c3cffb00
source rows=98
scientific status=post-hoc outcome-blind descriptive ranking; Phase 3 conclusion remains frozen
outcome access=forbidden; no P3-E/P3-G/X1/X2 accuracy/correctness/gold/answer/flip fields
GPU/Slurm/model forward=forbidden
Git/code mutation=not authorized

scores:
  entropy_mean_z=(z_O_H+z_F_H)/2
  kl_mean_z=(z_O_K+z_F_K)/2
  compute both for all 98 strict-supported rows

rankings:
  all98 entropy rank: entropy_mean_z descending, width ascending, start ascending
  all98 KL rank: kl_mean_z descending, width ascending, start ascending
  admitted28 entropy and KL ranks use the same X2 stage-1 condition: E_rate>0,K_rate>0 and lower95CI>0 for O_H,O_K,F_H,F_K; equality fails
  preserve original min-z score/rank and X2 mean-z when derivable from the same row; do not use X2 artifact as scientific authority

required outputs:
  strict_split_z_all98.json and .csv
  strict_split_z_admitted.json and .csv
  strict_split_z_summary.json with top15 for each metric/all98 and admitted28, width4 orders, Spearman rank correlation between entropy and KL rankings, and largest rank disagreements

acceptance=source hash/count exact; formulas/order exact; admitted count 28; no outcome fields; exactly one terminal packet to planning
agility budget=one bounded arithmetic/write pass and one internal check; no framework, tests, full suite, receipt tree, old-Gate audit, GPU or scheduler work
repair budget=one narrow schema-field adaptation only; score/ranking definition changes require planning decision

final result:
  all98 entropy/KL Spearman rho=0.8113089659481412
  admitted28 entropy/KL Spearman rho=0.48275862068965514
  all98 entropy top3=4:7,14:15,3:4
  all98 KL top3=7:9,9:15,13:15
  admitted28 top3 are the same under each respective metric
  width4 entropy order=4:7,6:9,12:15,14:17,7:10 within admitted28
  width4 KL order=12:15,7:10,6:9,14:17,4:7 within admitted28
  summary SHA256=02c5defc311203d72d95e12bc04da7361f8f1ae12ad5aec29e8b66e8d31619c5
  decision=PASS for descriptive split ranking only; neither component alone is validated as a general outcome selector
```

### 2.4 Post-P3-X4：单层 15 循环体 K=2/3/4（用户于 2026-07-17 明确授权）

```text
status=PASS; authorization=false
executor thread=019f6bb3-8f65-7ea1-a2f9-425a7959680f
executor title=execute-LoopScope-H3V1-第三阶段后分析-Gate X4
executor model/reasoning=gpt-5.6-sol/max
planning/audit recipient=019f670d-24ca-7ad0-b92e-64438aa04ef1
routing supersession=the initial recipient ending 4efc1 was a handoff typo and is VOID; all X4 terminal delivery must target the exact live planning thread ending 4ef1
repo/branch/base=loopscope-tflt / loopscope / 684aff7a58bd62b6dd893e24689f3bfb4833f075
run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-posthoc-x4-layer15-k234-20260717T001249Z

objective=measure whether the sharp B14->B15 entropy-drop/KL-rise boundary becomes useful when decoder layer 15 alone is the repeated loop body, under K=2,3,4
scientific status=post-Phase3 targeted mechanism follow-up; does not modify the frozen Phase 3 selector conclusion

frozen cells:
  window=15:15 only
  K=2,3,4
  model=Qwen/Qwen3-1.7B-Base revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1
  task=MMLU test full 14042; 5-shot; float16
  fixed-horizon; iteration_mode=block; strategy=damped_euler; alpha=1; beta=0; cache=last; decode=bypass
  baseline comparator=reuse frozen ordinary no-loop baseline read-only
  contextual comparator=reuse frozen 12:15 K=2 result read-only; do not rerun

execution stages:
  1. inspect existing CLI/launcher and implement only the smallest card/sidecar needed for window 15:15 with K=2,3,4
  2. run targeted tests/dry-run only; no full suite or old-Gate re-audit
  3. one three-cell limit=5 structural smoke; verify exact recipe/revision/identity and operator_body_calls equals K
  4. if smoke passes, promptly submit one three-cell full array under available HPC2 resources
  5. after exact 3/3 terminal completion, analyze once and report accuracy/delta versus baseline, paired CI, flips/exact McNemar, and K-wise ordering; one compact verifier/check

allowed actions=read existing frozen code/artifacts; add minimal X4 card/script/source/test paths if required; targeted local/HPC2 checks; push loopscope; create fresh write-once root; submit smoke/full Slurm arrays; use low-frequency executor-owned automation for long wait
protected=AGENTS/control/report; wrapper.py; strategies.py; cache.py; config.py; lm-eval/task/renderer; prior run roots and comparator results
forbidden=other windows; K outside 2/3/4; model/task/recipe changes; partial full-outcome inspection; rerun baseline/12:15; retries after valid outcome; selector retuning; cross-model/cross-task work
scheduler boundary=continue existing Slurm array 9991667_[0-2] on emergency_gpua40; the previously discussed A800/emergency_gpu fallback is withdrawn; no cancel, resubmit, GRES/partition change, or mixed-hardware execution

acceptance:
  exact three cells 15:15@K2,K3,K4 complete on identical 14042 ordered identities
  loop instrumentation confirms actual single-layer body and operator_body_calls=K
  exact frozen model/task/recipe except K
  one analysis reports all three accuracies and paired comparisons to baseline; no favorable-result rerun
  terminal packet GATE_POST_P3_X4_FINAL_AUDIT or BLOCK visibly delivered to planning thread

agility budget=smallest implementation increment; targeted tests; one smoke then immediate full; no exhaustive tamper cases, receipt tree, repeated suite, or historical Gate audit
repair budget=up to two narrow pre-outcome executor-owned engineering repairs; no scientific/scheduler retry after valid full outcome without new planning authority
next Gate=none; X4 is terminal unless the user separately authorizes another hypothesis
audited commit=fb5ead49c6ad7b79fe8fd92e44e8ec6aa65d5c2d
decision=PASS; exact 15:15@K2/K3/K4 full cells completed and verifier status PASS
scientific result=K2 62.71899% (+0.00712 pp vs baseline, CI crosses zero); K3 62.51246%; K4 62.49822%; observed order K2>K3>K4; none improves on 12:15@K2
interpretation=the isolated B14->B15 entropy/KL reversal is not sufficient by itself; useful loop behavior depends on the wider 12:15 contextual block
```

各 Gate 主 executor 可按任务需要适度使用 subagent，最多同时 3 个 child（含主 executor 总并发不超过 4），无需用满。subagent 只承担边界清楚的独立阅读、实现或测试子任务，不获得新的科学、路径、网络、数据、GPU、Slurm、Git 或下一 Gate 权限；主 executor 始终独占共享工作树的最终写入整合、Git、外部系统动作、provenance 与唯一 terminal packet 责任，并须避免多个 agent 并发修改同一文件。

## 3. H3 冻结科学合同

本节是未来 P3-A machine-readable card 的语义权威。P3-A handoff 前必须把同一字段无语义变化地 materialize 为版本化配置并冻结 byte hash；executor 不得在实现时自行改公式、阈值、variant、数据分层或统计。

### 3.1 Model、task 与 loop outcome 配方

```text
card=H3_BASELINE_LOCAL_REVERSAL_WINDOW_RECOVERY_V1
model=Qwen/Qwen3-1.7B-Base
model_revision=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu
num_fewshot=5
dtype=float16

loop_k=2
iteration_mode=block
strategy=damped_euler
alpha=1.0
beta=0.0
cache_strategy=last
decode_mode=bypass

phase3_core_width=4
width4_universe=start 0..24; total 25
claim_scope=within-cell feasibility only
```

第三阶段核心不得同时改变 K、alpha、beta、strategy、cache、decode、model 或 task。任何变化必须回到规划线程另立 hypothesis card。

### 3.2 Primary no-loop trajectory

```text
trajectory_split=MMLU validation
trajectory_samples=all 1531 unique target records across all 57 subjects
trajectory_sampling=none
trajectory_record_rule=one formal primary record per identity
trajectory_boundaries=B_0...B_28
fewshot_split=dev
fewshot_count=5 per subject under the frozen renderer

primary_selector_inputs=boundary A/B/C/D choice distribution, H, KL-to-final, CE, identity/provenance
forbidden_selector_inputs=target gold, correctness, loop residual, r, q, NCA, K2/full accuracy, answer flips, test outcomes
legacy_validation512=secondary robustness only; never select variant or replace/join the 1531 primary population
```

工程 smoke 与正式 primary acquisition 必须使用隔离工件；smoke 记录不得拼入 1,531 primary atlas。

### 3.3 Outcome population 与数据隔离

```text
full_outcome_split=MMLU test
full_outcome_samples=14042 ordered paired samples
trajectory_vs_outcome_identity_intersection=0 required
trajectory_vs_outcome_sanitized_content_hash_intersection=descriptive only; zero-intersection admission constraint retired by P3C_CONTENT_OVERLAP_POLICY_V2 below
historical_development_cells=13
blind_completion_cells=12
```

#### P3C_CONTENT_OVERLAP_POLICY_V2

C0 对冻结的官方 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` 数据现场投影后确认：validation-1531 与 test-14042 的 canonical identity 交集为 0，但按 byte-frozen `question + ordered choices` 定义存在 28 个真实内容重复 hash，分别涉及 28 条 validation 与 28 条 test 记录。这是官方 split 的重复内容，不是 hash collision、实现错误或 outcome 泄漏；占 validation 约 1.83%、test 约 0.20%。

用户明确决定舍弃 sanitized-content 零交集约束，以 no-loop baseline 层级 choice trajectory 的选窗预测实验为第一目标。该字段继续作为数据描述与阶段末局限性证据，但不再参与 P3-C admission、PASS/BLOCK 或 selector 计算：

- canonical identity intersection 必须仍为 0；validation=1,531、test=14,042 与 57 subjects 的主总体保持不变，不删除、不重分 split；
- C0 记录实际 overlap count、重叠 content hashes 及两侧 identities，并明确 `outcome_values_read=false`；content overlap 的数量本身不触发 `BLOCK`，identity overlap 非零或出现 outcome access 仍立即 `BLOCK`；
- P3-C 不修改 selector 公式、variant、K=2/fixed-horizon 配方或 historical/blind 分区；byte-frozen card 中的 content-zero 字段由本 live control 明确 supersede，不要求重做 P3-A 或重冻结整张 card；
- 当前 executor 只在既有 P3-C sidecar/test 范围内做一次最小修复，targeted tests 通过后直接运行 C0→C1→C2；不得重复 full suite、重审旧 Gate、扩建 receipts 或增加篡改测试；
- overlap-excluded sensitivity 与限制表述由规划线程在阶段末合并审计时决定，默认不放在 P3-C 关键路径；若主结论临界，再做一次最小的 secondary sensitivity，不反向调 selector。

历史 13 窗：

```text
4:7,6:9,8:11,9:12,11:14,12:15,13:16,
14:17,15:18,17:20,18:21,20:23,22:25
```

预期 blind-12：

```text
0:3,1:4,2:5,3:6,5:8,7:10,
10:13,16:19,19:22,21:24,23:26,24:27
```

在 card byte-freeze 前必须只用非数值 outcome metadata 核验 blind-12 是否从未产生或被查看过 outcome。发现任一既有 outcome 时，应先把该 cell 降级为 historical 并发布 superseding card；不得继续把它称为 blind。

### 3.4 指标恒等式与局部结构

```text
H_(i,j) = -sum_c p_(i,j,c) log p_(i,j,c)
D_(i,j) = KL(p_(i,j) || p_(i,N))
C_(i,j) = H_(i,j) + D_(i,j) = CE(p_(i,j),p_(i,N))

E(s,w)      = mean_i[H_(i,s)-H_(i,s+w)]
K(s,w)      = mean_i[D_(i,s+w)-D_(i,s)]
CEdrop(s,w) = E(s,w)-K(s,w)

O_H(s,w) = E(s,w)-0.5[E(s-1,w)+E(s+1,w)]
O_K(s,w) = K(s,w)-0.5[K(s-1,w)+K(s+1,w)]

F_H(s,w) = E(s,w)-0.5[E(s-w,w)+E(s+w,w)]
F_K(s,w) = K(s,w)-0.5[K(s-w,w)+K(s+w,w)]
```

三个预声明 variant：

```text
SHIFT support=start 1..23; score=min(z_OH,z_OK)
FLANK support=start 4..20; score=min(z_FH,z_FK)
CONSENSUS support=start 4..20; score=min(z_OH,z_OK,z_FH,z_FK)
z_X=point_X/max(SE_boot(X),1e-12)
```

Eligibility：

- 中心 `E>0` 且 `K>0`；
- 对应左右比较窗口/层段 `E<0` 且 `K<0`；
- 所有相关 contrast 的 percentile bootstrap 95% CI 下界 `>0`；
- 支持域外一律 `UNSCORABLE/ABSTAIN`，不得补 0 或插值；
- point-eligible top-1 的 bootstrap selection frequency `<0.80` 时输出 `ABSTAIN`。

H/K 通过 CE 恒等式耦合。`CEdrop` 必须报告，但不得作为第三份独立证据重复加权。

### 3.5 Baseline trajectory 统计

```text
primary estimator=mean over all 1531 validation samples
bootstrap=subject-stratified joint sample bootstrap
bootstrap replicates=2000
bootstrap seed=20260716
```

对每个 subject `s`，在其固定 `n_s` 条 validation 样本内有放回抽取 `n_s` 条；57 个 subjects 全部保留。同一 replicate indices 必须用于全部窗口和全部指标。Pooled 1,531 ordinary bootstrap 与 legacy-512 nested subset 只作 robustness，不参与 variant 选择。

### 3.6 Retrospective variant selection

三个 variant 的 baseline-only 全支持域排名必须在读取任何 historical outcome value 前冻结。

历史 13 窗阶段只允许按以下规则选一个 variant：

1. historical supported domain 内唯一 top-1=`12:15`；
2. top-1 selection frequency `>=0.80`；
3. `4:7、6:9` 不进入 primary eligible set；
4. 若多个 variant 同时满足，固定优先级 `CONSENSUS > FLANK > SHIFT`；
5. 若无 variant 满足，H3 当前版本停止，不调公式或阈值挽救。

只有 `RETROSPECTIVE_TOP1_RECOVERY` 具备进入 P3-D 的科学条件。`SHORTLIST_ONLY`、`NOT_SUPPORTED` 或证据不完整均停止当前 blind-completion 路线，等待用户决定新卡或终止。

### 3.7 Blind primary endpoint

```text
n=selected variant 在 blind-12 中的可评分窗口数
m=ceil(n/3)
top_m=冻结排名最高的 m 个 blind windows
bottom_m=冻结排名最低的 m 个 blind windows
G_blind=mean[Delta_acc(top_m)]-mean[Delta_acc(bottom_m)]

outcome bootstrap=joint paired bootstrap over all 14042 sample indices
replicates=2000
seed=20260710

SUPPORTED iff lower95CI(G_blind)>0
REFUTED iff upper95CI(G_blind)<0
INCONCLUSIVE otherwise
```

Spearman/Kendall 只作描述性。全部 25 个 width-4 outcome 仍需完成，以检查 observed oracle 是否落在所选 variant 的支持域；oracle 在域外必须标记 `WIDTH4_COVERAGE_FAILURE`。

### 3.8 Width-4 科学标签

按下列优先级 first-match：

1. `WIDTH4_INCONCLUSIVE`：关键 evidence/identity/freeze/statistics 不完整；
2. `WIDTH4_HARMFUL_SELECTION`：冻结 selected window 得到统计支持的负收益；
3. `WIDTH4_COVERAGE_FAILURE`：回溯成功但 all-25 oracle 在支持域外；
4. `WIDTH4_WITHIN_CELL_FEASIBILITY_SUPPORTED`：回溯成功、oracle 在域内、top-1 命中、selected point gain 为正、主要 false positives 被排除、blind enrichment supported；
5. `WIDTH4_RANKING_SIGNAL_ONLY`：top-3/富集有价值，但不足以支持唯一 top-1；
6. `WIDTH4_RETROSPECTIVE_RECOVERY_ONLY`：只解释历史 13 窗；
7. `WIDTH4_NOT_SUPPORTED`：证据完整但未满足上述支持条件。

Gate `PASS` 只表示执行和证据可信，不等于科学标签为正。

## 4. Gate 状态表

| Gate | Objective | Executor | State | Audited object | Next admission condition |
|---|---|---|---|---|---|
| P3-A | 实现 validation-1531 pool、trajectory schema、SHIFT/FLANK/CONSENSUS analyzer 与 verifier | `019f67bd-c4b6-72e1-a3a8-0aad8ec60879`; old `019f67b6-d57d-7203-822a-dc1dd58af603` is `VOID` | `PASS` | `5c7b45f`; 29 targeted tests + verifier `38081a...`; protected diff empty | retained; authorization revoked |
| P3-B | 构建全量 validation pool 并采集 fresh no-loop `B_0...B_28` trajectory | `019f67e9-4e17-7b53-a1db-b0d673a15bff` | `PASS` | clean local/remote `3e40c36`; 41 Phase 3 tests; jobs `9988046` + `9989056_[0-31]` all `COMPLETED/0:0`; verifier `3cb1f752...` PASS; accounting `e89abcc7...` PASS | executor retained, all authority revoked; AGENTS handshake landed at `ef783c3`; P3-C requires separate authorization/new executor |
| P3-C | 冻结 baseline atlas；分阶段解封 historical-13；选择唯一 variant | `019f69ae-1ce3-7303-9221-c7779d57c8c5` | `PASS` | clean local/remote `1793df1`; selector `8e0737d6...`; adjudication `ef91c08f...`; `CONSENSUS`; `RETROSPECTIVE_TOP1_RECOVERY` | executor retained; authority revoked; P3-D admitted |
| P3-D | missing-12 工程 smoke/limit，不读科学 accuracy | `019f6a24-06eb-7320-8405-4f4758ce7095` | `PASS` | clean local/remote `1793df1`; jobs `9989896/9989919` each 12/12 `COMPLETED/0:0`; engineering summary `8cd9ab9f...`; accepted nonmaterial verifier false negative | executor retained; authority revoked; P3-E admitted |
| P3-E | missing-12 K=2 full 与一次性 all-25 analysis | `019f6a4f-e4c6-7170-add5-1c513c19aa16` | `PASS` | clean local/remote `2f96124`; job `9990349_[0-11]` 12/12 `COMPLETED/0:0`; analysis `41b502df...`; verifier `654fb139...` PASS; `WIDTH4_RANKING_SIGNAL_ONLY` | executor retained; authority revoked; width-4 endpoint complete |
| P3-F | width 2..8 outcome-blind 双排名：原规则 98 + edge-aware 168 | `019f6aea-2b1d-7103-9bef-8365eb15640b` (`execute-LoopScope-H3V1-第三阶段-Gate F`) | `PASS` | clean local/HPC2 `4130645`; exactly six artifacts; strict98/edge168 exact；verifier `0afaef48...` PASS；width4 exact reproduction；outcome unread | executor retained, authority revoked；P3-G shortlist frozen |
| P3-G | 可变宽度 targeted full | `019f6b0a-2979-7662-b380-10bd3ed4b352` (`execute-LoopScope-H3V1-第三阶段-Gate G`) | `PASS` | clean local/HPC2 `684aff7`; smoke `9990929_[0-3]` 与 full `9990976_[0-3]` 均 4/4 `COMPLETED/0:0`; analysis `fd952d4e...`; verifier `29bfa2d8...` PASS; science=`TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT` | executor retained, authority revoked；Phase 3 terminal complete |

状态名本身不授予权限。只有本 control、精确 handoff 与 executor identity 三者一致，当前 Gate 才算 `AUTHORIZED`。

## 5. P3-A 计划授权包

### Objective

- 校验并严格实现 planning 已 materialize 的 machine-readable H3 card；不得改其 bytes/SHA256；
- 实现 full-validation pool contract、no-loop trajectory schema、baseline-only analyzer、source validator 与 deterministic verifier；
- 仅做本地 pure-Python/fixture 测试，不访问 live dataset、model、HPC2 或 outcome。

### 默认允许范围（正式 handoff 仍需列 exact paths）

- `src/tflt/loopscope/` 下新的 Phase 3 sidecar modules；
- `configs/loopscope/` 下 Phase 3 card/schema；
- `scripts/loopscope/` 下 pool/analyzer/verifier helpers；
- `tests/` 下 Phase 3 测试；
- `docs/loopscope_phase3.md` 或 handoff 指定的 Phase 3 runbook。

### 默认禁止

- SSH、HPC2、下载、模型/数据访问、Slurm/GPU；
- historical/blind/full outcome values；
- 修改 `wrapper.py、strategies.py、cache.py、config.py`；
- 修改 lm-eval package/source/task/YAML/HFLM/TaskManager/simple_evaluate/renderer/batching/metric；
- 改第 3 节科学合同；
- 进入 P3-B。

### Must-pass

- 1,531/57 subjects、无抽样/重复、validation split 与 forbidden-field 合同可被 fixture 验证；
- `B_0...B_28`、choice distribution/H/KL/CE finite 与 identity closure；
- SHIFT/FLANK/CONSENSUS 支持域、eligibility、score、ABSTAIN、tie-break 与 bootstrap 可复现；
- historical/blind partition 和 outcome-forbidden process/file allowlist 测试；
- validation/test canonical identity disjointness 与 sanitized-content overlap 描述性 verifier；
- Phase 3 targeted tests 与 `git diff --check` 通过；全量 suite 不得新增失败。已知基线唯一失败是历史 Phase 2 card 对 `phase2_analysis.py` 的 immutable hash mismatch（actual `cf60e24a...` vs frozen `817b24ec...`），不得为 P3-A 修改历史 card；若 Phase 3 改动引入额外失败或触碰该文件则为材料性失败。

P3-A 仅对上表绑定 executor 生效；本节仍需与精确 `GATE_P3A_HANDOFF` 同时读取，且不授权 P3-B。

## 6. P3-B 计划授权包

### Objective 与顺序

```text
B0 renderer/projection source closure
→ B1 validation-1531 pool freeze
→ B2 isolated 4-sample no-loop smoke
→ B3 full validation-1531 no-loop acquisition
```

### Admission

- P3-A 由规划线程独立审计 `PASS`；
- card SHA256=`5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825`；model 与 expected tokenizer revision=`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`；B2 必须用 strict revision closure 实测确认 tokenizer commit，任何不一致在 primary launch 前 `BLOCK`；
- planning-signed Phase 1 renderer anchor 冻结为 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/inputs/loopscope-qwen17-mmlu-phase1-20260711-043615/lm_eval_mmlu_projection.jsonl`，file/projection SHA256=`439e41113458ccfbd51d211ab7fae4e91ecc02723c7b14ad6c1969803aa61a6a`；对应 `lm_eval_mmlu_renderer_manifest.json` file SHA256=`48ba67c204249b5f2f9a6cf711f8678711b5e606398d35ba4d919d4eee70e6e7`、internal manifest SHA256=`e1baaa92c8ab3804c75a5e415a324d00fed0277d17c26b16d12b8c99aafbe949`；renderer source=`105ff765d068bd246387249d3941a697c964070834d92d7a3268056b7a68a8a9`、template=`15dea12df4d7dc69b4d9d425a51bc14976f560b3e5ed52a319895cde79f2dabb`、render contract=`eff9612c2984c7f2668e5cfb11d9eec89a5468c6f227ff94e4a226843043d6f4`、dataset fingerprint=`c3f97b35254b7953d20cffc65414e1c78a792128f215805962c0db5d4d185e11`；
- 新 P3-B executor、write-once root、HPC2/GPU/Slurm 与 adaptive-sharding 权限被明确授权。

### Input/output boundary

P3-B exact read-only inputs 只允许：上述 planning-signed renderer projection/manifest、冻结 card/config、已审计 dedicated repo commit/code、只读 venv/runtime、model/tokenizer cache，以及 B0 隔离 safe-projection 进程对 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` validation cache 的一次读取。历史 projection 缺少 `question/ordered_choices`，因此 B0 获得这一窄权限：只提取 question 与有序四 choices，并按 `(task_name,target_doc_id,target_doc_sha256)` 与 1,531 条历史 anchor 精确 join；loader 中的 target gold 不得读取、序列化、比较、记录或影响控制流，输出先通过 closed-world/forbidden-field scan。B0 receipt/hash 闭合后，raw dataset 与含额外/forbidden keys 的历史 projection 对 B1-B3 均转为禁止输入。test-14042 canonical identity 隔离与 sanitized-content overlap 描述保留给 P3-C/C0 只读 metadata Gate，不阻塞 P3-B。

P3-B 可以只在新的 Phase 3 路径实现最小 live producer/launcher/merge/verifier：`src/tflt/loopscope/phase3_acquisition.py`、`scripts/loopscope/run_qwen17_phase3_p3b.py`、`tests/test_loopscope_phase3_acquisition.py`，以及必要的 `docs/loopscope_phase3.md` 更新。不得修改旧 `probe.py`、`mmlu_renderer.py`、CLI、P3-A card/schema/statistics/verifier 语义或任何 Phase 1/2/protected source；若最小实现无法在这些路径内完成则 `BLOCK`。

B2 smoke 与 B3 primary 必须隔离。B3 对全部 1,531 identities 各生成一个正式 record；允许 disjoint shards，但 merge 后必须恰好 1,531、无 missing/duplicate/extra。

P3-B 最低 write-once 工件包括：live `source_manifest.json`（不得覆盖 committed contract-only source manifest）、renderer source receipt、validation-1531 pool/manifest/forbidden scan、独立 smoke records/manifest/report、launch 前冻结的 shard manifest、逐 shard records/receipt/stdout/stderr/Slurm terminal state、合并后的 `validation1531_no_loop_trajectories.jsonl` 与 trajectory manifest、shard merge receipt、probe report/verifier receipt，以及 scheduler throttle/resource accounting receipt。raw logits/hidden states 不得持久化；只允许 selector-safe 四选项概率、H/KL/CE/agreement 与 provenance。

### Resource rule

不预设 GPU 数量、GPU-hours 或墙钟硬上限。Admission 完成后按 HPC 空闲资源选择合理 shard count 并尽快提交；launch 前冻结 shard membership，之后只可调整 scheduler throttle，不得改变 sample membership/order、batching 语义、scientific argv/config 或 artifact identity。

### Terminal

只允许 `GATE_P3B_FINAL_AUDIT` 或 `BLOCK`。P3-B `PASS` 只证明 trajectory 可信，不解封任何 outcome。

## 7. P3-C 计划授权包

P3-C 是只读 outcome 分阶段解封 Gate：

```text
C0 canonical source/outcome-metadata closure
→ C1 outcome-blind 25-window atlas and deployment ranks freeze
→ C2 historical-13 values unseal and retrospective adjudication
```

### Admission 与绑定

- executor=`019f69ae-1ce3-7303-9221-c7779d57c8c5`，模型=`gpt-5.6-sol / max`；仅该线程拥有本 Gate 权限；
- repo/branch/base=`loopscope-tflt / loopscope / ef783c313298145d809bf6adafc2a0108c5639f6`，授权时 local/origin clean；
- write-once run root=`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3c-20260716T064601Z`；
- card SHA256=`5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825`；P3-B trajectory verifier internal SHA256=`3cb1f7529e1f3e362d6a6eb0df6ec4cf08ccd118bceb98dd8d3b1e5343cc9cd2`；
- P3-B canonical root=`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z`；Phase 1 canonical root=`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022`；
- Phase 1 `control/phase1_run_manifest.json` expected SHA256=`e497983d6b47ba2ff8e296a5375838737ad666b5f7dc95b1f366a8bd21111750`，ordered test identity SHA256=`872c9f6bd7f40e3c6fa40b13cc862033379ac0a9cbc439a1593653945f3d521d`。任何 live bytes/mapping/revision 不闭合即 `BLOCK`。

### 最小实现范围

只允许新增或修改：

```text
src/tflt/loopscope/phase3_p3c.py
scripts/loopscope/run_qwen17_phase3_p3c.py
tests/test_loopscope_phase3_p3c.py
docs/loopscope_phase3.md  # 仅必要 runbook
```

必须复用既有 `phase3_pool/phase3_schema/phase3_analysis` 与 Phase 1 paired helpers；不得修改 `phase3_analysis.py、phase3_pool.py、phase3_schema.py、phase3_verifier.py` 或既有 P3-A/P3-B/Phase 1/2/protected source。实现与 tests 必须在任何 C2 outcome value 解封前 commit/push 并在 HPC2 dedicated clone `ff-only` 闭合；C1 freeze 后禁止因历史结果再改 selector 公式、阈值、支持域、排序或代码。

### C0/C1

- 只读 source、identity 和 outcome existence metadata；禁止 accuracy、answer、correctness、results values；
- 在隔离 safe-projector 中从冻结 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` test 只提取 identity、question 与 ordered choices；target/gold 不得访问、序列化、比较、记录或影响控制流；
- 核验 validation-1531/test-14042 的 canonical identity 交集为 0，并如实记录 sanitized-content overlap；
- 用 fresh validation trajectory 生成三个 variant 的全支持域 score、rank、top-k、coverage 与 ABSTAIN；
- C0、C1 必须是 outcome-root 不可见的独立进程；C1 完成前不得把 historical outcome 加载进同一分析进程；
- C0 产出 test metadata manifest、identity/content-overlap receipt、refreshed blind-existence receipt 与 exact historical source allowlist receipt；C1 write-once 产出 `baseline_width4_atlas.json`、三个 variant report、`selector_freeze.json` 与 `selector_freeze_receipt.json`，并独立从 trajectory 重算验真；
- C1 receipt 必须绑定 card、source/pool/trajectory/analysis implementation hashes，且明确 `outcome_fields_consumed=false`、`variant_selection_performed=false`、historical source 未加载。

### C2

- 只有 C1 freeze receipt/hash 闭合后才激活 historical-13 exact value allowlist；
- C2 必须在新进程中先验 C1 freeze，再只读 Phase 1 manifest 唯一绑定的 baseline 与 historical-13 `results.json`/referenced `samples_*.jsonl`；聚合 `selection_report.json` 和既有 paired-analysis summary 不得作为 outcome source；
- blind-12 value roots 在整个 P3-C 始终禁止；
- 每个 baseline/historical source 必须精确闭合 14,042 ordered identities、model/tokenizer revision、K=2 fixed-horizon recipe、绝对路径与文件 hash；`paired_comparison()` 的 intersection fallback 不得代替 exact identity closure；
- 按第 3.6 节选择唯一 variant，不重算 baseline score；输出 `retrospective_adjudication.json`、冻结的 missing-12 predictions 与 verifier receipt。

若不是 `RETROSPECTIVE_TOP1_RECOVERY`，不得自动进入 P3-D，也不得在当前卡内调公式。

### 外部动作与 repair 边界

- 允许 SSH/HPC2 只读 source inspection、metadata-safe projector、C0/C1/C2 CPU 分析；若 C1 bootstrap 不适合 login node，可提交合理 CPU Slurm job，长等待必须用 executor-owned automation/heartbeat，不得人工轮询；
- 禁止 GPU、模型 forward、loop/K2 新实验、缓存下载或写入、Blind-12 outcome、P3-D 及后续 Gate；
- outcome 解封前，新增三文件范围内最多两次低风险实现修复；C1 freeze/C2 解封后不得自行修改科学实现或通过重跑寻找有利结论，发现材料性缺口直接 `BLOCK`。

### Terminal

只允许 `GATE_P3C_FINAL_AUDIT` 或 `BLOCK`。

## 8. P3-D 与 P3-E 计划授权包

### P3-D：missing-12 engineering smoke

- 固定 12 个 blind cells；先 4-sample audit，再 `limit=5` engineering smoke；
- K=2 科学配方与 Phase 1 保持相同；
- 工程阶段不得按 accuracy、排序或答题结果做科学判断；
- no-code/no-control：发现必须改代码时 `BLOCK` 回规划线程；
- 按空闲资源使用 job array；只可调 throttle，不能删慢 cell 或改变配方。

P3-D `PASS` 只证明 12 个 cells 可同构 full 运行，不授权 P3-E。

### P3-E：blind full 与 all-25 adjudication

- exactly missing-12 K=2 full；不得根据 smoke/partial results 减少或增加 cells；
- missing-12 全部完成前禁止查看部分科学 outcome；
- 完成后一次性解封并运行 frozen all-25 analyzer；
- 报告第 3.7/3.8 节 primary endpoint、labels 与全部 secondary paired analysis；
- 有效科学样本产生后不得自动 retry/requeue/resubmit；scheduler-only replacement 仅在作业未开始、无科学输出且 handoff 预授权时允许。

P3-E 只允许 `GATE_P3E_FINAL_AUDIT` 或 `BLOCK`。负科学结果在 evidence 完整时仍可工程 `PASS`。

## 9. P3-F 与 P3-G 条件分支

这两个 Gate 不是 P3-E 的自动后续。

只有 P3-E 独立审计 `PASS`，且 science label 为 `WIDTH4_WITHIN_CELL_FEASIBILITY_SUPPORTED` 或经用户明确接受的 `WIDTH4_RANKING_SIGNAL_ONLY`，规划线程才可另立 variable-width card。

### P3-F

- 用户已明确要求顺序完成两套结果；active card=`H3_VARIABLE_WIDTH_DUAL_RANKING_V1`/SHA256 `b99352c4...`，旧 card `a793a320...` 保留但不再执行。
- 复用 P3-B validation-1531 的既有 `B_0...B_28` trajectory；不得重新运行模型、数据集 forward、GPU 或 Slurm。
- 第一张表严格保留原 CONSENSUS：只含具备完整 `s±1、s±w` 的 98 个窗口，不写 unsupported/ABSTAIN 行；score=`min(z_OH,z_OK,z_FH,z_FK)`。
- 第二张表覆盖全部 168 个窗口：`R(s,w)` 取 `s-1,s+1,s-w,s+w` 中所有合法同宽起点，数量 2–4；`A_H/A_K` 比较中心与该集合均值，edge-aware score=`min(z_AH,z_AK)`。它必须单独命名，禁止与 strict score 数值混排或宣称为原 CONSENSUS。
- 两套规则都使用每层归一化 `E_rate/K_rate/CEdrop_rate` 和同一 subject-stratified joint bootstrap（2,000，seed `20260716`）；各自全局排序固定为 score 降序、width 升序、start 升序。
- width=4 的 17 个 score、`12:15=35.83923516567705`、唯一 eligible 与 selection frequency=`1.0` 必须精确向后兼容；这是进入全宽度结果解释的关键检查。
- 只做最小 sidecar、targeted tests、一次 write-once 真实分析和一次独立重算；不重复 full suite、旧 Gate 审计、篡改矩阵或 receipt 扩建。
- 输出严格 98 JSON/CSV、edge-aware 168 JSON/CSV、一个双排名 summary、一个独立 verifier；P3-F 不读取任何 accuracy/correctness/answer flip，不自动授权 P3-G。

### P3-G

- 只运行 P3-F frozen shortlist 中最多 4 个新 K=2 full cells；
- 资源不设 GPU/时间硬上限，但科学 cells 上限保持冻结；
- 结果只能支持 targeted variable-width feasibility，不能声称所有宽度 global optimum；
- 有效科学样本产生后不得自动 retry。

## 10. 资源、远程与保护边界

### 10.1 资源政策

第三阶段不设置 GPU 数、GPU-hours 或墙钟硬上限。任何 GPU Gate 必须先满足：

1. 前一 Gate planning audit `PASS`；
2. 当前科学矩阵与 manifest 已冻结；
3. 新 executor 与 external action 权限已在 control/handoff 中绑定；
4. 当前 HPC queue/node/partition/QoS/account 合法且有合理空闲资源。

满足后应尽快提交。资源变化只能调整 shard/job-array 并发，不得改 science。

### 10.1.1 长时作业 automation 规则

当前及后续所有 Phase 3 executor 在提交 HPC2/GPU/Slurm 作业后，只允许做一次有界提交确认；若作业仍为 `PENDING/RUNNING` 或预计无法在当前交互轮次内结束，必须在该 Gate executor 线程中创建真实的 Codex automation/heartbeat，随后停止人工轮询并允许线程 idle。cadence 依据预计时长使用约 10/30/60 分钟；每次唤醒只做一次有界只读检查并结束，状态无材料变化时不通知。monitor prompt 必须冻结 executor/thread、Gate、host、job IDs、run root、只读命令、允许查看的日志/工件与 terminal 判据。

automation 只有 read-only monitoring 权限，不继承 retry、requeue、resubmit、cancel、repair、scheduler throttle、写工件、改参或科学裁决权限。terminal success/failure、SSH trust 异常或材料性 blocker 出现时，它只将证据送回原 executor；executor 恢复后先暂停或删除 monitor，再执行已授权的验证/终态流程。planning/audit thread 不做 heartbeat polling。若当前 Codex 线程没有 `automation_update` capability，executor 必须在开始长等待前发送 `BLOCK`，不得用 sleep/循环轮询或远端常驻 monitor 冒充 Codex automation。

“送回原 executor”必须是可验证的续跑握手，不是被动状态留言。monitor prompt 必须同时冻结 planning thread、executor thread、Gate、job/run root、terminal 后下一项已授权动作及 fallback recipient。terminal wakeup 的 final answer 本身不等于 executor 已恢复：monitor 在一次有界只读检查后必须先暂停/删除自身，再通过可用的 thread-message capability 向精确 executor 发送 `AUTOMATION_TERMINAL_RESUME` 并保留投递成功回执；禁止只写“executor 可恢复”后结束。若不能触发或确认该 executor 新一轮，必须向 planning thread 发送并在自身输出完整 `AUTOMATION_RELAY_REQUIRED`，由 planning thread 只做一次有界 relay recovery；不得重跑作业、创建新 executor 或把该恢复动作变成轮询。

executor 的 `GATE_P3X_FINAL_AUDIT` / `BLOCK` 同样只有在精确 planning thread 的投递得到可见确认后才算已送达；无法确认时必须输出带完整终态包的 `TERMINAL_DELIVERY_UNCONFIRMED`，不得静默 idle 或声称已经报告。

为避免在 P3-B executor 活跃期间改变 `loopscope-tflt` 的 HEAD/dirty provenance，本次先由本 control 与 superseding handoff 立即生效；P3-B 终态审计完成并撤销 executor 写权限后、任何 P3-C 授权之前，planning thread 必须把同一续跑握手并入 `loopscope-tflt/AGENTS.md`，以单独 governance commit 推送并更新下一 Gate base。该落地是 P3-C 的治理 admission 条件。

### 10.2 远程根

```text
HPC2 dedicated repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
LoopScope workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
expected Phase 3 run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-<gate>-<UTC>/
audited venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711
```

每次 run 使用新的 timestamped write-once sibling root；不得覆盖或移动历史 run。

### 10.3 Protected state

- Phase 1 canonical run、Phase 2 Gate roots、历史 controls/reports 与 reproduction checkout 永久只读；
- lm-eval framework/source/task/YAML/renderer/batching/metric 冻结；
- `wrapper.py、strategies.py、cache.py、config.py` 默认禁止修改；
- `.planning/*control.md` 与 `AGENTS.md` 只由当前 planning/audit thread 更新；executor 禁止修改；
- 禁止 amend/rebase/reset/force-push、删除/覆盖工件或把 LoopScope 自动 merge 回 `main`。

## 11. Repair、retry 与 terminal 协议

### 11.1 Repair

普通 Gate 默认：

```text
executor-owned low-risk repair loops <= 2
bundled audit-returned repair cycles <= 1
```

只允许修复同一可复现的 engineering root cause，且不得改变 frozen science、权限、数据或 external side effects。model/task/split/sample/window/K/metric/threshold/ranking/manifest semantics、保护代码或资源权限变化必须 `BLOCK` 回规划线程。

各 Gate 的专门 retry 规则优先于通用 repair。科学 run 一旦产生有效 outcome，不得静默重跑。

### 11.2 Event-driven terminal

每个 Gate 只绑定一个独立 executor。规划线程发出 handoff 后不轮询、不提前创建下一 Gate executor。

executor 终态必须发送一次：

```text
GATE_P3X_FINAL_AUDIT
```

或：

```text
BLOCK
```

terminal packet 至少包含 executor、Gate、授权/最终 commit、dirty、files changed、exact commands/exit codes、job IDs/run root/artifact hashes、结果、偏差、repair 次数、未执行动作、保护状态与请求决定。

只有 planning/audit thread 可以作 `PASS / PASS_WITH_FIXES / BLOCK`。`PASS` 后只解除当前 executor 的 Gate/外部动作授权，不归档、不关闭，线程继续保留在本项目中作为可追溯执行记录；下一 Gate 必须使用新的 executor 与新的显式授权。

## 12. 最低 evidence 契约

根据实际 Gate 至少生成或闭合：

```text
phase3_card.json
phase3_card_verifier_receipt.json
source_manifest.json
validation1531_renderer_source_receipt.json
validation1531_pool.jsonl
validation1531_pool_manifest.json
validation1531_forbidden_field_scan_receipt.json
validation1531_no_loop_smoke_report.json
validation1531_shard_manifest.json
validation1531_no_loop_probe_report.json
validation1531_no_loop_probe_verifier_receipt.json
validation1531_shard_merge_receipt.json
validation1531_vs_test14042_disjointness_receipt.json
legacy512_subset_robustness.json
baseline_width4_atlas.json
shift_overlap_report.json
nonoverlap_flank_report.json
consensus_report.json
selector_freeze.json
selector_freeze_receipt.json
retrospective_adjudication.json
blind12_full_source_manifest.json
width4_all25_analysis.json
width4_all25_verifier_receipt.json
p3b_scheduler_throttle_resource_accounting.json
p3d_scheduler_throttle_resource_accounting.json
p3e_scheduler_throttle_resource_accounting.json
p3f_scheduler_throttle_resource_accounting.json
p3g_scheduler_throttle_resource_accounting.json
```

每个工件按风险绑定 repo/branch/commit/dirty、model/tokenizer/dataset revision、producer namespace、sample/candidate manifest hashes、exact argv/config、run root/job ids、timestamps、source hashes、analysis implementation hash 与 verifier result。

Scheduler receipt 必须 write-once 记录 pre-launch manifest hash、每次 throttle 变化、前后 scientific argv/config hash 一致性，以及实际 partition/QoS/account、节点、requested/allocated GPU、排队/运行时间和 GPU-time。

## 13. 决策记录

| 日期 | Gate/范围 | 决定 | 依据 | 下一条件 |
|---|---|---|---|---|
| 2026-07-16 | Phase 2 closeout | Gate D `PASS`；Phase 2 supplemental complete | Phase 2 control 与终态报告 | 新 hypothesis card |
| 2026-07-16 | trajectory population | primary 改为 MMLU validation 全部 1,531；不补到 2,048 | canonical validation 仅有 1,531 unique targets；用户明确决定 | fresh no-loop acquisition 独立 Gate |
| 2026-07-16 | resource policy | 不设 GPU 数/GPU-hours/墙钟硬上限；获权并满足 admission 后按空闲资源尽快提交 | 用户明确决定 | 每 Gate 仍需独立 external-action 授权 |
| 2026-07-16 | information architecture | 科学报告精简；原详细计划内容转为 Phase 3 control；AGENTS current-control 指针切换 | 用户明确决定 | 本 control 成为当前动态权威 |
| 2026-07-16 | current authorization | 只创建 control 与更新治理指针；P3-A 及以后均未授权 | 本轮用户请求范围 | 用户明确授权 P3-A 并绑定 executor |
| 2026-07-16 | execution mandate | 按协议逐 Gate 推进至第三阶段科学终点；executor 统一 `gpt-5.6-sol / max`，审计仅阻断材料性问题 | 用户最新明确指令 | 每次只激活当前 Gate |
| 2026-07-16 | P3-A authorization | card/receipt 已冻结并推送；授权 executor `019f67b6-d57d-7203-822a-dc1dd58af603` 在本地实现 P3-A | `loopscope@ed9b31b`; card `5b5cb333...` | terminal packet 后独立审计 |
| 2026-07-16 | P3-A executor restart | 用户作废 `019f67b6-d57d-7203-822a-dc1dd58af603` 并要求新建 executor；旧线程无可采纳证据、无后续权限 | 用户最新明确指令；作废时 repo clean at `ed9b31b` | 绑定新的 5.6-sol-max executor |
| 2026-07-16 | P3-A replacement authorization | 授权新 executor `019f67bd-c4b6-72e1-a3a8-0aad8ec60879` 使用同一 frozen base/card 完成 P3-A | repo clean `ed9b31b`; card `5b5cb333...` | superseding handoff 后执行 |
| 2026-07-16 | executor subagents | 各 Gate executor 可适度使用最多 3 个 child subagents；不扩大 Gate/路径/外部权限，主 executor 保留唯一整合与终态责任 | 用户最新明确授权 | executor 自行按需并行 |
| 2026-07-16 | P3-A final audit | `PASS`；解除 executor `019f67bd-c4b6-72e1-a3a8-0aad8ec60879` 的 Gate 授权并保留线程 | 独立核验 local/origin clean `5c7b45f`、14 个授权路径、保护 diff/card hashes、29 targeted tests `OK`、verifier manifest `38081a...`；full suite 仅保留已接受的 Phase 2 历史 hash mismatch | 新建 P3-B executor |
| 2026-07-16 | executor retention | 完成的执行线程不归档、不关闭；解除授权后继续保留在项目中 | 用户最新明确指令 | 后续所有 Gate 一致执行 |
| 2026-07-16 | P3-B authorization | 绑定 executor `019f67e9-4e17-7b53-a1db-b0d673a15bff`，授权最小 live producer、B0 isolated safe projection、B1 pool freeze、B2 smoke、B3 validation-1531 no-loop acquisition 及 HPC2/GPU/Slurm adaptive sharding | clean local/origin `338809d`；P3-A PASS；renderer anchor、card、model/dataset revision、run root 与权限边界已冻结 | 等待唯一 `GATE_P3B_FINAL_AUDIT` 或 `BLOCK` |
| 2026-07-16 | long-job monitoring governance | 长时 HPC2/GPU/Slurm 等待禁止 executor 人工轮询；提交确认后改用 executor-owned Codex automation，10/30/60 分钟低频唤醒，每次仅一次有界只读检查 | 用户最新明确指令；既有 Gate E 低频 heartbeat 经验；research-gate long-job invariant；AGENTS commit `07a10de` 已推送且 superseding resume handoff 已送达 P3-B | 当前 executor 先核验新 base；获得真实 job id 后创建 monitor |
| 2026-07-16 | P3-B BLOCK audit | `PASS_WITH_FIXES`；授权同一 executor 使用首次 audit-returned repair cycle 做一个两文件、单 commit 的 accounting-lineage 修复；禁止重跑任何科学 acquisition | 独立核验 local/remote clean `a519091`；受保护 diff 空；`9988046` 与 `9989056_[0-31]` 全部 `COMPLETED/0:0`；verifier `PASS/1531/57/32` 且 outcome unread；resource receipt 现场 absent；代码确认 closed producer hash 被错误地与 current implementation hash 比较 | 只重跑 affected tests 与失败的 resource-receipt command；成功后返回 `GATE_P3B_FINAL_AUDIT`，否则 `BLOCK`；P3-C 继续锁定 |
| 2026-07-16 | P3-B final audit | `PASS`；撤销 executor `019f67e9-4e17-7b53-a1db-b0d673a15bff` 的全部执行权限并保留线程可见不归档 | 独立核验 local/remote clean `3e40c36`、两文件 repair diff、Phase 3 tests exit 0、accounting file `e6be5b6b...`/internal `e89abcc7...` PASS、lineage `44ae277→3e40c36`、33 行 Slurm 全部 `COMPLETED/0:0`、trajectory/verifier hashes 不变、monitor registry 无 P3-B 条目 | planning 完成 AGENTS handshake governance commit；P3-C 仍锁定，须另行显式授权 |
| 2026-07-16 | confirmed handoff governance | 在 P3-B 关闭后以单独 commit `ef783c3` 将 automation terminal self-wake、relay fallback 与 terminal delivery confirmation 写入 `AGENTS.md` | 仅 `AGENTS.md` 一处稳定规则变更；SHA256=`bd72c671...`；local/origin clean | P3-C prospective base=`ef783c3`；仍须独立授权并绑定新 executor |
| 2026-07-16 | P3-C authorization | 用户确认规划/审计线程应继续指导剩余实现与实验；绑定新 executor `019f69ae-1ce3-7303-9221-c7779d57c8c5`，按 C0→C1→C2 分阶段执行 metadata closure、outcome-blind selector freeze 与 historical-13 retrospective adjudication | clean local/origin `ef783c3`；P3-B PASS；card/P3-B/Phase 1 roots、write-once root、进程隔离与 exact outcome allowlist 已冻结 | 等待唯一 `GATE_P3C_FINAL_AUDIT` 或 `BLOCK`；P3-D 继续锁定 |
| 2026-07-16 | P3-C C0 BLOCK review | `PASS_WITH_FIXES` 并继续同一 Gate；用户明确以 no-loop trajectory 选窗预测为第一目标，按 `P3C_CONTENT_OVERLAP_POLICY_V2` 舍弃 sanitized-content 零交集 admission，同一 executor 获一次最小修复后直接 C0→C1→C2 | 独立核验 local/origin clean `1cc93f3` 且仅 3 个授权路径；C0 canonical identity overlap=0；28 个官方 split 内容重复仅占 validation 1.83%/test 0.20%，且 outcome 未读。该问题是数据限制而非身份泄漏 | 保持 1,531/14,042 总体；C0 描述实际 overlap，不以数量判停；只跑 targeted tests；随后完成 C1/C2 并返回唯一 terminal packet；P3-D 仍锁定 |
| 2026-07-16 | P3-C C0 list-window BLOCK review | `PASS_WITH_FIXES`；向同一 executor 追加一次 exact single-purpose runtime repair，随后继续 C0→C1→C2 | 独立核验 clean local/origin `b43d4b4`、3 个授权路径不变；失败为 `raw_window` 是 JSON list 时执行字典 membership 导致 `TypeError`，发生于 outcome 读取与 artifact 创建前；唯一正常修法是仅对 string window 做 blind-window membership | 只改 `phase3_p3c.py` 与对应 test；加一个 list-valued unrelated window 回归；只跑 P3-C targeted tests；一次 commit/push/ff-only sync；C0 一次后直接 C1/C2 |
| 2026-07-16 | P3-C final audit | `PASS`；science=`RETROSPECTIVE_TOP1_RECOVERY`，variant=`CONSENSUS`；撤销 P3-C executor 权限并保留线程 | 三项决定性核验：local/origin clean `1793df1` 且仅 3 个 P3-C 路径；远端 selector/adjudication hashes 与 packet 一致；CONSENSUS 唯一 eligible/top-1=`12:15`、frequency=1.0、排除 `4:7/6:9`，blind-12 outcome unread，预测已冻结 | 新建并绑定 P3-D executor；只做 missing-12 engineering audit/smoke，不读 accuracy |
| 2026-07-16 | P3-D authorization | 绑定 executor `019f6a24-06eb-7320-8405-4f4758ce7095`，使用 `gpt-5.6-sol / max` 执行固定 missing-12 的 4-sample audit 与 limit=5 smoke | P3-C PASS；clean base `1793df1`；frozen predictions `0b3f12c6...`；write-once root `phase3-p3d-20260716T085629Z` | 所有 12 cells 工程同构、identity/revision/recipe 闭合；不读取 accuracy；唯一 terminal packet；P3-E 锁定 |
| 2026-07-16 | P3-D final audit | `PASS`；撤销 P3-D executor 权限并保留线程 | 三项轻量验收：local/HPC2 clean `1793df1`；远端 summary hashes 精确；D1/D2 均 12/12、D2 57 tasks/285 samples 且 identity 等于 Phase 1 anchor。D2 verifier 在 identity 前因 env snapshot 未序列化 offline flags 而 false-negative，但 12 个 immutable launchers 均导出 flags，接受为非材料偏差 | 新建并绑定 P3-E executor；执行 missing-12 full 与一次性 all-25 分析 |
| 2026-07-16 | P3-E authorization | 绑定 executor `019f6a4f-e4c6-7170-add5-1c513c19aa16`，使用 `gpt-5.6-sol / max` 完成 exact missing-12 full、全完成后一次性解封并分析 all-25 | P3-D PASS；clean base `1793df1`；P3-C selector/prediction 与 P3-D 12-cell engineering closure 已冻结；run root `phase3-p3e-20260716T094417Z` | 禁止 partial outcome；12/12 full 后一次 analyzer/verifier；完整 evidence 即工程 PASS，科学标签如实报告；P3-F 仍条件锁定 |
| 2026-07-16 | P3-E final audit | `PASS`；撤销 P3-E executor 权限并保留线程；width-4 主线完成，science=`WIDTH4_RANKING_SIGNAL_ONLY` | 三项轻量验收：local/origin/HPC2 clean `2f96124` 且仅 3 个授权新增路径；job `9990349_[0-11]` 12/12 `COMPLETED/0:0`、每 cell 57 tasks/14,042 identities；analysis/verifier hashes 精确且 verifier PASS。`12:15` selector/outcome rank 均为 1，但 blind `G_blind=+0.01424 pp`、CI 跨 0，`7:10` 实际为负 | P3-F 不自动 admission；等待用户决定是否接受 ranking-signal-only 并另立 variable-width card |
| 2026-07-16 | Phase 3 width-4 consolidated audit | P3-A 至 P3-E 全部工程 `PASS`；当前规则保留 `12:15` 唯一观测最优并未接纳有害 point window，但没有筛出更优窗口，blind 富集不成立 | all-25 oracle=`12:15`、Δ=`+0.41305 pp`；accepted point-eligible=`[12:15]`；harmful accepted=`[]`；blind primary=`INCONCLUSIVE` | 结束当前 width-4 阶段；P3-F/P3-G 保持条件锁定 |
| 2026-07-16 | P3-F user decision and authorization | 用户明确接受 `WIDTH4_RANKING_SIGNAL_ONLY` 并要求继续计算全宽度总表；冻结 width 2..8 card；最初误绑定内部 executor，后由 visible-thread correction supersede | clean exact card commit `d3666dd`；card SHA256 `a793a320...`；local/HPC2/GitHub branch 均验证 exact commit；现有 trajectory 足以计算，无需新 forward | CPU-only 生成 168 行总表、98 行 global ranking 与 verifier；唯一 terminal packet；P3-G 锁定 |
| 2026-07-16 | P3-F no-ABSTAIN policy pause | 用户要求不使用 `ABSTAIN`；立即中断尚未交付 terminal artifact 的 P3-F executor 并撤销其当前权限 | 原 CONSENSUS 的双侧 same-width flank 只有 98 个支持窗；删除 70 行与为 70 个边缘窗重定义单侧/SHIFT score 是两个不同科学合同，不能静默替换 | 等待用户明确表格只保留 98 个原规则 scored rows，还是要求 168 个窗口全部采用新 edge-aware score |
| 2026-07-16 | P3-F dual-ranking superseding authorization | 用户要求先做原规则 98 个有分窗口，再做重新定义规则让 168 个窗口全部有分；冻结 dual card 并恢复同一未完成 executor | 新规则 `R(s,w)` 只使用可用的 `s±1,s±w` 同宽参照，2–4 个；edge-aware score 与 strict CONSENSUS 分开命名、分开排序；clean local/HPC2/GitHub base `7303f28` | 同一 Gate 顺序生成 strict98 与 edge168，两表共享 bootstrap；唯一 terminal packet；P3-G 锁定 |
| 2026-07-16 | P3-F visible-thread correction | 用户指出看不到 Gate F 执行线程；中断并作废内部 `/root/execute_loopscope_h3v1_phase3_gate_f`，新建并绑定可见项目线程 `019f6aea-2b1d-7103-9bef-8365eb15640b`，标题 `execute-LoopScope-H3V1-第三阶段-Gate F`，模型 `gpt-5.6-sol/max` | 内部 executor 已留下 clean pushed implementation commit `4130645`，但其自报和 child evidence 不采纳；可见 executor 以该 Git 状态为基线独立做最小核对、正式分析和 verifier | 只接受可见 executor 主动交付的唯一 terminal packet；线程完成后保留可见且不归档；P3-G 继续锁定 |
| 2026-07-16 | P3-F final audit | `PASS`；撤销 executor `019f6aea-2b1d-7103-9bef-8365eb15640b` 权限并保留可见不归档；在 outcome-blind 状态冻结 P3-G proposed shortlist=`7:9,2:9,13:15,3:7` | 轻量核验 local/HPC2 clean `4130645`、dual card `b99352c4...`、exact six-file root、strict98/edge168 rows 与 verifier `0afaef48...` PASS；strict top1=`7:9`，edge top1=`2:9`，两规则唯一 eligible 仍为 `12:15` | P3-G 仍锁定；等待用户独立授权最多四个新 K=2 full cells，不自动读取 outcome 或提交 Slurm |
| 2026-07-16 | P3-G user authorization and dispatch | 用户明确“开始gate G”；冻结 targeted-full card 并绑定新可见 executor `019f6b0a-2979-7662-b380-10bd3ed4b352`，模型 `gpt-5.6-sol/max` | card/base `b7e28ea`、SHA256 `ccc4a285...` 已在 local/HPC2/GitHub clean 闭合；exact cells=`7:9,2:9,13:15,3:7`；primary comparator=`12:15`；baseline secondary | targeted tests 后做四-cell limit=5 structural smoke，再尽快提交四-cell full；automation 监控；4/4 后一次 analysis/verifier；P3-G 为阶段终点 |
| 2026-07-16 | P3-G final audit and Phase 3 closeout | P3-G `PASS`；撤销 executor 权限并保留可见不归档；Phase 3 P3-A–P3-G 全部关闭；science=`TARGETED_VARIABLE_WIDTH_NO_IMPROVEMENT` | 轻量核验 local/HPC2 clean `684aff7` 且只改 3 个授权路径；job `9990976_[0-3]` 4/4 `COMPLETED/0:0`；三份 summary hashes 精确且 verifier `29bfa2d8...` PASS。四候选均显著差于 `12:15`，best `7:9` 仍低 `-0.733514 pp` | 阶段终点；没有 active authority；任何后续扩展须新 phase/hypothesis card |
| 2026-07-17 | Post-P3-X4 authorization | 用户明确要求测试单层15循环体的 K=2/3/4；绑定可见 executor `019f6bb3-8f65-7ea1-a2f9-425a7959680f`，模型 `gpt-5.6-sol/max` | Phase 3 已关闭且 repo clean `684aff7`；B14→B15 观测到 H −0.5113、KL +0.5438；冻结 window=`15:15`、三种 K、同一 Qwen3-1.7B/MMLU full/fixed-horizon 配方 | 最小 sidecar/targeted tests→三-cell limit=5 smoke→三-cell full；automation 监控；唯一 X4 terminal packet；不授权其他窗口或下一 Gate |
| 2026-07-17 | Post-P3-X4 final audit | `PASS`；撤销 executor 权限并保留可见不归档；无下一 Gate | 轻量验收确认 local/origin clean `fb5ead4` 且仅新增 4 个授权路径；远端分析记录 `9991667_[0-2]` 3/3 A40 `COMPLETED/0:0`、每 cell 57 tasks/14,042 identities；独立 verifier=`PASS`。K2/K3/K4 accuracy 分别为 62.71899%/62.51246%/62.49822%，均未超过 `12:15@K2` | 单层 15 的 B14→B15 反转不足以产生收益；保留“需要更宽上下文窗口”的解释；终止 X4，不自动开展 X5、其他窗口或跨模型实验 |

## 14. 当前 blocker 与下一动作

- 当前 verified state：P3-A 至 P3-G 全部工程 `PASS`；Phase 3 terminal complete。最终科学结论为当前规则恢复并保留 `12:15`，但 blind enrichment 不成立且可变宽度 targeted candidates 没有更优结果。
- 当前 executor：none；Post-P3-X4 executor `019f6bb3-8f65-7ea1-a2f9-425a7959680f` 与此前 executor 均已撤销权限并保留可见不归档。
- 当前 blocker：none。
- 当前授权：none。
- 当前 actions not authorized：其他 window/K、baseline 或 `12:15` 重跑、selector 修改、cross-model/cross-task/adaptive 扩展、有效 outcome 后重跑；均需新的用户授权。
- authority owner：用户与本 planning/audit thread。
- 下一 admission：none；任何 X5、其他窗口、cross-model/cross-task 或 selector 扩展都需要用户另行授权。
