# GATE_G_HANDOFF

```text
Project/phase=LoopScope Phase 5 post-terminal method-development extension
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f8f8d-89fd-7660-a915-1d0201374274
Authorized executor title=execute-LoopScope-Relative-Biphasic-Reversal-第5阶段-Gate G
Gate=G
Method=RELATIVE_BIPHASIC_REVERSAL_V1
State=AUTHORIZED
```

## 1. Objective and claim boundary

基于已经取得的两份 MMLU validation no-loop trajectory，按同一新版数学合同离线重算：

- `Qwen/Qwen3-1.7B-Base × MMLU 5-shot`；
- `Qwen/Qwen3-4B-Base × MMLU 5-shot`。

每个模型分别输出完整候选排名、旧 strict eligibility、soft-relative eligibility、
shared-biphasic eligibility、最终新版 eligible set，以及 selected window 或明确 `ABSTAIN`。
本 Gate 是由已见的 1.7B/4B MMLU outcome 启发的 retrospective method-development 与
falsification；两个 cell 都参与了规则形成。即使排除已知有害窗口或保留已知较好窗口，
也只能称为 retrospective coherence，不能称为 prospective validation、positive gain 或
通用自动选窗器。

本 Gate 不做模型 forward，不加载模型或数据集，不使用 GPU/CUDA/Slurm，不运行任何
loop/full-accuracy outcome，也不读取任何尚未观察的 outcome。

## 2. Authoritative sources and starting provenance

### Local control and code

- policy: `loopscope-tflt/AGENTS.md`
- active control: `.planning/loopscope_phase5_control.md`
- Phase 3 historical control: `.planning/phase3/loopscope_phase3_control.md`
- repository:
  `/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`
- branch: `loopscope`
- authorized base commit:
  `338bd22008135bc5d3df05aaa125b7c28e7ac783`
- protected reference ancestor:
  `4f59bd93eca4da3cbf458a93508f91c5b23912bc`
- expected starting dirty state: only the pre-existing untracked directory `报告/`, containing
  `报告/LoopScope_第五阶段多宽度Top4全量Outcome.md`; preserve it and do not stage, overwrite,
  delete, rename, or treat it as Gate G output.
- `origin/loopscope=f33a37a4e5c0cfc461184a3471277897c77cf83b`; local is already ahead by
  five historical commits. No push is authorized.

### Frozen cards and canonical inputs

Phase 3:

- `configs/loopscope/phase3_card.json`
  SHA256 `5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825`
- `configs/loopscope/phase3_variable_width_dual_card.json`
  SHA256 `b99352c415d9aa041f90d6d8c5b7cffd3771c549b348db36f10740998ec23612`
- trajectory:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z/validation1531_no_loop_trajectories.jsonl`
  SHA256 `fb7ae66a1a2696f46727c7ca5069ade2e2f9d6e181db03ed1d005cb2eed1525f`
- trajectory manifest:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z/validation1531_no_loop_trajectory_manifest.json`
  SHA256 `75a568c6f6595f74d34d1e86efc0f716934e3a2a1cf972daab7874e4afe89cf0`
- legacy strict table:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3f-dual-20260716T121430Z/strict_consensus_98_table.json`
  SHA256 `74332060f0b1da99202a7e647bc5b18a6733a992ed152889043c77b5c3cffb00`

Phase 5:

- `configs/loopscope/phase5_multiwidth_mid40_card.json`
  SHA256 `8c0029e34c87240a3afa58effbfe9045a4c49c6b27c1062a78b06b9ec423f769`
- trajectory:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z/formal/validation1531_phase5_trajectories.jsonl`
  SHA256 `aa4894e632f557ddc4614e88d06ec1fc4c525463be7531aef69fa3c28671edf9`
- trajectory manifest:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z/formal/validation1531_phase5_trajectory_manifest.json`
  SHA256 `343689e4d2c4d8bf40f2237233a5333f1a7d53cef219e226e56d017d5aad21c6`
- legacy central-40 table:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-e-multiwidth-mid40-20260723T100836Z/phase5_multiwidth_mid40_scores.json`
  SHA256 `a4bd8b7be3b531b89961667b1cbfba9db0f4020e01bfa8efc97a3ccd43826e44`

Both trajectory populations must close at exactly `1,531` unique validation identities,
`57` subjects, and respectively `29` (`B0…B28`) / `37` (`B0…B36`) boundaries. The main
analyzer may read only identity, subject, boundary ID, choice entropy, and KL-to-final.

## 3. Frozen candidate domains

The common depth rule is frozen before analysis:

```text
trim_per_side(L)=ceil(0.30*L)
central_blocks(L)=[trim_per_side(L), L-trim_per_side(L)-1], inclusive
publish W(s,w) only when every decoder block s…s+w-1 is inside central_blocks(L)
widths={3,4,5,6}
```

This gives:

| Model | L | Central blocks | w3 starts | w4 starts | w5 starts | w6 starts | Total |
|---|---:|---|---|---|---|---|---:|
| Qwen3-1.7B-Base | 28 | `9…18` | `9…16` (8) | `9…15` (7) | `9…14` (6) | `9…13` (5) | 26 |
| Qwen3-4B-Base | 36 | `11…24` | `11…22` (12) | `11…21` (11) | `11…20` (10) | `11…19` (9) | 42 |

The four relative references `s-1,s+1,s-w,s+w` may fall outside the central published
domain but must be legal same-width windows in the full model. Shape-support transitions may
use at most one immediately adjacent transition outside the chosen window, as frozen below;
they may not leave `B0…BL`.

Diagnostic-only 4B windows `4:7` and `5:8` are outside the published candidate domain. They
must be computed separately for retrospective falsification and must never enter the 42-row
ranking or selection set.

## 4. Frozen slope, net trend, and relative score

For identity `i` and native boundary transition `j -> j+1`:

```text
gH_i,j = H_i,j - H_i,j+1      # positive means entropy decreases
gK_i,j = D_i,j+1 - D_i,j      # positive means KL-to-final increases
```

For inclusive decoder-layer window `W(s,w)=[s,s+w-1]`, corresponding to
`B_s -> B_(s+w)`:

```text
G_m(s,w) = mean_i[(1/w) * sum_{j=s}^{s+w-1} g^m_i,j],  m in {H,K}
```

Thus `G_H=E_raw/w` and `G_K=K_raw/w`. `NetPositive` requires strict
`G_H>0 and G_K>0`; equality fails.

The legacy relative ranking remains unchanged:

```text
O_m = G_m(s,w) - 0.5*(G_m(s-1,w)+G_m(s+1,w))
F_m = G_m(s,w) - 0.5*(G_m(s-w,w)+G_m(s+w,w))
z_X = X / max(SE_boot(X), 1e-12)
S_REL = min(z_OH,z_OK,z_FH,z_FK)
```

All published candidates are ranked by:

```text
S_REL descending, width ascending, start ascending
```

Raw `S_REL` values are model-local and are not numerically calibrated across the two models.

## 5. Legacy strict and Soft Relative Eligibility

`legacy_strict` is diagnostic only and must reproduce the historical artifact:

- `G_H>0`, `G_K>0`;
- all four same-width reference windows have both `G_H<0` and `G_K<0`;
- the original percentile-bootstrap lower 95% CI is strictly positive for
  `O_H,O_K,F_H,F_K`.

`SoftRelativeStable` removes only the absolute reference-sign veto:

- `NetPositive`;
- original percentile-bootstrap lower 95% CI is strictly positive for all of
  `O_H,O_K,F_H,F_K`.

The original CI contract is retained for backward compatibility: sort the `R=2000`
bootstrap estimates and linearly interpolate at `q*(R-1)` for `q=.025,.975`.

## 6. Shared Biphasic Reversal

For every change point `tau=s+k`, `k=1…w-1`, and orientation `q in {+1,-1}`, compute
per identity first:

```text
L_i^m(tau) = mean_{j=s}^{tau-1} g_i,j^m
R_i^m(tau) = mean_{j=tau}^{s+w-1} g_i,j^m
```

Then aggregate across identities. A pair `a=(q,tau)` has these 12 one-sided direction
margins:

```text
segment:
  q*L_H, q*L_K, -q*R_H, -q*R_K

two-transition core:
  q*gH_(tau-2), q*gK_(tau-2)
  q*gH_(tau-1), q*gK_(tau-1)
 -q*gH_tau,     -q*gK_tau
 -q*gH_(tau+1), -q*gK_(tau+1)
```

Every scalar above is computed per sample and then aggregated. Judging only the mean
trajectory curve is forbidden.

The core support is fixed as follows:

- `w3`: `k=1` uses one left external transition; `k=2` uses one right external transition.
- `w4`: `k=2` is wholly internal; `k=1/3` use the corresponding one-step external support.
- `w5`: `k=2/3` are wholly internal; `k=1/4` use the corresponding external support.
- `w6`: `k=2/3/4` are wholly internal; `k=1/5` use the corresponding external support.

Record `boundary_supported`, which side is external, and exact supporting transition IDs.
If any required transition is outside `0…L-1`, the pair is unscorable. Do not pad, clamp,
interpolate, or drop a margin.

Diagnostic strengths:

```text
T_seg_m  = q*(L_m-R_m)
T_core_m = q*((g_m(tau-2)+g_m(tau-1))/2
              -(g_m(tau)+g_m(tau+1))/2)
```

For each pair and each of its 12 margins `r`:

```text
z_shape(a,r) = theta_hat(a,r)/max(SE_boot(a,r),1e-12)
S_SHAPE(a)   = min_r z_shape(a,r)
```

`S_SHAPE` is used only to locate/diagnose the internal change point. It is never averaged,
added, multiplied, or otherwise combined with `S_REL`.

The point pair `(q*,tau*)` is the unique maximizer of `S_SHAPE` over all scorable
`q,tau` pairs in that window. A point tie under `rel_tol=abs_tol=1e-12` yields
`INELIGIBLE_TAU_AMBIGUOUS`.

## 7. Bootstrap and simultaneous shape stability

Each model retains its canonical subject-stratified joint sample bootstrap:

| Model | R | Seed | PRNG/API | Joint reuse |
|---|---:|---:|---|---|
| 1.7B | 2000 | 20260716 | `random.Random` / `randrange(n_subject)` | same replicate index map for all H/K, layers, windows, widths, q/tau |
| 4B | 2000 | 20260722 | `random.Random` / `randrange(n_subject)` | same replicate index map for all H/K, layers, windows, widths, q/tau |

Canonical subject order and within-subject record order must be preserved. Standard errors
use `ddof=1`. Do not share draw indices across models and do not choose a seed using outcomes.

For each window independently, form a one-sided 95% simultaneous max-t lower bound across
every scorable `(q,tau)` and all 12 margins:

```text
e_b(a,r) = (theta_hat(a,r)-theta_b(a,r))/max(SE_boot(a,r),1e-12)
M_b      = max over all scorable (a,r) in this window of e_b(a,r)
c_0.95   = linear empirical quantile of {M_b} at .95
LCB_sim(a,r) = theta_hat(a,r)-c_0.95*max(SE_boot(a,r),1e-12)
```

This is a within-window search correction; it does not pool candidate windows or models.
`BiphasicDirectionStable` requires all 12 `LCB_sim(q*,tau*,r)>0`; equality fails.

For exact-pair stability, each bootstrap replicate recomputes all 12 margins and
`S_SHAPE_b(a)=min_r theta_b(a,r)/max(SE_full(a,r),1e-12)`, then re-searches all scorable
`q,tau`. A replicate tie under the frozen tolerance counts as no exact-pair match.

```text
tau_pair_frequency =
  count_b(unique argmax_a S_SHAPE_b(a) == (q*,tau*)) / 2000
```

`BiphasicStable` requires:

- unique point `(q*,tau*)`;
- all 12 simultaneous lower bounds strictly positive;
- `tau_pair_frequency >= 0.80` (equality passes).

The heterogeneity diagnostic is:

```text
pi(q,tau) =
  fraction of identities for which
  q*L_i^H>0, q*L_i^K>0, -q*R_i^H>0, -q*R_i^K>0 simultaneously
```

`pi` has no V1 threshold and never changes eligibility.

## 8. New eligibility, selection, and ABSTAIN

```text
NewEligible =
  Scorable
  AND NetPositive
  AND SoftRelativeStable
  AND BiphasicStable
```

Only `NewEligible` windows enter selection. Rank publication still covers all central
candidates.

If the eligible set is non-empty, choose the unique point top-1 by `S_REL`. A point tie under
`rel_tol=abs_tol=1e-12` yields `ABSTAIN_NO_UNIQUE_TOP1`.

Global window selection frequency is separate from `tau_pair_frequency`. Over the fixed
point-eligible set, every bootstrap replicate recomputes `S_REL_b` using the original
full-sample SE denominators. Replicate order uses score descending, width ascending, start
ascending. The frequency is the fraction of replicates in which the point top-1 window is
replicate top-1.

Final decision:

- no `NewEligible`: `ABSTAIN_NO_NEW_ELIGIBLE`;
- point top tie: `ABSTAIN_NO_UNIQUE_TOP1`;
- top window selection frequency `<0.80`: `ABSTAIN_WINDOW_UNSTABLE`;
- otherwise: publish `selected_window` with both window and `(q*,tau*)`.

Never call rank-1, soft-relative-only, shape-only, or selected status a positive-gain result.

## 9. Required structural checks

1. Synthetic negative control: a single-step H/K zigzag without two consecutive transitions
   on both sides must fail `BiphasicStable`.
2. Synthetic positive control: two consecutive aligned H/K transitions on each side of a
   shared change point must pass when all other frozen conditions are satisfied.
3. Phase 3 backward compatibility: for all 26 central candidates, reproduce legacy strict
   `G/O/F`, four z values, `S_REL`, eligibility and failure reasons from the canonical
   `strict_consensus_98_table.json`, absolute tolerance `1e-10`.
4. Phase 5 backward compatibility: reproduce all 42 Gate E rows and legacy fields exactly,
   absolute tolerance `1e-10`.
5. 4B `13:17`: publish every `q,tau` segment sign and core-support result, not only the best
   pair.
6. Diagnostic-only 4B `4:7` and `5:8`: compute the full new rule. If either is
   `NewEligible`, record `RETROSPECTIVE_FALSIFICATION_FAILED_HARMFUL_DIAGNOSTIC_ADMITTED`.
   This is a scientific failure of V1, not an engineering Gate failure.
7. Development-support windows `4B 15:18`, `4B 15:19`, and `1.7B 12:15` may be discussed
   only as `retrospective_coherence`; they are not validation targets or acceptance
   requirements.

The selector analyzer must accept no outcome file/path/field. The scientific table and
manifest must contain no accuracy, correctness, prediction, gain, flip, gold, or test fields.
A final narrative may use only the already observed development registry stated in this
handoff; it may not open any outcome artifact.

## 10. Allowed implementation and execution

Authorized repository writes, and no others:

- `configs/loopscope/phase5_relative_biphasic_reversal_card.json`
- `src/tflt/loopscope/phase5_relative_biphasic.py`
- `scripts/loopscope/run_qwen_phase5_gate_g.py`
- `tests/test_loopscope_phase5_relative_biphasic.py`

Authorized external report:

- `报告/LoopScope_第五阶段相对双相反转双模型离线重选窗.md`

Authorized remote:

- host `hpc2-hkustgz`;
- bounded project-scope read-only SSH;
- read-only access to the canonical paths in section 2;
- stage only the four authorized implementation/card files under the fresh run root;
- HPC2 login-CPU analysis and independent verifier;
- no mutation of the HPC2 source checkout is required or authorized.

Frozen write-once run root:

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-g-relative-biphasic-dualmodel-20260723T151428Z
```

Authorized Git:

- one local single-purpose commit containing only the four repository paths above;
- do not add the pre-existing `报告/` directory;
- no push, fetch, pull, rebase, merge, force operation, branch creation, or remote checkout
  mutation.

## 11. Required outputs

Under the exact run root:

- `qwen17_relative_biphasic_candidates.json` and `.csv`: exactly 26 published rows;
- `qwen4_relative_biphasic_candidates.json` and `.csv`: exactly 42 published rows;
- `qwen4_relative_biphasic_diagnostics.json`: complete `4:7`, `5:8`, `13:17`,
  `15:18`, `15:19` checks, including every pair for `13:17`;
- `relative_biphasic_cross_model_summary.json`;
- `relative_biphasic_summary_zh.md`;
- `relative_biphasic_verifier_receipt.json`;
- `relative_biphasic_manifest_receipt.json`.

Every published candidate row must include:

- model, width, start, window, candidate-domain identity;
- `G_H,G_K,O_H,O_K,F_H,F_K`, four SE/z/percentile CI values, `S_REL`, global and
  within-width rank;
- legacy strict flag/failures, NetPositive, SoftRelativeStable/failures;
- every scorable `(q,tau)` with support transitions, boundary support, 12 point margins,
  12 SE/z/simultaneous lower bounds, `T_seg_H/K`, `T_core_H/K`, `S_SHAPE`,
  `tau_pair_frequency`, and `pi`;
- point-best `(q*,tau*)`, ambiguity flag, BiphasicStable/failures;
- `NewEligible`, complete failure reasons;
- selected flag and model-level window selection frequency where applicable.

The summary must separately report for each model:

- complete ranking;
- legacy strict / soft relative / biphasic / NewEligible counts and sets;
- rank top-1 versus selected;
- exact ABSTAIN reason or selected window;
- `tau_pair_frequency` and global window selection frequency;
- retrospective falsification status;
- explicit cross-model comparability boundary.

## 12. Engineering verification and acceptance

Targeted critical-path verification:

1. `python -m py_compile` for the analyzer, launcher, and test file.
2. The new targeted unit test module.
3. Existing Phase 3 variable-width and Phase 5 Gate E targeted tests only where imports and
   compatibility make them directly relevant; do not run repeated full-suite audits.
4. `git diff --check`.
5. Exact four-path Git diff and protected-path diff.
6. Remote formal analyzer exactly once after input/hash/admission closure.
7. Independent remote verifier exactly once.

Must pass:

- exact branch/base/dirty-state admission;
- both input hashes, population counts, subjects, boundaries, candidate counts and bootstrap
  contracts close;
- 26/42 tables complete with no missing/duplicate/extra candidate;
- synthetic negative/positive controls pass;
- all Phase 3/5 legacy overlap checks reproduce at `1e-10`;
- analyzer consumes no outcome/test/gold/correctness fields;
- independent verifier recomputes candidate membership, formulas, bootstrap digests,
  max-t bounds, pair/window frequencies, eligibility and final decisions from canonical
  trajectories;
- local commit contains exactly the four authorized paths;
- protected paths and all Phase 1–5 canonical artifacts remain unchanged.

Scientific V1 may fail retrospective falsification while the engineering Gate passes, provided
the computation is exact and the failure is reported without repair or threshold changes.

## 13. Agility and repair budget

Smallest increment: one dual-model scalar analyzer, one versioned card, one CLI, one focused
test module. Reuse existing trajectory readers and bootstrap primitives where practical; do not
build a general hook/bootstrap framework.

Earliest real computation: after targeted tests and exact input-hash preflight, run one
login-CPU formal analysis, then one independent verifier. Stop once outputs and verifier close;
do not add extra hardening or historical Gate re-audits.

Repair budget:

- at most two narrow executor-owned pre-formal engineering repair loops;
- no formula, threshold, seed, candidate, diagnostic registry, or claim-boundary change;
- after formal outputs exist, do not overwrite or outcome-tune them; a material scientific or
  identity conflict must emit `BLOCK`;
- at most one bundled planning-audit-returned repair cycle.

## 14. Explicitly forbidden

- model/tokenizer/dataset loading; any forward;
- CUDA/GPU/Slurm;
- new loop or full-accuracy outcome;
- opening any Gate C/D/F or Phase 3 outcome payload;
- using outcome to tune formulas, thresholds, domains, orientation, change-point search, or
  report inclusion;
- adding `S_SHAPE` to `S_REL`;
- using `pi` as a threshold;
- modifying original Phase 3/5 cards, trajectories, selectors, outcomes, reports, run roots,
  `AGENTS.md`, or protected runtime files;
- publishing or pushing;
- automatically entering a later Gate.

## 15. Terminal requirement

The executor must send exactly one structured `GATE_G_FINAL_AUDIT` or `BLOCK` to planning
thread `019f8604-4717-7be2-8bf8-9d4a26a3d7f7`, retain visible delivery confirmation, and then
stop. It may not self-declare `PASS`.

The terminal packet must include exact thread identity, branch/commit/dirty state, four-path
diff, commands/tests/exit codes, remote run root, all artifact hashes, 26/42 closure,
per-model selected/ABSTAIN decision, eligible counts, pair/window frequencies, diagnostic
falsification result, deviations, repair loops, protected-state confirmation, and request
`PASS / PASS_WITH_FIXES / BLOCK`.
