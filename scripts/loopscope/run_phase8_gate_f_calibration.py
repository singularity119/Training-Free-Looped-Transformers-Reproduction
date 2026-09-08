#!/usr/bin/env python3
"""Gate F outcome-free t0 reuse, cross-K preflight, and Qwen 1.7B K3 t1 fit."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from run_phase8_gate_e_calibration import diagnostics, write_json


def select_rows(pool, model, scope):
    rows = pool['rows']
    identities = [r['identity'] for r in rows]
    if len(rows) != 512 or identities != pool['calibration_identities'] or len(set(identities)) != 512:
        raise ValueError('formal pool must preserve 512 unique identities in order')
    if scope == 'FORMAL_CALIBRATION':
        return rows
    # Exactly four: two initial identities and two remaining longest prompts.
    chosen = set(identities[:2])
    for row in sorted(rows, key=lambda r: (-r['prompt_token_lengths'][model], r['identity'])):
        chosen.add(row['identity'])
        if len(chosen) == 4:
            break
    return [r for r in rows if r['identity'] in chosen]


def saved_collector(collector, metadata):
    return dict(identities=collector.identities,
                matrices={str(t): collector.matrix(t) for t in range(collector.k)},
                positions={str(t): [collector.positions[(key, t)] for key in collector.keys]
                           for t in range(collector.k)}, metadata=metadata)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('t0-from-saved', 'preflight', 't1'), required=True)
    p.add_argument('--cell-index', type=int, choices=range(4), required=True,
                   help='model index * 2 + window index; t1 accepts only 2 or 3')
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--saved-residuals', type=Path)
    p.add_argument('--t1-root', type=Path, help='preflight q17 t1 basis output directory')
    p.add_argument('--commit', required=True)
    p.add_argument('--scope', choices=('PREFLIGHT_ONLY', 'FORMAL_CALIBRATION'), required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.mode == 'preflight' and args.scope != 'PREFLIGHT_ONLY':
        p.error('preflight requires PREFLIGHT_ONLY')
    if args.mode == 't1' and args.cell_index < 2:
        p.error('only Qwen 1.7B K3 t1 acquisition is authorized')
    if args.mode == 't0-from-saved' and args.saved_residuals is None:
        p.error('t0-from-saved requires --saved-residuals')
    repo = Path(__file__).resolve().parents[2]
    config = json.loads((repo / 'configs/loopscope/phase8_model_windows.json').read_text())
    spec = config['models'][args.cell_index // 2]
    window = spec['windows'][args.cell_index % 2]['layers']
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, model=spec, window=window, scope=args.scope,
                              source_commit=args.commit, run_root=str(args.run_root))))
        return 0
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('calibration requires Slurm')
    if args.scope == 'PREFLIGHT_ONLY' and os.environ.get('SLURM_JOB_PARTITION') != 'debug':
        raise RuntimeError('preflight requires debug partition')
    from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
    pool = json.loads(args.pool.read_text())
    if pool['dataset'] != dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'):
        raise ValueError('calibration dataset mismatch')
    rows = select_rows(pool, spec['model'], args.scope)
    identities = [r['identity'] for r in rows]
    args.run_root.mkdir(parents=True, exist_ok=False)
    k = 3 if args.mode == 't1' else 2
    metadata = dict(scope=args.scope, model=spec['model'], revision=spec['observed_revision'],
                    window=window, k=k, dtype=spec['historical_runtime_dtype'],
                    cache_strategy=spec['cache_strategy'], alpha=1.0, dataset=pool['dataset'],
                    source_commit=args.commit, run_root=str(args.run_root), pool=str(args.pool))
    write_json(args.run_root / 'command_args.json', dict(metadata=metadata, argv=sys.argv,
               mode=args.mode, saved_residuals=str(args.saved_residuals) if args.saved_residuals else None))
    import torch
    from tflt.loopscope.phase8_runtime import (AnswerResidualCollector, Phase8Runtime,
                                             fit_basis, save_basis, load_basis, tensor_smoke_checks)
    from tflt.loopscope.phase8_gate_f_basis import fit_t0_basis, load_t0_basis
    torch.set_grad_enabled(False)
    torch.manual_seed(20260905)
    started = time.monotonic()

    def cell_for(target_k):
        return dict(model=spec['model'], revision=spec['observed_revision'], window=window,
                    dtype=spec['historical_runtime_dtype'], cache_strategy=spec['cache_strategy'],
                    k=target_k, alpha=1.0)

    def persist_t0(saved):
        expected = {key: metadata[key] for key in ('model', 'revision', 'window', 'dtype', 'cache_strategy', 'alpha', 'dataset')}
        expected['k'] = 2
        if any(saved['metadata'].get(key) != value for key, value in expected.items()):
            raise ValueError('saved source differs from requested cell recipe')
        basis = fit_t0_basis(saved, saved['metadata'], identities, args.scope)
        # Retain original residual provenance and the F fitting invocation separately.
        basis['fit_provenance'] = metadata
        save_basis(args.run_root / 'basis.json', basis)
        loaded = [load_t0_basis(args.run_root / 'basis.json', cell_for(target_k), args.scope)
                  for target_k in (2, 3, 4)]
        if any(b['direction'] != basis['direction'] for b in loaded):
            raise ValueError('cross-K load changed shared t0 direction')
        return basis

    if args.mode == 't0-from-saved':
        saved = torch.load(args.saved_residuals, map_location='cpu', weights_only=True)
        original_metadata = dict(saved['metadata'])
        original_identities = list(saved['identities'])
        if original_identities != pool['calibration_identities']:
            raise ValueError('saved formal source must preserve all 512 pool identities')
        position_path = args.saved_residuals.parent / 'positions.json'
        if position_path.exists():
            recorded_positions = json.loads(position_path.read_text())
            if [r['identity'] for r in recorded_positions] != original_identities:
                raise ValueError('saved scorer position identity order mismatch')
            for index, entry in enumerate(recorded_positions):
                if not entry['candidates'] or any(c['position'] != saved['positions']['0'][index] for c in entry['candidates']):
                    raise ValueError('saved residual position differs from actual scorer context')
        if args.scope == 'PREFLIGHT_ONLY':
            if original_metadata['scope'] != 'FORMAL_CALIBRATION':
                raise ValueError('saved subset preflight requires a formal historical source')
            indexes = [original_identities.index(identity) for identity in identities]
            saved = dict(metadata=dict(original_metadata, scope='PREFLIGHT_ONLY'), identities=identities,
                         matrices={t: matrix[indexes] for t, matrix in saved['matrices'].items()},
                         positions={t: [positions[i] for i in indexes] for t, positions in saved['positions'].items()})
        basis = persist_t0(saved)
        write_json(args.run_root / 'summary.json', dict(status='CALIBRATION_COMPLETE', mode=args.mode,
                   metadata=metadata, identities=identities, basis_path=str(args.run_root / 'basis.json'),
                   saved_residuals=str(args.saved_residuals), source_metadata=original_metadata, source_identity_count=len(original_identities),
                   source_positions_checked=position_path.exists(),
                   cross_k_load_verified=[2, 3, 4], row_count=basis['row_count'],
                   elapsed_seconds=time.monotonic()-started, target_gold_loaded=False, choice_scores_saved=False))
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tflt.config import LoopConfig
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase8_adapter import build_hflm_class, score_choices
    if importlib.metadata.version('lm_eval') != '0.4.11':
        raise RuntimeError('lm_eval version mismatch')
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], revision=spec['observed_revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(spec['model'], revision=spec['observed_revision'],
                local_files_only=True, torch_dtype=getattr(torch, spec['historical_runtime_dtype'])).to('cuda').eval()
    if model.config._commit_hash != spec['observed_revision']:
        raise RuntimeError('model revision mismatch')
    adapter = build_hflm_class(HFLM)(pretrained=model, tokenizer=tokenizer, batch_size='1', trust_remote_code=True)
    write_json(args.run_root / 'env.json', dict(source_commit=args.commit, model_revision=model.config._commit_hash,
               model_dtype=str(model.dtype), python=sys.executable,
               versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'lm_eval', 'datasets')},
               job_id=os.environ['SLURM_JOB_ID'], partition=os.environ.get('SLURM_JOB_PARTITION'),
               gpu=torch.cuda.get_device_name(), total_memory=torch.cuda.get_device_properties(0).total_memory))

    class CheckedRuntime(Phase8Runtime):
        """Validate actual model tensors on the tiny preflight intervention path."""
        def __call__(self, delta, t):
            result = super().__call__(delta, t)
            if t == 0:
                if result is not delta:
                    raise ValueError('t0 changed')
            else:
                position = self._active[1]
                raw = delta[0, position].float()
                v = self._device_direction
                expected = (raw - 0.5 * v * torch.dot(v, raw)).to(delta.dtype)
                if not torch.equal(result[0, position], expected):
                    raise ValueError('pre-answer damping differs from formula')
                if not torch.equal(result[:, :position], delta[:, :position]) or not torch.equal(result[:, position+1:], delta[:, position+1:]):
                    raise ValueError('other token changed')
            return result

    def acquire(target_k, runtime):
        adapter.phase8_runtime = runtime
        cfg = LoopConfig(model_alias=spec['model'], window=tuple(window), k=target_k, alpha=1.0,
              beta=0., strategy='damped_euler', iteration_mode='block', cache_strategy=spec['cache_strategy'],
              decode_mode='bypass', residual_transform=runtime)
        positions = []
        with looped_model(model, cfg):
            for i, row in enumerate(rows):
                scores = score_choices(adapter, row['prompt'], row['identity'])
                if len(scores) != 4 or not all(math.isfinite(float(s[0])) for s in scores):
                    raise ValueError('standard scorer returned nonfinite or incomplete choices')
                positions.append(dict(identity=row['identity'], candidates=adapter.phase8_positions))
                if (i+1) % 32 == 0:
                    print(json.dumps(dict(k=target_k, completed=i+1, total=len(rows))), flush=True)
        torch.cuda.synchronize()
        return positions

    if args.mode == 't1':
        collector = AnswerResidualCollector(identities, 3)
        positions = acquire(3, Phase8Runtime(collector=collector))
        saved = saved_collector(collector, metadata)
        basis = fit_basis(collector, metadata)
        save_basis(args.run_root / 'basis.json', basis)
        load_basis(args.run_root / 'basis.json', metadata)
        torch.save(saved, args.run_root / 'residuals.pt')
        write_json(args.run_root / 'positions.json', positions)
        write_json(args.run_root / 'diagnostics.json', diagnostics(saved['matrices'], basis, torch, 3))
        checks = dict(collector=collector.summary())
    else:
        checks = dict(tensor_checks=tensor_smoke_checks(torch), cross_k_t0={}, intervention={})
        collectors = {}
        for target_k in (2, 3, 4):
            collector = AnswerResidualCollector(identities, target_k)
            positions = acquire(target_k, Phase8Runtime(collector=collector))
            collectors[target_k] = collector
            current = saved_collector(collector, dict(metadata, k=target_k))
            torch.save(current, args.run_root / ('residuals-k%d.pt' % target_k))
            write_json(args.run_root / ('positions-k%d.json' % target_k), positions)
            equal = torch.equal(collectors[2].matrix(0), collector.matrix(0))
            same_positions = all(collector.positions[(key, 0)] == collectors[2].positions[(key, 0)] for key in collector.keys)
            checks['cross_k_t0'][str(target_k)] = dict(exact_equal=equal, positions_equal=same_positions)
            if not equal or not same_positions:
                raise ValueError('material t0 K dependence: shared direction cannot proceed')
        basis = persist_t0(saved_collector(collectors[2], metadata))
        for target_k in (2, 3, 4):
            loaded = load_t0_basis(args.run_root / 'basis.json', cell_for(target_k), args.scope)
            runtime = CheckedRuntime(loaded['direction'])
            acquire(target_k, runtime)
            if not runtime.calls or any(call['applied'] != (call['t'] >= 1) for call in runtime.calls):
                raise ValueError('wrong intervention timing')
            checks['intervention'][str(target_k)] = dict(call_count=len(runtime.calls), verified=True)
        if args.cell_index >= 2:
            t1_root = args.t1_root or args.run_root / 't1'
            t1_root.mkdir(parents=True, exist_ok=False)
            t1_meta = dict(metadata, k=3, run_root=str(t1_root))
            t1_basis = fit_basis(collectors[3], t1_meta)
            save_basis(t1_root / 'basis.json', t1_basis)
            load_basis(t1_root / 'basis.json', t1_meta)
            torch.save(saved_collector(collectors[3], t1_meta), t1_root / 'residuals.pt')
            acquire(3, CheckedRuntime(t1_basis['direction']))
            checks['q17_k3_t1_fit_load_intervention'] = True
        write_json(args.run_root / 'verification.json', checks)
    summary = dict(status='CALIBRATION_COMPLETE', mode=args.mode, metadata=metadata, identities=identities,
        checks=checks, elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        oom_count=0, max_prompt_tokens=max(r['prompt_token_lengths'][spec['model']] for r in rows),
        target_gold_loaded=False, choice_scores_saved=False)
    write_json(args.run_root / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
