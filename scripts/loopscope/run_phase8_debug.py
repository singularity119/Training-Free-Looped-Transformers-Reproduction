#!/usr/bin/env python3
"""Gate B only: eight-identity, one-model debug on the standard HFLM scorer."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-index', type=int, choices=(0, 1), required=True)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--commit', required=True, help='Clean source revision checked before submission')
    p.add_argument('--cell-index', type=int, choices=range(4), help='Split debug work if required by 29-minute limit')
    p.add_argument('--dry-run', action='store_true')
    return p


def finite_scores(scores):
    if len(scores) != 4 or not all(math.isfinite(x) for x in scores):
        raise ValueError('expected four finite raw choice likelihoods')
    return scores


def main(argv=None):
    args = parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    contract = json.loads((repo / 'configs/loopscope/phase8_model_windows.json').read_text())
    model_spec = contract['models'][args.model_index]
    cells = [(w['layers'], k) for w in model_spec['windows'] for k in contract['K']]
    if args.cell_index is not None:
        cells = [cells[args.cell_index]]
    recipe = dict(model=model_spec, cells=cells, alpha=1, batch_size=1,
                  rank=1, strength=0.5, scope='SMOKE_ONLY', source_commit=args.commit,
                  pool=str(args.pool), partition='debug', time_limit='00:29:00')
    if args.dry_run:
        print(json.dumps(recipe, indent=2))
        return 0
    if os.environ.get('SLURM_JOB_PARTITION') != 'debug':
        raise RuntimeError('Gate B model execution requires Slurm debug partition')
    pool = json.loads(args.pool.read_text())
    rows = pool['rows']
    if pool['status'] != 'SMOKE_ONLY' or len(rows) != 8 or len({r['identity'] for r in rows}) != 8:
        raise ValueError('Gate B requires exactly eight distinct SMOKE_ONLY identities')
    fit_rows = [r for r in rows if r['role'] == 'fit']
    verify_rows = [r for r in rows if r['role'] == 'verify']
    if len(fit_rows) != 4 or len(verify_rows) != 4 or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Gate B pool roles/split differ')
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json(args.run_root / 'command_args.json', dict(recipe, argv=sys.argv))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
    from tflt.config import LoopConfig
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase8_adapter import build_hflm_class, score_choices
    from tflt.loopscope.phase8_runtime import (AnswerResidualCollector, Phase8Runtime,
        fit_basis, save_basis, load_basis, tensor_smoke_checks)
    if importlib.metadata.version('lm_eval') != '0.4.11':
        raise RuntimeError('frozen scorer requires lm_eval 0.4.11')
    torch.set_grad_enabled(False)
    torch.manual_seed(20260905)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    revision = model_spec['observed_revision']
    tokenizer = AutoTokenizer.from_pretrained(model_spec['model'], revision=revision,
                                               local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_spec['model'], revision=revision,
        local_files_only=True, trust_remote_code=True,
        torch_dtype=getattr(torch, model_spec['historical_runtime_dtype'])).to('cuda').eval()
    if model.config._commit_hash != revision:
        raise RuntimeError('model revision differs from frozen source')
    vanilla = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='1', trust_remote_code=True)
    adapter = build_hflm_class(HFLM)(pretrained=model, tokenizer=tokenizer,
                                    batch_size='1', trust_remote_code=True)
    versions = {name: importlib.metadata.version(name) for name in ('torch','transformers','lm_eval','datasets')}
    write_json(args.run_root / 'env.json', dict(versions=versions, python=sys.executable,
        platform=platform.platform(), model_revision=model.config._commit_hash,
        tokenizer_revision=revision, model_dtype=str(model.dtype), max_length=adapter.max_length,
        job_id=os.environ.get('SLURM_JOB_ID'), partition=os.environ.get('SLURM_JOB_PARTITION'),
        gpu=torch.cuda.get_device_name(), total_memory=torch.cuda.get_device_properties(0).total_memory,
        source_commit=args.commit))

    def original(row):
        requests = [Instance(request_type='loglikelihood', doc={}, arguments=(row['prompt'], ' '+letter),
                             idx=i, metadata=('phase8', row['identity'], 1))
                    for i, letter in enumerate('ABCD')]
        return finite_scores([float(x[0]) for x in vanilla.loglikelihood(requests, disable_tqdm=True)])

    def adapted(row, runtime=None):
        previous = adapter.phase8_runtime
        adapter.phase8_runtime = runtime
        try:
            return finite_scores([float(x[0]) for x in score_choices(adapter, row['prompt'], row['identity'])])
        finally:
            adapter.phase8_runtime = previous

    def require_same(left, right, label):
        delta = max(abs(a-b) for a,b in zip(left,right))
        if delta != 0:
            raise RuntimeError('%s score mismatch max_abs=%r; measure native repeat before assigning tolerance' % (label,delta))
        return delta

    native = []
    for row in verify_rows:
        a, b = original(row), original(row)
        require_same(a,b,'native repeat')
        c = adapted(row)
        require_same(a,c,'native adapter')
        native.append(dict(identity=row['identity'], scores=a, adapter_scores=c))
    write_json(args.run_root / 'native.json', native)
    semantic = tensor_smoke_checks(torch)
    reports = []
    for window, k in cells:
        name = 'w%d-%d-k%d' % (*window,k)
        cell_dir = args.run_root / name
        cell_dir.mkdir()
        cfg = dict(model_alias=model_spec['model'], window=tuple(window), k=k,
                   alpha=1.0, beta=0.0, strategy='damped_euler', iteration_mode='block',
                   cache_strategy=model_spec['cache_strategy'], decode_mode='bypass')
        collector = AnswerResidualCollector([r['identity'] for r in fit_rows], k)
        collecting = Phase8Runtime(collector=collector)
        with looped_model(model, LoopConfig(**cfg, residual_transform=collecting)):
            for row in fit_rows:
                adapted(row, collecting)
        if collector.summary()['rows_by_t'] != {str(t):4 for t in range(k)}:
            raise ValueError('calibration residual timestep/identity membership differs')
        metadata = dict(scope='SMOKE_ONLY', model=model_spec['model'], revision=revision,
                        window=window, k=k, dtype=model_spec['historical_runtime_dtype'],
                        cache_strategy=model_spec['cache_strategy'], alpha=1.0,
                        dataset=pool['dataset'])
        basis = fit_basis(collector, metadata)
        save_basis(cell_dir / 'basis.json', basis)
        loaded = load_basis(cell_dir / 'basis.json', expected_metadata=metadata)
        if basis != loaded:
            raise ValueError('basis serialization did not preserve values')
        # Only bounded answer-position rows are persisted, never full hidden states.
        torch.save(dict(identities=collector.identities, t1=collector.matrix(), metadata=metadata), cell_dir / 'residual_t1.pt')
        results = []
        for row in verify_rows:
            with looped_model(model, LoopConfig(**cfg)):
                reference = original(row)
                none = adapted(row)
            require_same(reference, none, 'None')
            zero = Phase8Runtime(loaded['direction'], strength=0)
            with looped_model(model, LoopConfig(**cfg, residual_transform=zero)):
                zero_scores = adapted(row, zero)
            require_same(reference,zero_scores,'zero strength')
            spectral = Phase8Runtime(loaded['direction'])
            with looped_model(model, LoopConfig(**cfg, residual_transform=spectral)):
                scores = adapted(row,spectral)
            if not spectral.calls or any(x['applied'] != (x['t'] >= 1) for x in spectral.calls):
                raise ValueError('intervention step coverage differs')
            results.append(dict(identity=row['identity'], original=reference, none=none,
                zero=zero_scores, spectral=scores, calls=spectral.calls,
                positions=adapter.phase8_positions))
        report = dict(name=name, metadata=metadata, collector=collector.summary(),
                      fit_calls=collecting.calls, results=results, basis_roundtrip=True)
        write_json(cell_dir / 'debug.json',report)
        reports.append(report)
    elapsed = time.monotonic()-started
    summary = dict(status='DEBUG_COMPLETE', scope='SMOKE_ONLY', recipe=recipe,
        semantic_checks=semantic, cell_names=[r['name'] for r in reports],
        elapsed_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), oom_count=0,
        verified_identity_count=len(verify_rows), fit_identity_count=len(fit_rows))
    write_json(args.run_root / 'summary.json',summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
