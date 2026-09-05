#!/usr/bin/env python3
"""Gate C unmodified Euler answer-residual acquisition; no target outcomes."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def select_rows(pool, model, scope):
    rows = pool['rows']
    if len(rows) != 512 or [r['identity'] for r in rows] != pool['calibration_identities']:
        raise ValueError('formal pool must preserve all 512 identities in order')
    if scope == 'FORMAL_CALIBRATION':
        return rows
    chosen = {r['identity'] for r in rows[:2]}
    chosen.update(r['identity'] for r in sorted(rows, key=lambda r: (-r['prompt_token_lengths'][model], r['identity']))[:2])
    return [r for r in rows if r['identity'] in chosen]


def diagnostics(matrices, basis, torch, k):
    v = torch.tensor(basis['direction'], dtype=torch.float64)
    projections, norms, energy = {}, {}, {}
    for t, matrix in matrices.items():
        d = matrix.double()
        a = d @ v
        n = torch.linalg.vector_norm(d, dim=1)
        projections[t], norms[t] = a.tolist(), n.tolist()
        den = float(n.square().sum())
        energy[t] = float(a.square().sum()) / den if den else None
    repeated = torch.stack([torch.tensor(projections[str(t)], dtype=torch.float64) for t in range(1,k)], dim=1)
    den = repeated.abs().sum(dim=1)
    coherence = [float(abs(s)/d) if float(d) else None for s,d in zip(repeated.sum(dim=1),den)]
    pairs = repeated[:,:-1] * repeated[:,1:]
    nonzero = pairs != 0
    return dict(projections_by_t=projections, residual_norms_by_t=norms,
        projection_energy_fraction_by_t=energy, cumulative_repeated_projection=(repeated.sum(dim=1)/k).tolist(),
        coherence=coherence, coherence_undefined_count=coherence.count(None),
        adjacent_same_sign_fraction=float((pairs[nonzero]>0).double().mean()) if bool(nonzero.any()) else None,
        adjacent_zero_pair_count=int((~nonzero).sum()),
        interpretation='in-sample descriptive; K2 has one repeated step, not accumulation evidence')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-index',type=int,choices=(0,1),required=True)
    p.add_argument('--cell-index',type=int,choices=range(4),required=True)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--scope',choices=('PREFLIGHT_ONLY','FORMAL_CALIBRATION'),required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    repo=Path(__file__).resolve().parents[2]
    config=json.loads((repo/'configs/loopscope/phase8_model_windows.json').read_text())
    spec=config['models'][args.model_index]
    window,k=[(w['layers'],k) for w in spec['windows'] for k in config['K']][args.cell_index]
    if args.dry_run:
        print(json.dumps(dict(model=spec,window=window,k=k,scope=args.scope,source_commit=args.commit)))
        return 0
    if not os.environ.get('SLURM_JOB_ID') or (args.scope=='PREFLIGHT_ONLY' and os.environ.get('SLURM_JOB_PARTITION')!='debug'):
        raise RuntimeError('calibration requires Slurm; preflight requires debug')
    pool=json.loads(args.pool.read_text())
    rows=select_rows(pool,spec['model'],args.scope)
    args.run_root.mkdir(parents=True,exist_ok=False)
    metadata=dict(scope=args.scope,model=spec['model'],revision=spec['observed_revision'],window=window,k=k,
        dtype=spec['historical_runtime_dtype'],cache_strategy=spec['cache_strategy'],alpha=1.0,
        dataset=pool['dataset'],source_commit=args.commit,run_root=str(args.run_root),pool=str(args.pool))
    write_json(args.run_root/'command_args.json',dict(metadata=metadata,argv=sys.argv))
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tflt.config import LoopConfig
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase8_adapter import build_hflm_class,score_choices
    from tflt.loopscope.phase8_runtime import AnswerResidualCollector,Phase8Runtime,fit_basis,save_basis
    if importlib.metadata.version('lm_eval')!='0.4.11':
        raise RuntimeError('lm_eval version mismatch')
    torch.set_grad_enabled(False)
    torch.manual_seed(20260905)
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(spec['model'],revision=spec['observed_revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(spec['model'],revision=spec['observed_revision'],local_files_only=True,
        torch_dtype=getattr(torch,spec['historical_runtime_dtype'])).to('cuda').eval()
    if model.config._commit_hash!=spec['observed_revision']:
        raise RuntimeError('model revision mismatch')
    adapter=build_hflm_class(HFLM)(pretrained=model,tokenizer=tokenizer,batch_size='1',trust_remote_code=True)
    write_json(args.run_root/'env.json',dict(source_commit=args.commit,model_revision=model.config._commit_hash,
        model_dtype=str(model.dtype),python=sys.executable,versions={n:importlib.metadata.version(n) for n in ('torch','transformers','lm_eval','datasets')},
        job_id=os.environ['SLURM_JOB_ID'],partition=os.environ.get('SLURM_JOB_PARTITION'),gpu=torch.cuda.get_device_name(),
        total_memory=torch.cuda.get_device_properties(0).total_memory))
    collector=AnswerResidualCollector([r['identity'] for r in rows],k)
    runtime=Phase8Runtime(collector=collector)
    adapter.phase8_runtime=runtime
    cfg=LoopConfig(model_alias=spec['model'],window=tuple(window),k=k,alpha=1.0,beta=0.,strategy='damped_euler',
        iteration_mode='block',cache_strategy=spec['cache_strategy'],decode_mode='bypass',residual_transform=runtime)
    positions=[]
    acquisition=time.monotonic()
    with looped_model(model,cfg):
        for i,row in enumerate(rows):
            score_choices(adapter,row['prompt'],row['identity']) # unchanged scorer; discard all scores
            positions.append(dict(identity=row['identity'],candidates=adapter.phase8_positions))
            if (i+1)%32==0:
                print(json.dumps(dict(completed=i+1,total=len(rows),elapsed=time.monotonic()-acquisition)),flush=True)
    torch.cuda.synchronize()
    acquisition_elapsed=time.monotonic()-acquisition
    matrices={str(t):collector.matrix(t) for t in range(k)}
    basis=fit_basis(collector,metadata)
    save_basis(args.run_root/'basis.json',basis)
    torch.save(dict(identities=collector.identities,matrices=matrices,positions={str(t):[collector.positions[(key,t)] for key in collector.keys] for t in range(k)},metadata=metadata),args.run_root/'residuals.pt')
    write_json(args.run_root/'positions.json',positions)
    write_json(args.run_root/'diagnostics.json',diagnostics(matrices,basis,torch,k))
    summary=dict(status='CALIBRATION_COMPLETE',metadata=metadata,identities=collector.identities,collector=collector.summary(),
        elapsed_seconds=time.monotonic()-started,acquisition_seconds=acquisition_elapsed,trajectories_per_second=len(rows)/acquisition_elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),oom_count=0,
        max_prompt_tokens=max(r['prompt_token_lengths'][spec['model']] for r in rows),target_gold_loaded=False,choice_scores_saved=False)
    write_json(args.run_root/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
