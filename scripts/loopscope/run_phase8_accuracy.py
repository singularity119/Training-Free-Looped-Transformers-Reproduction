#!/usr/bin/env python3
"""Save untouched HFLM raw scores for one frozen Gate D cell/shard, without gold."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from tflt.loopscope.phase8_accuracy import panel,select_rows,checked_basis,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--cell-id',required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--scope',choices=('PREFLIGHT_ONLY','FORMAL_TEST'),required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--end',type=int,default=14042)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    repo=Path(__file__).resolve().parents[2]
    config=json.loads((repo/'configs/loopscope/phase8_model_windows.json').read_text())
    if manifest['cells']!=panel(config): raise ValueError('manifest differs from frozen panel')
    cell=next(c for c in manifest['cells'] if c['cell_id']==a.cell_id)
    pool=json.loads(a.pool.read_text())
    rows=select_rows(pool,cell,a.scope,a.start,a.end)
    if not os.environ.get('SLURM_JOB_ID') or (a.scope=='PREFLIGHT_ONLY' and os.environ.get('SLURM_JOB_PARTITION')!='debug'):
        raise RuntimeError('GPU scoring requires Slurm, preflight requires debug')
    basis=checked_basis(cell) if cell['arm']=='Spectral' else None
    a.run_root.mkdir(parents=True,exist_ok=False)
    metadata=dict(cell=cell,scope=a.scope,start=a.start,end=a.end,source_commit=a.commit,pool=str(a.pool),manifest=str(a.manifest))
    write_json(a.run_root/'command_args.json',metadata)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tflt.config import LoopConfig
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase8_adapter import build_hflm_class,score_choices
    from tflt.loopscope.phase8_runtime import Phase8Runtime
    if importlib.metadata.version('lm_eval')!='0.4.11': raise RuntimeError('lm_eval version mismatch')
    torch.set_grad_enabled(False)
    torch.manual_seed(20260905)
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(cell['model'],revision=cell['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cell['model'],revision=cell['revision'],local_files_only=True,torch_dtype=getattr(torch,cell['dtype'])).to('cuda').eval()
    if model.config._commit_hash!=cell['revision']: raise RuntimeError('model revision mismatch')
    runtime=Phase8Runtime(basis['direction']) if basis else None
    adapter=build_hflm_class(HFLM)(pretrained=model,tokenizer=tokenizer,batch_size='1',trust_remote_code=True,phase8_runtime=runtime)
    manager=nullcontext()
    if cell['arm']!='Native':
        cfg=LoopConfig(model_alias=cell['model'],window=tuple(cell['window']),k=cell['k'],alpha=1.,beta=0.,strategy='damped_euler',iteration_mode='block',cache_strategy=cell['cache_strategy'],decode_mode='bypass',residual_transform=runtime)
        manager=looped_model(model,cfg)
    write_json(a.run_root/'env.json',dict(source_commit=a.commit,model_revision=model.config._commit_hash,model_dtype=str(model.dtype),python=sys.executable,versions={n:importlib.metadata.version(n) for n in ('torch','transformers','lm_eval','datasets')},job_id=os.environ['SLURM_JOB_ID'],partition=os.environ.get('SLURM_JOB_PARTITION'),gpu=torch.cuda.get_device_name(),total_memory=torch.cuda.get_device_properties(0).total_memory))
    positions=dict(max_input_length=0,left_truncated_requests=0,multitoken_requests=0,callback_calls=0,applied_calls=0)
    acquisition=time.monotonic()
    with manager,(a.run_root/'scores.jsonl').open('x') as out:
        for i,row in enumerate(rows):
            scores=[float(pair[0]) for pair in score_choices(adapter,row['prompt'],row['identity'])]
            if len(scores)!=4 or not all(math.isfinite(x) for x in scores): raise ValueError('four finite scores required')
            out.write(json.dumps(dict(identity=row['identity'],subject=row['subject'],doc_index=row['doc_index'],scores=scores),allow_nan=False)+'\n')
            for pos in adapter.phase8_positions:
                positions['max_input_length']=max(positions['max_input_length'],pos['input_length'])
                positions['left_truncated_requests']+=int(pos['left_truncated_tokens']>0)
                positions['multitoken_requests']+=int(pos['continuation_length']>1)
            if runtime:
                if not runtime.calls or not any(c['applied'] for c in runtime.calls): raise ValueError('spectral path did not apply')
                positions['callback_calls']+=len(runtime.calls)
                positions['applied_calls']+=sum(c['applied'] for c in runtime.calls)
                runtime.calls.clear()
            if (i+1)%32==0:
                out.flush()
                print(json.dumps(dict(completed=i+1,total=len(rows),elapsed=time.monotonic()-acquisition)),flush=True)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-acquisition
    write_json(a.run_root/'summary.json',dict(status='SCORES_COMPLETE',metadata=metadata,count=len(rows),target_gold_loaded=False,elapsed_seconds=time.monotonic()-started,acquisition_seconds=elapsed,samples_per_second=len(rows)/elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),oom_count=0,positions=positions))

if __name__=='__main__': main()
