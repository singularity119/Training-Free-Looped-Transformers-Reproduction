import os,json,sys,datetime
from pathlib import Path
b={'gate': 'D-7', 'executor': '01a142a7-562a-7e83-b085-70999fbbc780', 'planning': '01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f', 'source_commit': '3866f6aa0d775f0b6e721f90a88fff0af5455abc', 'stage': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z', 'source': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z/source', 'run_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-d7-20261016T031135Z', 'input_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-d7-20261016T031135Z', 'artifact_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/artifacts/phase11-gate-d7-20261016T031135Z', 'python': '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python', 'original_pool': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z/mmlu_pro-prepared/mmlu_pro-pool.json', 'runtime_seed': 20261016, 'target_gold_loaded': False, 'git_archive_exit': 0, 'deployment_exit': 0}
sys.path[:0]=[b['source']+'/src',b['source']+'/scripts/loopscope']
from phase11_d7 import load_job,binding,validate_bundle,TAG_FIELDS
from run_phase11_accuracy import source_provenance
source=source_provenance(b['source_commit'])
plan=json.loads(Path(b['stage']+'/plans/debug/debug-plan.json').read_text());task=plan['tasks'][0]
batch=json.loads(Path(task['batch_root']+'/batch_summary.json').read_text());assert batch['status']=='BATCH_COMPLETE' and batch['target_count']==5
scope=json.loads(Path(task['batch_root']+'/scope_check.json').read_text());assert scope['status']=='D7_FULL_DAMPING_CUDA_VERIFIED' and scope['target_gold_loaded'] is False and len(scope['evidence'])==2
facts=[]
for job in task['jobs']:
 scope_name,cfg,cell,pool,ids=load_job(job);root=Path(job['run_root']);meta=binding(job,scope_name,cfg,cell,ids)
 summary=json.loads((root/'summary.json').read_text());assert summary['status']=='RAW_COMPLETE' and summary['metadata']==meta and summary['count']==1
 for i in ids:
  record=json.loads((root/'complete'/f'{i:06d}.json').read_text());r,t=record['record'],record['telemetry'];validate_bundle(i,r,t,meta,pool)
  assert t['last_cache_length']==t['prefill_input_tokens']+r['generated_token_count']-1
 env=json.loads((Path(summary['attempt_root'])/'env.json').read_text());assert env['effective_rng_seeds']=={k:20261016 for k in ('python','numpy','torch','cuda_all_devices')}
 facts.append(dict(cell_key=job['cell_key'],source_commit=job['source_commit'],count=summary['count'],effective_rng_seeds=env['effective_rng_seeds'],prefill_input_tokens=t['prefill_input_tokens'],generated_tokens=r['generated_token_count'],prefill_count=t['prefill_count'],decode_count=t['decode_count'],last_cache_length=t['last_cache_length'],online=t.get('online'),summary_path=str(root/'summary.json')))
result=dict(status='D7_DEBUG_INDEPENDENTLY_VERIFIED',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=source,synthetic_count=5,scope_status=scope['status'],scope_evidence=scope['evidence'],facts=facts,functional_path_valid=True,formal_resource_sizing=False,debug_job='13114497_0',target_gold_loaded=False)
p=Path(b['stage']+'/verification/debug_verified.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
