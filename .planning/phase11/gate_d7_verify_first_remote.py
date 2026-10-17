"""Scalar-only independent full-first verification, then raw closure; no gold read."""
import json, sys
from pathlib import Path
from datetime import datetime, timezone

x = json.loads(sys.argv[1])
b = x['execution']
sys.path[:0] = [b['source']+'/src', b['source']+'/scripts/loopscope']
from phase11_d7 import load_job, binding, TAG_FIELDS
from summarize_phase11_d7_resources import summarize
from analyze_phase11_d7 import close, write, read

now = lambda: datetime.now(timezone.utc).isoformat()
first_path = b['stage']+'/plans/first/first-plan.json'
canary_path = b['stage']+'/plans/canary/canary-plan.json'
first, canary = read(first_path), read(canary_path)
assert x['scheduler']['all_success'] and x['scheduler']['done_record_count'] == 36084
assert all(v['state']=='COMPLETED' and v['exit_code']=='0:0' for v in x['scheduler']['scheduler'].values())
facts, seen, count = [], set(), 0
for plan in (canary, first):
 for task in plan['tasks']:
  batch = read(task['batch_root']+'/batch_summary.json')
  assert batch['status']=='BATCH_COMPLETE' and batch['target_count']==sum(len(j['indices']) for j in task['jobs'])
  for job in task['jobs']:
   scope,cfg,cell,pool,indices = load_job(job)
   root=Path(job['run_root']);meta=binding(job,scope,cfg,cell,indices);summary=read(root/'summary.json')
   assert summary['status']=='RAW_COMPLETE' and summary['metadata']==meta and summary['count']==len(indices)
   env=read(Path(summary['attempt_root'])/'env.json')
   assert env['effective_rng_seeds']=={k:20261016 for k in ('python','numpy','torch','cuda_all_devices')}
   assert env['partition']=='emergency_gpu' and env['qos']=='emergency_gpu' and 'A800' in env['gpu']
   assert sorted(int(p.stem) for p in (root/'complete').glob('*.json'))==indices
   for index in indices:
    assert (job['cell_key'],index) not in seen
    seen.add((job['cell_key'],index))
    bundle=read(root/'complete'/f'{index:06d}.json');r,t=bundle['record'],bundle['telemetry']
    row=pool['rows'][index];tag={k:meta[k] for k in TAG_FIELDS}
    assert r['status']=='OK' and r['identity']==row['identity'] and r['canonical_index']==index and r['category']==row['subject'] and r['candidate_labels']==row['choice_labels'] and r['d7_binding']==tag and t['d7_binding']==tag
    n=r['generated_token_count'];assert 0<n<=8192 and len(r['generated_token_ids'])==n and r['max_new_tokens']==8192
    eos=r['generated_token_ids'][-1] in r['eos_token_ids']
    assert r['end_reason']==('eos' if eos else 'length_limit') and (eos or n==8192)
    assert t['identity']==row['identity'] and t['canonical_index']==index and t['prefill_count']==1 and t['decode_count']==n-1 and t['prefill_input_tokens']==len(row['input_ids']) and t['last_cache_length']==len(row['input_ids'])+n-1 and t['cache_growth_closed'] and t['incremental_generation']
    if cell['arm']=='Online':
     online=t['online'];expected={str(j):int(cell['direction_policy']=='current_t' or j>0)*n for j in range(cell['k'])};fits={str(j):int(cell['direction_policy']=='current_t' or j==0) for j in range(cell['k'])}
     assert online['applied_by_t']==expected and online['direction_fits_by_t']==fits and online['decode_fit_count']==0 and online['directions_released'] and online['forward_count']==n and online['callback_count']==n*cell['k'] and online['damping_scope']=='full'
     assert all(v['algorithm']=='cuda_gesvd_fp32_reduced_svd' for v in online['last_prefill_fit_metadata'].values())
    else: assert 'online' not in t
    count+=1
   facts.append({'cell_key':job['cell_key'],'count':len(indices),'raw_root':str(root),'effective_seed':20261016,'source_commit':job['source_commit']})
assert count==36096 and len(seen)==36096
assert all(sum(k==cell for k,i in seen)==12032 for cell in ('native','current_t_k2_l05','fixed_t0_k3_l09'))
output=Path(x['output_dir']);output.mkdir(parents=True,exist_ok=False)
verification={'status':'D7_FIRST_36096_INDEPENDENTLY_VERIFIED','verified_at_utc':now(),'count':count,'facts':facts,'identity_seed_source_cache_direction_verified':True,'generated_text_or_answers_examined':False,'verifier_gold_read':False,'gate_prior_interim_gold_access':True,'prior_interim':x['prior_interim'],'debug_kbody_stash1_first_evidence':b['stage']+'/verification/debug_verified.json','scheduler':x['scheduler']}
write(output/'verification.json',verification)
config='/hpc2hdd/home/xhuang225/shared/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/config.json'
resources=summarize([canary_path,first_path],config,2860,packing_cap=4)
for value in resources['cells'].values():value.pop('canonical_indices',None)
resources['environment_count']=len(resources.pop('environments'))
resources['scheduler_records']=x['scheduler']
resources['gate_prior_interim_gold_access']=True
resources['selection_bias']={'design':'All first-three cells, each full12032 including retainedcanary4; Loop cells still unmeasured proxies','caveat':'Full first-cell observed means and packed interference now measured; Loop generation lengths may differ, queue/startup/tails remain. Projections are not confidence intervals or upper bounds.'}
resources['packing']['limitation']='Packing4 completed all first36096; retain cap4, batch1 independent processes. Full8192 KV additive bound retained.'
resources['actual_device_min_headroom_fraction']=min(1-d['peak_used_bytes']/d['total_bytes'] for d in resources['devices'])
resources['actual_first_allocation_gpuh']=x['scheduler']['allocation_gpuh']+0.45166666666666666
resources['debug_allocation_gpuh']=69/3600
resources['actual_gate_to_date_gpuh']=resources['actual_first_allocation_gpuh']+resources['debug_allocation_gpuh']
assert resources['packing']['candidate_from_memory_bound']>=4 and resources['actual_device_min_headroom_fraction']>=.2
assert all(v['status']=='BATCH_COMPLETE' for v in resources['batch_attempts'])
write(output/'resources.json',resources)
closure,predictions=close(first,first=True)
assert not predictions
closure.update(source_plan=first_path,gate_target_gold_previously_loaded=True,prior_interim=x['prior_interim'],gold_provenance_note='Legacy GOLD_UNREAD status is required by immutable producer barrier and means this raw-only closure did not read gold. Gate previously read gold under explicit user interim exception; no claim of Gate-level unread gold.',independent_verification=str(output/'verification.json'))
write(output/'first_batch_closure.json',closure)
print(json.dumps({'output_dir':str(output),'verification':{k:v for k,v in verification.items() if k!='facts'},'resources':resources,'closure':{k:v for k,v in closure.items() if k!='cells'}},ensure_ascii=False))
