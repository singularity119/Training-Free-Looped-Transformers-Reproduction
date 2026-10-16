import json,sys,subprocess,datetime
from pathlib import Path
b={'gate': 'D-7', 'executor': '01a142a7-562a-7e83-b085-70999fbbc780', 'planning': '01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f', 'source_commit': '3866f6aa0d775f0b6e721f90a88fff0af5455abc', 'stage': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z', 'source': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z/source', 'run_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-d7-20261016T031135Z', 'input_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-d7-20261016T031135Z', 'artifact_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/artifacts/phase11-gate-d7-20261016T031135Z', 'python': '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python', 'original_pool': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z/mmlu_pro-prepared/mmlu_pro-pool.json', 'runtime_seed': 20261016, 'target_gold_loaded': False, 'git_archive_exit': 0, 'deployment_exit': 0}
sys.path[:0]=[b['source']+'/src',b['source']+'/scripts/loopscope']
from build_phase11_d7_plan import build
from phase11_d7_batch import save
from run_phase11_accuracy import source_provenance
source_provenance(b['source_commit'])
v=json.loads(Path(b['stage']+'/verification/canary_verified.json').read_text());r=json.loads(Path(b['stage']+'/verification/canary_resources.json').read_text())
assert v['status']=='D7_RETAINABLE_CANARY_VERIFIED' and v['record_count']==12 and r['packing']['candidate_from_memory_bound']==4
assert 4*r['packing']['largest_padded_process_bytes']<=.8*r['packing']['device_capacity_bytes']
root=Path(b['stage']+'/plans/first');assert not root.exists()
recipe,plan=build('first',b['source'],b['run_root'],b['source_commit'],b['input_root']+'/formal/cot-pool.json',packing=4,tasks=8,shard_size=256,canary_plan=b['stage']+'/plans/canary/canary-plan.json')
assert plan['record_count']==36096 and plan['scheduled_record_count']==36084 and len(plan['tasks'])==8 and {c['cell_key'] for c in plan['cells']}=={'native','current_t_k2_l05','fixed_t0_k3_l09'}
plan['resource_admission']=dict(packing=4,source_evidence=b['stage']+'/verification/canary_resources.json',memory_bound_bytes_per_process=r['packing']['largest_padded_process_bytes'],planned_memory_bytes=4*r['packing']['largest_padded_process_bytes'],headroom_fraction=1-4*r['packing']['largest_padded_process_bytes']/r['packing']['device_capacity_bytes'],measured_packing=[1],concurrency_status='conservative additive memory+KV admission; preserve first formal packed actual memory/throughput evidence',batch_size=1,gpu_cap=8)
save(root/'first-plan.json',plan,once=True);save(root/'recipe.json',recipe,once=True)
qualification=[]
for cmd in [['/opt/slurm/bin/squeue','-h','-r','-u','xhuang225','-o','%i|%P|%T|%b'],['/opt/slurm/bin/scontrol','show','partition','emergency_gpu','-o'],['/opt/slurm/bin/sacctmgr','-nP','show','qos','emergency_gpu','format=Name,Priority,MaxTRESPU,MaxSubmitPU,MaxWall']]:
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=25);assert q.returncode==0,(q.returncode,q.stdout,q.stderr);qualification.append(dict(command=cmd,stdout=q.stdout))
assert not qualification[0]['stdout'].strip(),'Existing user inflight must be reconciled before adding all8'
assert 'PriorityTier=300' in qualification[1]['stdout'] and 'MaxTime=14-00:00:00' in qualification[1]['stdout'] and 'gres/gpu:a800=8' in qualification[2]['stdout']
save(b['stage']+'/verification/first_qualification.json',qualification,once=True)
cmd=['/opt/slurm/bin/sbatch','--parsable','--account=root','--partition=emergency_gpu','--qos=emergency_gpu','--gres=gpu:a800:1','--cpus-per-task=8','--mem=64G','--time=14-00:00:00','--array=0-7%8','--job-name=p11-d7-first','--output='+b['run_root']+'/logs/first-%A_%a.out','--error='+b['run_root']+'/logs/first-%A_%a.err',b['source']+'/scripts/loopscope/phase11_d7.sbatch',b['source'],b['python'],str(root/'first-plan.json')]
s=subprocess.run(cmd,capture_output=True,text=True);receipt=dict(command=cmd,exit_code=s.returncode,stdout=s.stdout,stderr=s.stderr,source_commit=b['source_commit'],submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),first_batch_total=36096,scheduled_records=36084,reused_retained_canary=12,gpus=8,packing=4,child_processes=sum(len(t['jobs']) for t in plan['tasks']),target_gold_loaded=False)
save(b['stage']+'/verification/first_submission.json',receipt,once=True)
if s.returncode: raise RuntimeError(receipt)
base=s.stdout.strip().split(';')[0];assert base.isdecimal();ids=[base+'_'+str(i) for i in range(8)]
jobset=dict(gate='D-7',executor=b['executor'],planning=b['planning'],source_commit=b['source_commit'],stage='first',jobs=[dict(job_id=j,plan=str(root/'first-plan.json'),task_index=i) for i,j in enumerate(ids)],target_gold_loaded=False)
save(b['stage']+'/plans/first-jobset.json',jobset,once=True)
q=subprocess.run(['/opt/slurm/bin/squeue','-h','-r','-j',base,'-o','%i|%T|%R'],capture_output=True,text=True,timeout=25);assert q.returncode==0 and set(ids).issubset({x.split('|')[0] for x in q.stdout.splitlines()}),(q.returncode,q.stdout,q.stderr)
receipt.update(job_ids=ids,jobset=b['stage']+'/plans/first-jobset.json',initial_queue=q.stdout.strip());save(b['stage']+'/verification/first_submission_verified.json',receipt,once=True);print(json.dumps(receipt))
