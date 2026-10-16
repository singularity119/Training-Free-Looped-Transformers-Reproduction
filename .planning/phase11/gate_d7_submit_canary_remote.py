import json,os,sys,subprocess,datetime
from pathlib import Path
b={'gate': 'D-7', 'executor': '01a142a7-562a-7e83-b085-70999fbbc780', 'planning': '01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f', 'source_commit': '3866f6aa0d775f0b6e721f90a88fff0af5455abc', 'stage': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z', 'source': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z/source', 'run_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-d7-20261016T031135Z', 'input_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-d7-20261016T031135Z', 'artifact_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/artifacts/phase11-gate-d7-20261016T031135Z', 'python': '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python', 'original_pool': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z/mmlu_pro-prepared/mmlu_pro-pool.json', 'runtime_seed': 20261016, 'target_gold_loaded': False, 'git_archive_exit': 0, 'deployment_exit': 0}
sys.path[:0]=[b['source']+'/src',b['source']+'/scripts/loopscope']
from build_phase11_d7_plan import build
from phase11_d7_batch import save
verified=json.loads(Path(b['stage']+'/verification/debug_verified.json').read_text()); assert verified['status']=='D7_DEBUG_INDEPENDENTLY_VERIFIED' and verified['source']['source_head']==b['source_commit']
root=Path(b['stage']+'/plans/canary');assert not root.exists()
recipe,plan=build('canary',b['source'],b['run_root'],b['source_commit'],b['input_root']+'/formal/cot-pool.json',packing=1,tasks=3)
assert plan['scheduled_record_count']==12 and len(plan['cells'])==3
save(root/'canary-plan.json',plan,once=True);save(root/'recipe.json',recipe,once=True)
pool=json.loads(Path(b['input_root']+'/formal/cot-pool.json').read_text())
membership=dict(canonical_indices=plan['canary_indices'],selection_rule='sort by prompt token length then canonical index; floor((12032-1)*q), q=.25/.5/.75/1; run selected identities in canonical order',members=[dict(canonical_index=i,identity=pool['rows'][i]['identity'],prompt_token_length=len(pool['rows'][i]['input_ids'])) for i in plan['canary_indices']],target_gold_loaded=False)
save(root/'membership.json',membership,once=True)
commands=[['/opt/slurm/bin/squeue','-h','-r','-u','xhuang225','-o','%i|%P|%T|%b'],['/opt/slurm/bin/scontrol','show','partition','emergency_gpu','-o'],['/opt/slurm/bin/sacctmgr','-nP','show','qos','emergency_gpu','format=Name,Priority,MaxTRESPU,MaxSubmitPU,MaxWall']]
qualification=[]
for cmd in commands:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=25);assert r.returncode==0,(cmd,r.returncode,r.stdout,r.stderr);qualification.append(dict(command=cmd,stdout=r.stdout))
assert 'PriorityTier=300' in qualification[1]['stdout'] and 'MaxTime=14-00:00:00' in qualification[1]['stdout'] and 'gres/gpu:a800=8' in qualification[2]['stdout']
save(b['stage']+'/verification/canary_qualification.json',qualification,once=True)
cmd=['/opt/slurm/bin/sbatch','--parsable','--account=root','--partition=emergency_gpu','--qos=emergency_gpu','--gres=gpu:a800:1','--cpus-per-task=8','--mem=64G','--time=14-00:00:00','--array=0-2','--job-name=p11-d7-canary','--output='+b['run_root']+'/logs/canary-%A_%a.out','--error='+b['run_root']+'/logs/canary-%A_%a.err',b['source']+'/scripts/loopscope/phase11_d7.sbatch',b['source'],b['python'],str(root/'canary-plan.json')]
r=subprocess.run(cmd,capture_output=True,text=True)
receipt=dict(command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,source_commit=b['source_commit'],submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),record_count=12,gpus=3,packing=1,membership=membership,target_gold_loaded=False)
save(b['stage']+'/verification/canary_submission.json',receipt,once=True)
if r.returncode:raise RuntimeError(receipt)
base=r.stdout.strip().split(';')[0];assert base.isdecimal();ids=[base+'_'+str(i) for i in range(3)]
jobset=dict(gate='D-7',executor=b['executor'],planning=b['planning'],source_commit=b['source_commit'],stage='canary',jobs=[dict(job_id=jid,plan=str(root/'canary-plan.json'),task_index=i) for i,jid in enumerate(ids)],target_gold_loaded=False)
save(b['stage']+'/plans/canary-jobset.json',jobset,once=True)
v=subprocess.run(['/opt/slurm/bin/squeue','-h','-r','-j',base,'-o','%i|%T|%R'],capture_output=True,text=True,timeout=25)
assert v.returncode==0 and set(ids).issubset({x.split('|')[0] for x in v.stdout.splitlines()}),(v.returncode,v.stdout,v.stderr)
receipt.update(job_ids=ids,jobset=b['stage']+'/plans/canary-jobset.json',initial_queue=v.stdout.strip());save(b['stage']+'/verification/canary_submission_verified.json',receipt,once=True);print(json.dumps(receipt))
