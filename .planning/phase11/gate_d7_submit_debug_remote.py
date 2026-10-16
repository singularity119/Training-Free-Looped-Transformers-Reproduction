import json,os,subprocess,datetime
from pathlib import Path
b={'gate': 'D-7', 'executor': '01a142a7-562a-7e83-b085-70999fbbc780', 'planning': '01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f', 'source_commit': '3866f6aa0d775f0b6e721f90a88fff0af5455abc', 'stage': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z', 'source': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z/source', 'run_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-d7-20261016T031135Z', 'input_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-d7-20261016T031135Z', 'artifact_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/artifacts/phase11-gate-d7-20261016T031135Z', 'python': '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python', 'original_pool': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z/mmlu_pro-prepared/mmlu_pro-pool.json', 'runtime_seed': 20261016, 'target_gold_loaded': False, 'git_archive_exit': 0, 'deployment_exit': 0}
p=Path(b['stage']+'/plans/debug/debug-plan.json')
plan=json.loads(p.read_text()); assert plan['source_commit']==b['source_commit'] and plan['stage']=='debug'
assert len(plan['tasks'])==1 and all(j['synthetic'] and j['engineering'] for j in plan['tasks'][0]['jobs'])
assert not Path(b['stage']+'/plans/debug-jobset.json').exists()
cmd=['/opt/slurm/bin/sbatch','--parsable','--account=root','--partition=debug','--qos=debug','--gres=gpu:a40:1','--cpus-per-task=8','--mem=64G','--time=00:29:00','--array=0-0','--job-name=p11-d7-debug','--output='+b['run_root']+'/logs/debug-%A_%a.out','--error='+b['run_root']+'/logs/debug-%A_%a.err',b['source']+'/scripts/loopscope/phase11_d7.sbatch',b['source'],b['python'],str(p)]
r=subprocess.run(cmd,capture_output=True,text=True)
receipt=dict(command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=b['source_commit'],target_gold_loaded=False)
Path(b['stage']+'/verification/debug_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
if r.returncode: raise RuntimeError(receipt)
job=r.stdout.strip().split(';')[0]; assert job.isdecimal(),receipt
jobset=dict(gate='D-7',executor=b['executor'],planning=b['planning'],source_commit=b['source_commit'],stage='debug',jobs=[dict(job_id=job+'_0',plan=str(p),task_index=0)],target_gold_loaded=False)
Path(b['stage']+'/plans/debug-jobset.json').write_text(json.dumps(jobset,indent=2)+'\n')
v=subprocess.run(['/opt/slurm/bin/scontrol','show','job',job+'_0','-o'],capture_output=True,text=True)
assert v.returncode==0 and ('JobId='+job+'_0') in v.stdout,(v.returncode,v.stdout,v.stderr)
receipt.update(job_ids=[job+'_0'],jobset=b['stage']+'/plans/debug-jobset.json',initial_scontrol=v.stdout.strip())
Path(b['stage']+'/verification/debug_submission_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
