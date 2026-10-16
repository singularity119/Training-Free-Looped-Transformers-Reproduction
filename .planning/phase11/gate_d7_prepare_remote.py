import json,os,subprocess
from pathlib import Path
b={'gate': 'D-7', 'executor': '01a142a7-562a-7e83-b085-70999fbbc780', 'planning': '01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f', 'source_commit': '3866f6aa0d775f0b6e721f90a88fff0af5455abc', 'stage': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z', 'source': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d7-20261016T031135Z/source', 'run_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-d7-20261016T031135Z', 'input_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-d7-20261016T031135Z', 'artifact_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/artifacts/phase11-gate-d7-20261016T031135Z', 'python': '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python', 'original_pool': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z/mmlu_pro-prepared/mmlu_pro-pool.json', 'runtime_seed': 20261016, 'target_gold_loaded': False, 'git_archive_exit': 0, 'deployment_exit': 0}
os.chdir(b['source'])
env=dict(os.environ,PYTHONPATH=b['source']+'/src',PYTHONDONTWRITEBYTECODE='1',HF_HOME='/hpc2hdd/home/xhuang225/shared/hf_home',HF_DATASETS_CACHE='/hpc2hdd/home/xhuang225/shared/datasets',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
checks=[]
for pattern in ('test_phase11_d7_producer.py','test_phase11_d7_plan.py','test_phase11_mmlupro_s2.py','test_phase11_d7_analysis.py'):
 cmd=[b['python'],'-m','unittest','discover','-s','tests','-p',pattern]
 r=subprocess.run(cmd,env=env,capture_output=True,text=True)
 checks.append(dict(command=cmd,exit_code=r.returncode,output=r.stdout+r.stderr))
 if r.returncode: raise RuntimeError(checks[-1])
Path(b['stage']+'/verification/cpu_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
for synthetic in (False,True):
 cmd=[b['python'],'scripts/loopscope/phase11_d7.py','--prepare-inputs','--original-pool',b['original_pool'],'--output-dir',b['input_root']+('/synthetic' if synthetic else '/formal'),'--commit',b['source_commit']]+(['--synthetic'] if synthetic else [])
 r=subprocess.run(cmd,env=env,capture_output=True,text=True)
 Path(b['stage']+'/verification/'+('synthetic' if synthetic else 'formal')+'_prepare.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode: raise RuntimeError((r.returncode,r.stdout,r.stderr))
 print(r.stdout.strip(),flush=True)
cmd=[b['python'],'scripts/loopscope/build_phase11_d7_plan.py','--stage','debug','--source',b['source'],'--run-prefix',b['run_root'],'--commit',b['source_commit'],'--pool',b['input_root']+'/formal/cot-pool.json','--synthetic-pool',b['input_root']+'/synthetic/cot-pool.json','--output-dir',b['stage']+'/plans/debug']
r=subprocess.run(cmd,env=env,capture_output=True,text=True)
if r.returncode: raise RuntimeError((r.returncode,r.stdout,r.stderr))
print(r.stdout.strip(),flush=True)
print(json.dumps(dict(status='D7_CPU_INPUTS_AND_DEBUG_PLAN_READY',source_commit=b['source_commit'],tests=21,target_gold_loaded=False)))
