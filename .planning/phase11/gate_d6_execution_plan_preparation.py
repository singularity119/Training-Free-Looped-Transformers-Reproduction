import copy,json
from pathlib import Path
from phase11_d6_batch import validate_task
b={'gate': 'D-6', 'executor': '01a13d93-a681-7d20-ab05-434dc91e1097', 'planning': '01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f', 'source_commit': 'b1130f8ca66140d4c2d4add5078c6eab64b07535', 'stage': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d6-20261015T033000Z', 'source': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-d6-20261015T033000Z/source', 'run_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-d6-20261015T033000Z', 'input_root': '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-d6-20261015T033000Z', 'python': '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python', 'git_archive_exit': 0, 'deployment_exit': 0, 'target_gold_loaded': False}
root=Path(b['stage'])/'plans'
p=json.loads((root/'formal-plan.json').read_text())
canary=copy.deepcopy(p['tasks'][0]);canary['number']=8
canary['batch_root']=b['run_root']+'/batches/canary-00'
for j in canary['jobs']:
 j['indices']=[0];j['run_root']=b['run_root']+'/canary/'+j['cell_key']
for j in p['tasks'][0]['jobs']:j['indices'].remove(0)
for c in p['cells']:
 c['raw_roots'][0]['indices'].remove(0)
 c['raw_roots'].insert(0,{'run_root':b['run_root']+'/canary/'+c['cell_key'],'records_path':b['run_root']+'/canary/'+c['cell_key']+'/records.jsonl','telemetry_path':b['run_root']+'/canary/'+c['cell_key']+'/telemetry.jsonl','indices':[0]})
p['tasks'].insert(0,canary)
p['admission']={'initial_formal':'12 retainable records, same index0 for all12 frozen cells','packing':5,'remaining_array':'0-7 only after canary functional/resource check','not_a_length_pilot':True}
pairs=[(j['cell_key'],i) for t in p['tasks'] for j in t['jobs'] for i in j['indices']]
assert len(pairs)==len(set(pairs))==5376
assert sum(validate_task(t) for t in p['tasks'])==5376
with (root/'execution-plan.json').open('x') as f:json.dump(p,f)
print(json.dumps({'execution_plan':str(root/'execution-plan.json'),'records':5376,'initial_formal_records':12,'remaining_records':5364,'tasks':9}))
