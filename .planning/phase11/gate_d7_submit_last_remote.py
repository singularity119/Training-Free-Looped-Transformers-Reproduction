"""Submit only the authorized final Loop24064 after the exact first raw receipt."""
import json,sys,subprocess,datetime
from pathlib import Path
x=json.loads(sys.argv[1]);b=x['execution'];closure_dir=Path(x['first_closure_dir'])
sys.path[:0]=[b['source']+'/src',b['source']+'/scripts/loopscope']
from build_phase11_d7_plan import build
from phase11_d7_batch import save,read
from run_phase11_accuracy import source_provenance
source_provenance(b['source_commit'])
v=read(closure_dir/'verification.json');r=read(closure_dir/'resources.json');receipt_path=closure_dir/'first_batch_closure.json'
assert v['status']=='D7_FIRST_36096_INDEPENDENTLY_VERIFIED' and v['count']==36096
assert r['packing']['candidate_from_memory_bound']>=4 and 4 in r['packing']['validated_concurrency'] and r['actual_device_min_headroom_fraction']>=.2
assert 4*r['packing']['largest_padded_process_bytes']<=.8*r['packing']['device_capacity_bytes']
root=Path(b['stage']+'/plans/last');assert not root.exists()
recipe,plan=build('last',b['source'],b['run_root'],b['source_commit'],b['input_root']+'/formal/cot-pool.json',packing=4,tasks=8,shard_size=256,first_plan=b['stage']+'/plans/first/first-plan.json',first_closure=str(receipt_path))
assert plan['record_count']==60160 and plan['scheduled_record_count']==24064 and len(plan['tasks'])==8 and {j['cell_key'] for t in plan['tasks'] for j in t['jobs']}=={'loop_k2','loop_k3'}
provenance={'gate_prior_gold_access':True,'first_gold_access_context':'USER_AUTHORIZED_TWO_CELL_INTERIM','prior_interim':x['prior_interim'],'k3_interim':x['k3_interim'],'producer_gold_access':False,'note':'Immutable producer GOLD_UNREAD fields denote its own gold-free collection/closure only; Gate already accessed gold under explicit user exceptions. No score-driven change.'}
plan['gold_provenance']=provenance;recipe['gold_provenance']=provenance
plan['resource_admission']={'packing':4,'source_evidence':str(closure_dir/'resources.json'),'validated_first_packing':4,'actual_first_headroom_fraction':r['actual_device_min_headroom_fraction'],'padded_process_bytes':r['packing']['largest_padded_process_bytes'],'batch_size':1,'gpu_cap':8,'ordinary_loop_unmeasured':True,'conservative_scope':'Keep first-validated packing4; Loop still unmeasured, preserve failed attempts and only missing if recovery needed'}
save(root/'last-plan.json',plan,once=True);save(root/'recipe.json',recipe,once=True)
qualification=[]
for cmd in [['/opt/slurm/bin/squeue','-h','-r','-u','xhuang225','-o','%i|%P|%T|%b'],['/opt/slurm/bin/scontrol','show','partition','emergency_gpu','-o'],['/opt/slurm/bin/sacctmgr','-nP','show','qos','emergency_gpu','format=Name,Priority,MaxTRESPU,MaxSubmitPU,MaxWall']]:
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=25);assert q.returncode==0,(q.returncode,q.stdout,q.stderr);qualification.append({'command':cmd,'stdout':q.stdout})
assert not qualification[0]['stdout'].strip(),'Existing user inflight must be reconciled before adding8'
assert 'PriorityTier=300' in qualification[1]['stdout'] and 'MaxTime=14-00:00:00' in qualification[1]['stdout'] and 'gres/gpu:a800=8' in qualification[2]['stdout']
save(b['stage']+'/verification/last_qualification.json',qualification,once=True)
cmd=['/opt/slurm/bin/sbatch','--parsable','--account=root','--partition=emergency_gpu','--qos=emergency_gpu','--gres=gpu:a800:1','--cpus-per-task=8','--mem=64G','--time=14-00:00:00','--array=0-7%8','--job-name=p11-d7-last','--output='+b['run_root']+'/logs/last-%A_%a.out','--error='+b['run_root']+'/logs/last-%A_%a.err',b['source']+'/scripts/loopscope/phase11_d7.sbatch',b['source'],b['python'],str(root/'last-plan.json')]
s=subprocess.run(cmd,capture_output=True,text=True)
receipt={'command':cmd,'exit_code':s.returncode,'stdout':s.stdout,'stderr':s.stderr,'source_commit':b['source_commit'],'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_panel_total':60160,'scheduled_records':24064,'first_retained_complete':36096,'gpus':8,'packing':4,'child_processes':sum(len(t['jobs']) for t in plan['tasks']),'first_closure':str(receipt_path),'gold_provenance':provenance}
save(b['stage']+'/verification/last_submission.json',receipt,once=True)
if s.returncode:raise RuntimeError(receipt)
base=s.stdout.strip().split(';')[0];assert base.isdecimal();ids=[base+'_'+str(i) for i in range(8)]
jobset={'gate':'D-7','executor':b['executor'],'planning':b['planning'],'source_commit':b['source_commit'],'stage':'last','jobs':[{'job_id':j,'plan':str(root/'last-plan.json'),'task_index':i} for i,j in enumerate(ids)],'target_gold_loaded':False,'gate_prior_interim_gold_access':True}
save(b['stage']+'/plans/last-jobset.json',jobset,once=True)
q=subprocess.run(['/opt/slurm/bin/squeue','-h','-r','-j',base,'-o','%i|%T|%R'],capture_output=True,text=True,timeout=25)
assert q.returncode==0 and set(ids).issubset({x.split('|')[0] for x in q.stdout.splitlines()}),(q.returncode,q.stdout,q.stderr)
receipt.update(job_ids=ids,jobset=b['stage']+'/plans/last-jobset.json',initial_queue=q.stdout.strip())
save(b['stage']+'/verification/last_submission_verified.json',receipt,once=True)
print(json.dumps(receipt))
