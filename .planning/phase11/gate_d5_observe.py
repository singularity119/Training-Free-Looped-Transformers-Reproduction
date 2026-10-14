#!/usr/bin/env python3
"""One bounded D5 scheduler/checkpoint observation; never reads scores or gold."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import json,subprocess,sys
from pathlib import Path
s,j=json.loads(sys.argv[1]),json.loads(sys.argv[2]); ids=j['job_ids']; wanted=set(ids)
parents=','.join(sorted({i.split('_')[0] for i in ids}))
acct=subprocess.run(['/opt/slurm/bin/sacct','-n','-P','-j',parents,'--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,Partition,QOS,Start,End'],capture_output=True,text=True,timeout=25)
queue=subprocess.run(['/opt/slurm/bin/squeue','--array','-h','-j',parents,'-o','%i|%T|%M|%R'],capture_output=True,text=True,timeout=25)
if acct.returncode or queue.returncode: raise RuntimeError('scheduler visibility failed: '+acct.stderr+queue.stderr)
states={}
for line in acct.stdout.splitlines():
 f=line.split('|')
 if f[0] in wanted: states[f[0]]={'job_id':f[0],'state':f[1].split()[0],'exit_code':f[2],'elapsed_seconds':int(f[3] or 0),'alloc_tres':f[4],'partition':f[5],'qos':f[6],'start':f[7],'end':f[8]}
for line in queue.stdout.splitlines():
 f=line.split('|')
 if f[0] in wanted and f[0] not in states: states[f[0]]={'job_id':f[0],'state':f[1],'queue_elapsed':f[2],'reason':f[3],'exit_code':None,'elapsed_seconds':0}
missing=sorted(wanted-set(states))
plan=json.loads(Path(j['plan']).read_text()); progress=[]; total=done=0
for task in plan['tasks']:
 for job in task['jobs']:
  n=len(job['indices']); total+=n; checkpoint=Path(job['run_root'])/'checkpoint.json'; count=0
  if checkpoint.exists():
   d=json.loads(checkpoint.read_text()); ix=d['done']
   if not isinstance(ix,list) or len(set(ix))!=len(ix) or not set(ix).issubset(job['indices']): raise ValueError('checkpoint membership differs')
   count=len(ix)
  done+=count
  progress.append({'family':job['family'],'repeat':job['repeat'],'cell_id':job['cell_id'],'run_root':job['run_root'],'done':count,'target':n,'summary_exists':(Path(job['run_root'])/'summary.json').exists()})
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE','REVOKED'}
all_terminal=not missing and all(x['state'] in terminal for x in states.values())
all_success=all_terminal and all(x['state']=='COMPLETED' and x['exit_code']=='0:0' for x in states.values())
failed=[x for x in states.values() if x['state'] in terminal and (x['state']!='COMPLETED' or x['exit_code']!='0:0')]
v={'gate':'D-5','wait_class':j['wait_class'],'job_ids':ids,'scheduler':list(states.values()),'missing_job_ids':missing,'all_terminal':all_terminal,'all_success':all_success,'terminal_failures':failed,'done_record_count':done,'target_count':total,'progress':progress,'target_gold_loaded':False}
print(json.dumps(v,ensure_ascii=False))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    setup = json.loads((root / "gate_d5_setup.json").read_text())
    jobset = json.loads(args.jobset.read_text())
    cmd = shlex.quote(setup["python"]) + " - " + shlex.quote(json.dumps(setup)) + " " + shlex.quote(json.dumps(jobset))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ClearAllForwardings=yes", "hpc2-hkustgz", cmd], input=REMOTE, capture_output=True, text=True, timeout=70)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    value = json.loads(result.stdout)
    value["observed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value["executor"] = setup["executor"]
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "progress"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
