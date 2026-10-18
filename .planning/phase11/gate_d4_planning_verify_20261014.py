"""Read-only D4 acceptance: raw closure, fixed S2/statistics, and exact jobs.
Run via SSH stdin with pinned Python; only aggregate JSON goes to local stdout.
"""
import collections,datetime,importlib.util,json,math,re,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
root=Path('/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope')
src=root/'staging/phase11-gate-d4-final-s2-20261013T092046Z/source'
out=root/'artifacts/phase11-gate-d4-final-20261013T141800Z'
sys.path.insert(0,str(src/'src'))
spec=importlib.util.spec_from_file_location('final_analyzer',src/'scripts/loopscope/analyze_phase11_gpqa_8192.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
read=lambda p:json.loads(Path(p).read_text())
a=read(out/'analysis/analysis.json');original_closure=read(out/'analysis/raw_closure.json');barrier=read(out/'analysis/extraction_barrier.json');ss=read(out/'analysis/s2_spec.json');settlement=read(out/'scheduler_settlement.json')
assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()=='f40cd50247dae1d4c196e9252acb43c6119e70e9'
assert subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True).strip()==''
closure,panel,aligned=mod.close_raw_panel(a['paths']['membership'],read(a['paths']['cell_roots']),a['source_commits'])
assert closure['record_counts']==original_closure['record_counts'] and closure['identities']==original_closure['identities']
assert original_closure['timestamp']<ss['frozen_at_utc']<barrier['timestamp']<a['gold_read_timestamp']
# Independently spell the frozen S2 whole-line rule, then compare every saved prediction.
pattern=r'final[ \t]+answer[ \t]*[:：][ \t]*(?:\([ \t]*([A-D])[ \t]*\)|([A-D]))[ \t]*[.!;,．！。]?'
saved={}
for line in (out/'analysis/predictions.jsonl').open():
 d=json.loads(line);key=(d['cell_id'],d['identity']);assert key not in saved;saved[key]=d
assert len(saved)==8064
predictions={};modecounts=collections.Counter();ends=collections.Counter();risk=collections.Counter()
for cid,rs in aligned.items():
 predictions[cid]=[]
 for r in rs:
  matches=[];in_fence=None;fenced=[];lines=r['generated_text'].splitlines()
  for i,line in enumerate(lines):
   t=line.strip();fm=re.match(r'^(`{3,}|~{3,})',t)
   if fm:in_fence=None if in_fence==fm[1][0] else fm[1][0]
   m=re.fullmatch(pattern,t,re.I)
   if m:matches.append((i,(m[1] or m[2]).upper()));fenced.append(bool(in_fence))
  labels={label for _,label in matches};label=next(iter(labels)) if len(labels)==1 else None
  prediction='ABCD'.index(label) if label else None
  s=saved[(cid,r['identity'])]['S2'];assert prediction==s['prediction_index'] and label==s['predicted_label']
  assert len(matches)==s['valid_final_line_count']
  predictions[cid].append(prediction);modecounts[s['answer_mode']]+=1;ends[r['end_reason']]+=1
  if label:
   risk['accepted']+=1
   risk['accepted_in_fence']+=int(any(fenced))
   risk['accepted_length_limit']+=int(r['end_reason']=='length_limit')
   tail='\n'.join(lines[matches[-1][0]+1:]).strip()
   meaningful=re.sub(r'[\s`~*_#<>/\-]+','',tail)
   risk['nonempty_tail']+=int(bool(meaningful))
   differing=[]
   for line in tail.splitlines():
    m=re.search(r'(?i)final\s+answer\s*[:：]\s*(?:\*\*)?\(?\s*([A-D])\b',line)
    if m and m[1].upper()!=label:differing.append(m[1].upper())
   risk['different_final_answer_surface_after']+=int(bool(differing))
# Gold is read only after fresh membership and prediction verification; no scores are written remotely.
goldrows=[json.loads(l) for l in Path(a['paths']['gold']).open() if l.strip()];gold={}
for r in goldrows:
 assert r['identity'] not in gold;g=r.get('label_index');g='ABCD'.index(r['gold_letter']) if g is None else g;assert type(g)==int and 0<=g<4;gold[r['identity']]=g
assert set(gold)==set(closure['identities'])
correct={cid:np.array([int(p is not None and p==gold[r['identity']]) for p,r in zip(predictions[cid],rs)]) for cid,rs in aligned.items()}
cs={c['cell_id']:c for c in a['s2']['cells']}
for cid,v in correct.items():
 c=cs[cid];assert len(v)==c['n']==448 and int(v.sum())==c['correct'];assert abs(float(v.mean())-c['accuracy'])<1e-12
 assert c['extraction_success_count']==sum(p is not None for p in predictions[cid])
contrasts=a['s2']['contrasts'];assert len(contrasts)==44
max_p_error=0.;max_ci_error=0.;computed=[]
for c in contrasts:
 ref=correct[c['reference']];trt=correct[c['treatment']];diff=trt-ref;g=int((diff==1).sum());l=int((diff==-1).sum());assert g==c['wrong_to_right'] and l==c['right_to_wrong']
 assert c['n']==448 and abs((g-l)/448*100-c['delta_pp'])<1e-12
 p=binomtest(g,g+l,0.5).pvalue if g+l else 1.;max_p_error=max(max_p_error,abs(p-c['mcnemar_exact_p']));assert abs(p-c['mcnemar_exact_p'])<1e-10
 draws=np.random.default_rng(20261008).multinomial(448,np.array([l,448-g-l,g],dtype=float)/448,size=10000)
 lo,hi=np.quantile((draws[:,2]-draws[:,0])*(100./448),[.025,.975]);err=max(abs(lo-c['bootstrap']['ci_low_pp']),abs(hi-c['bootstrap']['ci_high_pp']));max_ci_error=max(max_ci_error,err);assert err<1e-10
 computed.append(p)
familycounts=collections.Counter(c['family'] for c in contrasts);assert sorted(familycounts.values())==[2,12,30]
for fam in familycounts:
 ids=[i for i,c in enumerate(contrasts) if c['family']==fam];running=0.
 for rank,i in enumerate(sorted(ids,key=lambda i:computed[i])):
  running=max(running,(len(ids)-rank)*computed[i]);adj=min(1.,running);assert abs(adj-contrasts[i]['holm_adjusted_p'])<1e-10
# One exact job-set scheduler check; do not query unrelated jobs or change scheduler state.
jobs=settlement['jobs'];ids=[x['job_id'] for x in jobs];assert len(ids)==len(set(ids))==290
cmd=['/opt/slurm/bin/sacct','-n','-P','-X','-j',','.join(ids),'--format=JobID%50,State,ExitCode,ElapsedRaw,AllocTRES,Partition,QOS,Start,End']
p=subprocess.run(cmd,text=True,capture_output=True,timeout=45);assert p.returncode==0,p.stderr
actual={}
for line in p.stdout.splitlines():
 parts=line.split('|');jid=parts[0]
 if jid in ids:actual[jid]=parts
assert set(actual)==set(ids)
for j in jobs:
 v=actual[j['job_id']];assert v[1]=='COMPLETED' and v[2]=='0:0';assert int(v[3])==j['allocation_seconds'];assert v[4]==j['alloc_tres'] and v[5]==j['partition'] and v[6]==j['qos']
 assert v[7]==j['start'] and v[8]==j['end']
q=subprocess.run(['/opt/slurm/bin/squeue','-h','-u','xhuang225','-o','%i'],text=True,capture_output=True,timeout=20);assert q.returncode==0,q.stderr
assert not set(q.stdout.split())&set(ids)
costs=collections.Counter();events=[]
for j in jobs:
 assert re.search(r'(?:^|,)gres/gpu=1(?:,|$)',j['alloc_tres']);assert 'cpu=8,' in j['alloc_tres'];costs[j['class']]+=j['allocation_seconds']/3600
 if j['class']=='formal':
  assert j['partition']==j['qos']=='emergency_gpu' and 'gres/gpu:a800=1' in j['alloc_tres'];events.extend([(j['start'],1),(j['end'],-1)])
active=peak=0
for _,delta in sorted(events,key=lambda x:(x[0],x[1])):active+=delta;peak=max(peak,active)
assert peak==8 and active==0
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'D4_DECISIVE_CHECKS_PASSED','analysis_source':str(src),'analysis_commit':'f40cd50247dae1d4c196e9252acb43c6119e70e9','record_count':8064,'cell_count':18,'samples_per_cell':448,'fresh_raw_membership_and_lifecycle_verified':True,'frozen_s2_predictions_matching':8064,'cells_independently_recounted':18,'contrasts_independently_recomputed':44,'family_counts':dict(familycounts),'max_exact_p_error':max_p_error,'max_ci_error_pp':max_ci_error,'holm_rejections':sum(c['holm_reject_alpha_0_05'] for c in contrasts),'minimum_holm_p':min(c['holm_adjusted_p'] for c in contrasts),'original_barrier_order_verified':True,'raw_closure_time':original_closure['timestamp'],'extraction_barrier_time':barrier['timestamp'],'gold_read_time':a['gold_read_timestamp'],'s2_mode_counts':dict(modecounts),'end_reasons':dict(ends),'parser_boundary_screen_full_population':dict(risk),'scheduler_exact_jobs':len(jobs),'scheduler_all_completed_zero':True,'inflight_exact_jobs':0,'costs_gpuh':dict(costs),'peak_formal_a800':peak,'gold_and_predictions_exported':False,'remote_writes':False,'gpu_execution':False}
print(json.dumps(result,ensure_ascii=False,indent=2))
