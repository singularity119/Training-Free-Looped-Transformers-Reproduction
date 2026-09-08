#!/usr/bin/env python3
"""One Gate F analysis after complete new and retained raw-score closure."""
import argparse
import json
import os
from pathlib import Path
from close_phase8_gate_f_panel import close
from analyze_phase8_accuracy import read_json, write_json, now, load_frozen_gold, independent_count_check, write_tables
from tflt.loopscope.phase8_gate_f_accuracy import analyze_panel

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--closure',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--cache-dir',default=os.environ.get('HF_DATASETS_CACHE'))
a=p.parse_args()
old=read_json(a.closure)
if old['status']!='GATE_F_FULL_LOGICAL_PANEL_CLOSED' or old['target_gold_loaded'] is not False:
    raise ValueError('complete Gate F closure required before gold')
sources=old['source_closures']
fresh=close(sources['F'],sources['D'],sources['E'])
if fresh!=old: raise ValueError('closure differs on fresh verification')
a.output_dir.mkdir(parents=True,exist_ok=False)
write_json(a.output_dir/'pre_outcome_verification.json',dict(fresh,saved_at=now()))
rows=read_json(fresh['pool'])['rows']
cells=[]
for entry in fresh['cells']:
    scores={}
    for root in entry['roots']:
        with (Path(root)/'scores.jsonl').open() as handle:
            for line in handle:
                row=json.loads(line)
                if row['identity'] in scores: raise ValueError('duplicate score identity')
                scores[row['identity']]=row['scores']
    cells.append(dict(entry['cell'],scores=[scores[row['identity']] for row in rows]))
gold_started=now()
gold,evidence=load_frozen_gold(rows,a.cache_dir)
subjects=[r['subject'] for r in rows]
result=analyze_panel(cells,gold,subjects)
result.update(schema='loopscope.phase8.gate_f.analysis.v1',closure=str(a.closure),gold_load_started=gold_started,gold_source_evidence=evidence)
result['fresh_count_verification']=independent_count_check(cells,gold,subjects,result,families=('T0_MINUS_T1_EXPLORATORY','T0_MINUS_LOOP_EXPLORATORY'))
result['completed_at']=now()
write_json(a.output_dir/'analysis.json',result)
write_tables(a.output_dir,result)
print(json.dumps(dict(status='ANALYZED',cells=36,contrasts=len(result['contrasts']),output_dir=str(a.output_dir))))
