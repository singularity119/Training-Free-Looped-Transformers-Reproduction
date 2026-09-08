#!/usr/bin/env python3
"""Close 16 new plus 20 retained cells before any Gate F outcome computation."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_gate_f_verify import verify as verify_f
from tflt.loopscope.phase8_accuracy_verify import verify as verify_old
from tflt.loopscope.phase8_gate_f_accuracy import logical_panel, scientific_key
from tflt.loopscope.phase8_accuracy import write_json


def close(f_path,d_path,e_path):
    receipts=[]
    for path, kind in ((f_path,'F'),(d_path,'D'),(e_path,'E')):
        old=json.loads(Path(path).read_text())
        if old['status'] != 'FULL_PANEL_CLOSED' or old['target_gold_loaded'] is not False:
            raise ValueError('requires saved complete gold-free closure')
        if kind=='F': fresh=verify_f(old['cell_roots'],old['pool'],old['manifest'],True)
        else: fresh=verify_old(old['cell_roots'],old['pool'],old['manifest'],True,gate_e=kind=='E')
        receipts.append((kind,fresh))
    if len({r['pool'] for _,r in receipts})!=1:
        raise ValueError('retained pools must use same canonical source')
    expected={scientific_key(c):c for c in logical_panel()}
    entries={}
    for kind,receipt in receipts:
        manifest=json.loads(Path(receipt['manifest']).read_text())
        for cell in manifest['cells']:
            if cell['arm']=='Native': continue
            key=scientific_key(cell)
            if key not in expected or key in entries:
                raise ValueError('duplicate or unexpected logical cell')
            frozen=expected[key]
            if any(cell[k]!=frozen[k] for k in ('model','revision','window','k','arm','dtype','cache_strategy')):
                raise ValueError('reused recipe differs')
            roots=[s['root'] for s in receipt['shards'] if s['cell_id']==cell['cell_id']]
            entries[key]=dict(cell=cell,gate=kind,roots=roots,source_manifest=receipt['manifest'])
    if set(entries)!=set(expected) or len(entries)!=36:
        raise ValueError('all36 logical cells required')
    counts={g:sum(e['gate']==g for e in entries.values()) for g in ('F','D','E')}
    if counts!={'F':16,'D':16,'E':4}: raise ValueError('16new/20retained membership differs')
    return dict(status='GATE_F_FULL_LOGICAL_PANEL_CLOSED',target_gold_loaded=False,
        pool=receipts[0][1]['pool'],cell_count=36,n_per_cell=14042,counts_by_gate=counts,
        source_closures=dict(F=str(f_path),D=str(d_path),E=str(e_path)),
        cells=[entries[k] for k in expected])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('f-closure','d-closure','e-closure','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=close(a.f_closure,a.d_closure,a.e_closure)
    write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}))
