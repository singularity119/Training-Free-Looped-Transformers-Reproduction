"""Frozen Gate D panel and canonical, gold-free acquisition membership."""
from pathlib import Path
import json
from .phase8_pool import DATASET_REPO, DATASET_REVISION

W='/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope'
BASIS_ROOTS=[W+'/runs/phase8-gate-c-20260905T100000Z-formal-q4-a1',W+'/runs/phase8-gate-c-20260905T100000Z-formal-q17-a1']

def panel(config):
    cells=[]
    for index,spec in enumerate(config['models']):
        prefix=('q4','q17')[index]
        base=dict(model_index=index,model=spec['model'],revision=spec['observed_revision'],dtype=spec['historical_runtime_dtype'],cache_strategy=spec['cache_strategy'])
        cells.append(dict(base,cell_id=prefix+'-native',window=None,k=None,arm='Native',basis_path=None))
        for wi,w in enumerate(spec['windows']):
            for ki,k in enumerate((2,4)):
                for arm in ('Loop','Spectral'):
                    cells.append(dict(base,cell_id=f'{prefix}-w{w["layers"][0]}-{w["layers"][1]}-k{k}-{arm.lower()}',window=w['layers'],k=k,arm=arm,basis_path=f'{BASIS_ROOTS[index]}/cell-{wi*2+ki}/basis.json' if arm=='Spectral' else None))
    return cells

def select_rows(pool,cell,scope,start,end):
    rows=pool['rows']
    if len(rows)!=14042 or pool['dataset']!={'repo':DATASET_REPO,'revision':DATASET_REVISION,'split':'test'}:
        raise ValueError('requires canonical frozen full test pool')
    if scope=='PREFLIGHT_ONLY':
        selected={0,1}
        selected.update(sorted(range(len(rows)),key=lambda i:(-rows[i]['prompt_token_lengths'][cell['model']],i))[:4])
        return [rows[i] for i in sorted(selected)]
    if not 0<=start<end<=len(rows):
        raise ValueError('invalid canonical shard bounds')
    return rows[start:end]

def checked_basis(cell):
    from .phase8_runtime import load_basis
    basis=load_basis(cell['basis_path'])
    expected=dict(scope='FORMAL_CALIBRATION',model=cell['model'],revision=cell['revision'],window=cell['window'],k=cell['k'],dtype=cell['dtype'],cache_strategy=cell['cache_strategy'],alpha=1.0,dataset={'repo':DATASET_REPO,'revision':DATASET_REVISION,'split':'validation'})
    if any(basis['metadata'].get(key)!=value for key,value in expected.items()) or basis['row_count']!=512 or basis['rank']!=1 or basis['lambda']!=0.5:
        raise ValueError('basis does not match frozen formal calibration cell')
    return basis

def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
