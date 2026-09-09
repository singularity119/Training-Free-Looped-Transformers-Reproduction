#!/usr/bin/env python3
"""Fresh gold-free closure of the nine new Gate G cells."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_gate_g_verify import verify
from tflt.loopscope.phase8_accuracy import write_json


def close(g_path):
    old = json.loads(Path(g_path).read_text())
    if old['status'] != 'FULL_PANEL_CLOSED' or old['target_gold_loaded'] is not False:
        raise ValueError('requires saved complete gold-free closure')
    fresh = verify(old['cell_roots'], old['pool'], old['manifest'], True)
    manifest = json.loads(Path(fresh['manifest']).read_text())
    return dict(status='GATE_G_FULL_LOGICAL_PANEL_CLOSED', target_gold_loaded=False,
        pool=fresh['pool'], cell_count=9, n_per_cell=14042,
        source_closure=str(g_path), cells=[dict(cell=cell, gate='G',
            roots=[s['root'] for s in fresh['shards'] if s['cell_id']==cell['cell_id']],
            source_manifest=fresh['manifest']) for cell in manifest['cells']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g-closure', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = close(a.g_closure)
    write_json(a.output, result)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}))
