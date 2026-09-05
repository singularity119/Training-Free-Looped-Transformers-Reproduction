#!/usr/bin/env python3
"""Verify a completed Gate B debug root without target labels or outcomes."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def verify(root):
    from tflt.loopscope.phase8_runtime import load_basis
    summary = read(root / 'summary.json')
    env = read(root / 'env.json')
    recipe = summary['recipe']
    assert summary['status'] == 'DEBUG_COMPLETE' and summary['scope'] == 'SMOKE_ONLY'
    assert env['partition'] == 'debug' and env['job_id']
    assert env['source_commit'] == recipe['source_commit']
    assert summary['oom_count'] == 0 and summary['elapsed_seconds'] < 29*60
    assert summary['fit_identity_count'] == summary['verified_identity_count'] == 4
    assert all(summary['semantic_checks'].values())
    native = read(root / 'native.json')
    assert len(native) == 4 and len({x['identity'] for x in native}) == 4
    for row in native:
        assert row['scores'] == row['adapter_scores']
        assert len(row['scores']) == 4 and all(math.isfinite(x) for x in row['scores'])
    expected = ['w%d-%d-k%d' % (*w,k) for w,k in recipe['cells']]
    assert summary['cell_names'] == expected
    for name, (window,k) in zip(expected,recipe['cells']):
        cell = read(root / name / 'debug.json')
        basis = load_basis(root / name / 'basis.json',cell['metadata'])
        assert basis['k'] == k and basis['row_count'] == 4 and basis['fit_t'] == 1
        assert basis['metadata']['scope'] == 'SMOKE_ONLY'
        assert basis['metadata']['window'] == window
        assert basis['metadata']['model'] == recipe['model']['model']
        assert cell['collector']['rows_by_t'] == {str(t):4 for t in range(k)}
        assert cell['collector']['row_count'] == 4*k
        assert cell['basis_roundtrip'] and (root / name / 'residual_t1.pt').is_file()
        assert [r['identity'] for r in cell['results']] == [r['identity'] for r in native]
        assert set(basis['source_identities']).isdisjoint(r['identity'] for r in native)
        for row in cell['results']:
            assert row['original'] == row['none'] == row['zero']
            assert all(len(row[field]) == 4 and all(math.isfinite(x) for x in row[field])
                       for field in ('original','none','zero','spectral'))
            assert {c['t'] for c in row['calls']} == set(range(k))
            assert all(c['identity'] == row['identity'] and c['applied'] == (c['t']>=1)
                       for c in row['calls'])
            assert len(row['positions']) == 4
            assert all(p['position'] == p['input_length']-p['continuation_length']
                       and p['position'] >= 0 for p in row['positions'])
    return dict(status='VERIFIED_DEBUG', run_root=str(root), cells=expected,
                model=recipe['model']['model'], source_commit=recipe['source_commit'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.run_root),indent=2))
