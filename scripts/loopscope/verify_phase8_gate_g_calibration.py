#!/usr/bin/env python3
"""Outcome-free independent closure of Gate G's three t1 and shared t0 bases."""
import argparse
import json
from pathlib import Path
from verify_phase8_gate_f_calibration import read, require, spectral_check, check_positions
from run_phase8_gate_g_calibration import select_rows


def verify(root, pool, scope='FORMAL_CALIBRATION'):
    import torch
    from tflt.loopscope.phase8_gate_g_calibration import MODEL, REVISION, load_t0_basis, verify_saved_t0
    from tflt.loopscope.phase8_runtime import load_basis
    from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
    root, pool = Path(root), Path(pool)
    bundle = read(pool)
    require(bundle['dataset'] == dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'), 'dataset mismatch')
    identities = [r['identity'] for r in select_rows(bundle, MODEL, scope)]
    summary, env = read(root/'summary.json'), read(root/'env.json')
    require(summary['status'] == 'CALIBRATION_COMPLETE' and summary['identities'] == identities
            and summary['target_gold_loaded'] is False and summary['choice_scores_saved'] is False,
            'calibration not complete or outcome-bearing')
    require(env['source_commit'] == summary['metadata']['source_commit']
            and env['model_revision'] == REVISION and env['model_dtype'] == 'torch.bfloat16', 'actual runtime mismatch')
    saved_by_k, results = {}, []
    for k in (2,3,4):
        cell_root = root/f'k{k}'
        saved = torch.load(cell_root/'residuals.pt', map_location='cpu', weights_only=True)
        meta = saved['metadata']
        expected = dict(scope=scope, model=MODEL, revision=REVISION, window=[15,18], k=k,
                        dtype='bfloat16', cache_strategy='first', alpha=1., dataset=bundle['dataset'],
                        pool=str(pool), run_root=str(cell_root), source_commit=env['source_commit'])
        require(meta == expected and saved['identities'] == identities, 'residual provenance mismatch')
        require(set(saved['matrices']) == set(saved['positions']) == {str(t) for t in range(k)}, 'missing timesteps')
        basis = load_basis(cell_root/'basis.json', expected)
        require(basis['fit_t'] == 1 and basis['rank'] == 1 and basis['lambda'] == .5
                and basis['source_identities'] == identities and basis['row_count'] == len(identities)
                and basis['algorithm'] == 'cpu_float64_reduced_svd_uncentered_unnormalized', 't1 recipe mismatch')
        matrix = saved['matrices']['1']
        require(matrix.ndim == 2 and matrix.shape == (len(identities), len(basis['direction'])), 'matrix shape mismatch')
        for t in range(k):
            value, positions = saved['matrices'][str(t)], saved['positions'][str(t)]
            require(value.shape == matrix.shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'invalid residual')
            require(len(positions) == len(identities) and all(type(p) is int and p >= 0 for p in positions)
                    and positions == saved['positions']['0'], 'invalid positions')
        check_positions(cell_root/'positions.json', saved, identities)
        saved_by_k[k] = saved
        require(torch.equal(saved_by_k[2]['matrices']['0'], saved['matrices']['0'])
                and saved_by_k[2]['positions']['0'] == saved['positions']['0'], 't0 cross-K dependence')
        results.append(dict(k=k, fit_t=1, **spectral_check(matrix, basis, torch)))
    source = saved_by_k[2]
    matrix = verify_saved_t0(source, source['metadata'], identities, scope)
    bases = [load_t0_basis(root/'t0/basis.json', dict(source['metadata'], k=k), scope) for k in (2,3,4)]
    require(all(b['direction'] == bases[0]['direction'] for b in bases), 'shared t0 changed')
    require(bases[0]['metadata'] == source['metadata'] and bases[0]['source_identities'] == identities
            and bases[0]['source_positions'] == source['positions']['0'], 't0 source mismatch')
    results.append(dict(source_k=2, applies_to_k=[2,3,4], fit_t=0, **spectral_check(matrix, bases[0], torch)))
    if scope == 'PREFLIGHT_ONLY':
        checks = read(root/'runtime_checks.json')
        require(len(checks['intervention']) == 6 and all(c['verified'] and c['call_count'] > 0
                for c in checks['intervention'].values()) and all(checks['tensor_checks'].values()), 'preflight intervention incomplete')
    return dict(status='VERIFIED_CALIBRATION', scope=scope, t1_count=3, t0_count=1,
                row_count=len(identities), cells=results, target_gold_loaded=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=('PREFLIGHT_ONLY','FORMAL_CALIBRATION'), default='FORMAL_CALIBRATION')
    a = p.parse_args(argv)
    result = verify(a.root, a.pool, a.scope)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
