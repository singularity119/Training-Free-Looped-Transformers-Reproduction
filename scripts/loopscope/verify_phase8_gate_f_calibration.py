#!/usr/bin/env python3
"""Verify all four shared t0 and two K3 t1 formal calibrations without outcomes."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spectral_check(matrix, basis, torch):
    d = matrix.double()
    v = torch.tensor(basis['direction'], dtype=torch.float64)
    v = v / torch.linalg.vector_norm(v)
    dv = d @ v
    sigma = basis['diagnostics']['singular_value_1']
    require(math.isfinite(sigma) and sigma > 0, 'invalid singular value')
    energy = float(d.square().sum())
    require(energy > 0, 'zero residual energy')
    rayleigh = float(torch.dot(dv, dv))
    errors = dict(rayleigh_relative_error=abs(rayleigh-sigma*sigma)/(sigma*sigma),
        eigen_residual_relative_error=float(torch.linalg.vector_norm(d.T@dv-sigma*sigma*v))/(sigma*sigma),
        top1_energy_fraction_absolute_error=abs(rayleigh/energy-basis['diagnostics']['top1_energy_fraction']))
    require(all(math.isfinite(x) and x <= 1e-4 for x in errors.values()), 'spectral verification failed')
    return errors


def check_positions(path, saved, identities):
    records = read(path)
    require([r['identity'] for r in records] == identities, 'scorer position identity mismatch')
    require(all(r['candidates'] and all(c['position'] == saved['positions']['0'][i]
                for c in r['candidates']) for i, r in enumerate(records)), 'scorer context position mismatch')


def verify(root, pool):
    import torch
    from tflt.loopscope.phase8_gate_f_basis import load_t0_basis, verify_saved_t0
    from tflt.loopscope.phase8_gate_f_accuracy import MODELS
    from tflt.loopscope.phase8_runtime import load_basis, identity_key
    from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
    root, pool = Path(root), Path(pool)
    bundle = read(pool)
    identities = bundle['calibration_identities']
    require(len(identities) == len(set(map(identity_key, identities))) == 512, 'pool requires 512 unique identities')
    require([r['identity'] for r in bundle['rows']] == identities, 'pool row order mismatch')
    require(bundle['dataset'] == dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'), 'pool dataset mismatch')
    results = []
    for fit_t, count in ((0, 4), (1, 2)):
        for index in range(count):
            cell_root = root / ('t0' if fit_t == 0 else 't1') / f'cell-{index}'
            summary = read(cell_root / 'summary.json')
            command = read(cell_root / 'command_args.json')
            metadata = summary['metadata']
            mi, wi = (index//2, index%2) if fit_t == 0 else (1, index)
            model, revision, dtype, cache, windows, _ = MODELS[mi]
            expected = dict(scope='FORMAL_CALIBRATION', model=model, revision=revision,
                dtype=dtype, cache_strategy=cache, window=list(windows[wi]), k=2 if fit_t == 0 else 3,
                alpha=1., dataset=bundle['dataset'], pool=str(pool), run_root=str(cell_root))
            require(all(metadata.get(key) == value for key,value in expected.items())
                    and metadata.get('source_commit'), 'formal fitting recipe/provenance mismatch')
            require(command['metadata'] == metadata and summary['status'] == 'CALIBRATION_COMPLETE'
                    and summary['identities'] == identities and summary['target_gold_loaded'] is False
                    and summary['choice_scores_saved'] is False, 'incomplete or outcome-bearing calibration')
            basis_path = cell_root / 'basis.json'
            if fit_t == 0:
                source = Path(summary['saved_residuals'])
                require(command['saved_residuals'] == str(source) and summary['source_identity_count'] == 512,
                        't0 saved-source provenance mismatch')
                saved = torch.load(source, map_location='cpu', weights_only=True)
                matrix = verify_saved_t0(saved, summary['source_metadata'], identities)
                bases = [load_t0_basis(basis_path, dict(expected, k=k)) for k in (2,3,4)]
                basis = bases[0]
                require(basis['metadata'] == saved['metadata'] and basis['fit_provenance'] == metadata
                        and basis['source_identities'] == identities
                        and basis['source_positions'] == saved['positions']['0'], 't0 basis source/fitting provenance mismatch')
                require(all(b['direction'] == basis['direction'] for b in bases)
                        and summary['cross_k_load_verified'] == [2,3,4], 't0 shared direction load mismatch')
                if (source.parent / 'positions.json').exists():
                    check_positions(source.parent / 'positions.json', saved, identities)
            else:
                basis = load_basis(basis_path, metadata)
                require(basis['fit_t'] == 1 and basis['k'] == 3 and basis['rank'] == 1
                        and basis['lambda'] == .5 and basis['source_identities'] == identities
                        and basis['row_count'] == 512
                        and basis['algorithm'] == 'cpu_float64_reduced_svd_uncentered_unnormalized', 't1 basis recipe mismatch')
                saved = torch.load(cell_root / 'residuals.pt', map_location='cpu', weights_only=True)
                require(saved['metadata'] == metadata and saved['identities'] == identities
                        and set(saved['matrices']) == set(saved['positions']) == {'0','1','2'}, 't1 residual provenance mismatch')
                matrix = saved['matrices']['1']
                require(matrix.ndim == 2 and matrix.shape[0] == 512 and matrix.shape[1] == len(basis['direction']), 't1 residual shape mismatch')
                for t in ('0','1','2'):
                    value, positions = saved['matrices'][t], saved['positions'][t]
                    require(value.shape == matrix.shape and value.dtype == torch.float32
                            and bool(torch.isfinite(value).all()), 'invalid t1 residual tensor')
                    require(len(positions) == 512 and all(type(p) is int and p >= 0 for p in positions)
                            and positions == saved['positions']['0'], 'invalid t1 position')
                check_positions(cell_root / 'positions.json', saved, identities)
                env = read(cell_root / 'env.json')
                require(env['source_commit'] == metadata['source_commit'] and env['model_revision'] == revision
                        and env['model_dtype'] == 'torch.'+dtype, 't1 actual runtime mismatch')
            results.append(dict(basis_path=str(basis_path), fit_t=fit_t, model=model, window=list(windows[wi]),
                row_count=512, **spectral_check(matrix, basis, torch)))
    return dict(status='VERIFIED_CALIBRATION', scope='FORMAL_CALIBRATION', t0_count=4, t1_count=2,
        pool=str(pool), root=str(root), relative_tolerance=1e-4, target_gold_loaded=False, cells=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(args.root, args.pool)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
