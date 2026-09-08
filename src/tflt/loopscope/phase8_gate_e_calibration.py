"""Independent, outcome-free verification of saved Gate E 4B K3 calibration cells.

Torch is imported only when verifying tensor artifacts, not at module import.
"""
import json
import math
from pathlib import Path

from .phase8_runtime import identity_key, load_basis
from .phase8_pool import DATASET_REPO, DATASET_REVISION


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_metadata(metadata, scope, pool_path, config):
    """Check the frozen cell recipe independently of the producer."""
    _require(metadata['scope'] == scope, 'basis scope differs from requested scope')
    model = next((m for m in config['models'] if m['model'] == metadata['model']
                  and m['model'] == 'Qwen/Qwen3-4B-Base'), None)
    _require(model is not None, 'unknown model')
    _require(metadata['revision'] == model['observed_revision'], 'model revision mismatch')
    _require(metadata['dtype'] == model['historical_runtime_dtype'], 'dtype mismatch')
    _require(metadata['cache_strategy'] == model['cache_strategy'], 'cache strategy mismatch')
    _require(metadata['window'] in [w['layers'] for w in model['windows']], 'window mismatch')
    _require(metadata['k'] == 3 and metadata['alpha'] == 1, 'K/alpha mismatch')
    _require(metadata['dataset'] == {'repo': DATASET_REPO, 'revision': DATASET_REVISION,
                                   'split': 'validation'}, 'dataset mismatch')
    _require(metadata['pool'] == str(pool_path), 'pool source mismatch')
    _require(bool(metadata['source_commit']) and bool(metadata['run_root']), 'missing provenance')
    return (metadata['model'], tuple(metadata['window']), metadata['k'])


def verify(cell_roots, pool, scope, require_all=False):
    import torch
    _require(scope in ('FORMAL_CALIBRATION', 'PREFLIGHT_ONLY'), 'invalid scope')
    pool = Path(pool)
    bundle = _read(pool)
    identities = bundle.get('calibration_identities', bundle.get('identities'))
    _require(isinstance(identities, list) and len(identities) == 512, 'pool must identify 512 rows')
    _require(len({identity_key(i) for i in identities}) == 512, 'duplicate pool identities')
    config = _read(Path(__file__).resolve().parents[3] / 'configs/loopscope/phase8_model_windows.json')
    cells, seen = [], set()
    for root in map(Path, cell_roots):
        summary = _read(root / 'summary.json')
        _require(summary['status'] == 'CALIBRATION_COMPLETE', 'incomplete cell')
        metadata = summary['metadata']
        key = verify_metadata(metadata, scope, pool, config)
        _require(metadata['run_root'] == str(root), 'run root provenance mismatch')
        env = _read(root / 'env.json')
        _require(env['source_commit'] == metadata['source_commit']
                 and env['model_revision'] == metadata['revision']
                 and env['model_dtype'] == 'torch.' + metadata['dtype'],
                 'actual runtime dtype/revision/source mismatch')
        _require(key not in seen, 'duplicate cell')
        seen.add(key)
        expected = identities if scope == 'FORMAL_CALIBRATION' else summary['identities']
        _require(summary['identities'] == expected, 'summary identity order mismatch')
        _require(bool(expected) and len({identity_key(i) for i in expected}) == len(expected),
                 'empty or duplicate cell identities')
        _require(all(i in identities for i in expected), 'preflight identity outside calibration pool')
        basis = load_basis(root / 'basis.json', metadata)
        _require(basis['source_identities'] == expected and basis['row_count'] == len(expected),
                 'basis source identity mismatch')
        _require(basis['k'] == key[2] and basis['fit_t'] == 1 and basis['rank'] == 1
                 and basis['lambda'] == 0.5, 'basis fit recipe mismatch')
        _require(basis['algorithm'] == 'cpu_float64_reduced_svd_uncentered_unnormalized',
                 'basis algorithm mismatch')
        saved = torch.load(root / 'residuals.pt', map_location='cpu', weights_only=True)
        _require(saved['metadata'] == metadata and saved['identities'] == expected,
                 'residual provenance or identity mismatch')
        k, n, width = key[2], len(expected), len(basis['direction'])
        _require(set(saved['matrices']) == set(saved['positions']) == {str(t) for t in range(k)},
                 'residual timestep mismatch')
        for t in range(k):
            matrix = saved['matrices'][str(t)]
            _require(matrix.dtype == torch.float32 and tuple(matrix.shape) == (n, width)
                     and bool(torch.isfinite(matrix).all()), 'invalid residual shape/dtype/finite')
            positions = saved['positions'][str(t)]
            _require(len(positions) == n and all(type(p) is int and p >= 0 for p in positions),
                     'invalid answer positions')
            _require(positions == saved['positions']['0'], 'answer position changes across timesteps')
        collector = summary['collector']
        _require(collector['identity_count'] == n and collector['k'] == k
                 and collector['row_count'] == n*k
                 and collector['rows_by_t'] == {str(t): n for t in range(k)}, 'collector count mismatch')
        d = saved['matrices']['1'].double()
        v = torch.tensor(basis['direction'], dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(v))
        v = v / norm
        dv = d @ v
        av = d.T @ dv
        rayleigh = float(torch.dot(dv, dv))
        sigma = basis['diagnostics']['singular_value_1']
        _require(math.isfinite(sigma) and sigma > 0 and rayleigh > 0, 'invalid top singular energy')
        relative_rayleigh = abs(rayleigh - sigma*sigma) / (sigma*sigma)
        relative_eigen = float(torch.linalg.vector_norm(av - sigma*sigma*v)) / (sigma*sigma)
        energy = float(torch.sum(d.square()))
        fraction = rayleigh / energy
        fraction_error = abs(fraction-basis['diagnostics']['top1_energy_fraction'])
        _require(relative_rayleigh <= 1e-4 and relative_eigen <= 1e-4 and fraction_error <= 1e-4,
                 'basis spectral verification failed')
        cells.append({'root': str(root), 'model': key[0], 'window': list(key[1]), 'k': k,
                      'identity_count': n, 'hidden_width': width, 'source_commit': metadata['source_commit'],
                      'basis_path': str(root / 'basis.json'), 'unit_norm': norm,
                      'rayleigh_relative_error': relative_rayleigh,
                      'eigen_residual_relative_error': relative_eigen,
                      'top1_energy_fraction_absolute_error': fraction_error})
    _require(bool(cells), 'no cells supplied')
    if require_all:
        expected_cells = {(m['model'], tuple(w['layers']), k) for m in config['models']
                          if m['model'] == 'Qwen/Qwen3-4B-Base'
                          for w in m['windows'] for k in (3,)}
        _require(seen == expected_cells, 'expected both Gate E 4B K3 cells')
    return {'status': 'VERIFIED_CALIBRATION', 'scope': scope, 'pool': str(pool),
            'cell_count': len(cells), 'relative_tolerance': 1e-4, 'cells': cells}
