"""Gate F explicit t0 directions fitted once from unchanged K2 calibration.

No torch import is needed for metadata checks or JSON basis loading. Historical
basis and residual artifacts remain untouched; save_basis writes new paths only.
"""
import json
import math
from pathlib import Path

from .phase8_pool import DATASET_REPO, DATASET_REVISION
from .phase8_runtime import fit_basis, identity_key

SCHEMA = 'loopscope-phase8-gate-f-t0-basis-v1'
ALGORITHM = 'cpu_float64_reduced_svd_uncentered_unnormalized'
CELL_FIELDS = ('model', 'revision', 'window', 'dtype', 'cache_strategy')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source_metadata(metadata, scope='FORMAL_CALIBRATION'):
    """Validate the frozen, unmodified K2 residual source recipe."""
    _require(scope in ('FORMAL_CALIBRATION', 'PREFLIGHT_ONLY'), 'invalid scope')
    config = json.loads((Path(__file__).resolve().parents[3] /
                         'configs/loopscope/phase8_model_windows.json').read_text())
    spec = next((m for m in config['models'] if m['model'] == metadata.get('model')), None)
    _require(spec is not None, 'unknown calibration model')
    _require(metadata.get('scope') == scope and metadata.get('k') == 2,
             't0 source must be requested scope and K2')
    _require(metadata.get('revision') == spec['observed_revision']
             and metadata.get('dtype') == spec['historical_runtime_dtype']
             and metadata.get('cache_strategy') == spec['cache_strategy']
             and metadata.get('window') in [w['layers'] for w in spec['windows']]
             and metadata.get('alpha') == 1., 'source recipe mismatch')
    _require(metadata.get('dataset') == dict(repo=DATASET_REPO, revision=DATASET_REVISION,
                                           split='validation'), 'source dataset mismatch')
    _require(all(metadata.get(key) for key in ('source_commit', 'run_root', 'pool')),
             'missing readable source provenance')


def validate_source_rows(saved, expected_metadata, expected_identities, scope='FORMAL_CALIBRATION'):
    """Torch-free identity and pre-answer-position closure before fitting."""
    validate_source_metadata(expected_metadata, scope)
    _require(saved.get('metadata') == expected_metadata, 'residual metadata mismatch')
    identities = list(expected_identities)
    _require(bool(identities) and len(set(map(identity_key, identities))) == len(identities),
             'source identities must be nonempty and unique')
    _require(scope != 'FORMAL_CALIBRATION' or len(identities) == 512,
             'formal t0 source must contain 512 identities')
    _require(saved.get('identities') == identities, 'residual identity order mismatch')
    _require(set(saved['positions']) == set(saved['matrices']) == {'0', '1'},
             'source requires both unchanged K2 residual timesteps')
    positions = saved['positions']['0']
    _require(len(positions) == len(identities)
             and all(type(p) is int and p >= 0 for p in positions), 'invalid pre-answer positions')
    _require(saved['positions']['1'] == positions, 'pre-answer position changed across timesteps')
    return identities


def verify_saved_t0(saved, expected_metadata, expected_identities, scope='FORMAL_CALIBRATION'):
    """Validate readable source closure and finite FP32 saved native residuals."""
    identities = validate_source_rows(saved, expected_metadata, expected_identities, scope)
    import torch
    matrix = saved['matrices']['0']
    _require(matrix.ndim == 2 and matrix.shape[0] == len(identities) and matrix.shape[1] > 0,
             'invalid t0 matrix shape')
    for value in saved['matrices'].values():
        _require(value.dtype == torch.float32 and value.shape == matrix.shape
                 and bool(torch.isfinite(value).all()), 'invalid residual dtype/shape/finite')
    return matrix


def fit_t0_basis(saved, expected_metadata, expected_identities, scope='FORMAL_CALIBRATION'):
    """Fit one t0 matrix using the original t1 fitter's identical SVD operations."""
    matrix = verify_saved_t0(saved, expected_metadata, expected_identities, scope)

    class MatrixSource:
        k = 2
        identities = list(expected_identities)
        keys = list(map(identity_key, identities))

        def matrix(self, t=1):
            return matrix

    basis = fit_basis(MatrixSource(), expected_metadata)
    basis.update(schema=SCHEMA, fit_t=0, source_k=2, applies_to_k=[2, 3, 4])
    del basis['k']
    basis['source_positions'] = list(saved['positions']['0'])
    return basis


def load_t0_basis(path, cell, scope='FORMAL_CALIBRATION'):
    """Load one shared direction for a model/window application cell at K2/3/4."""
    basis = json.loads(Path(path).read_text(encoding='utf-8'))
    _require(basis.get('schema') == SCHEMA and basis.get('fit_t') == 0
             and basis.get('source_k') == 2 and basis.get('applies_to_k') == [2, 3, 4]
             and 'k' not in basis, 'invalid shared t0 basis schema')
    _require(basis.get('rank') == 1 and basis.get('lambda') == .5
             and basis.get('algorithm') == ALGORITHM, 't0 fit recipe mismatch')
    metadata = basis['metadata']
    validate_source_metadata(metadata, scope)
    _require(cell.get('k') in (2, 3, 4) and cell.get('fit_t', 0) == 0,
             'invalid t0 application K or fit_t')
    _require(all(cell.get(key) == metadata[key] for key in CELL_FIELDS),
             'basis differs from application model/window recipe')
    identities = basis['source_identities']
    _require(basis.get('row_count') == len(identities) and bool(identities)
             and len(set(map(identity_key, identities))) == len(identities)
             and (scope != 'FORMAL_CALIBRATION' or len(identities) == 512),
             'invalid basis source identities')
    positions = basis['source_positions']
    _require(len(positions) == len(identities)
             and all(type(p) is int and p >= 0 for p in positions), 'invalid basis positions')
    direction = basis['direction']
    _require(bool(direction) and all(math.isfinite(v) for v in direction)
             and math.isclose(math.hypot(*direction), 1., rel_tol=1e-5, abs_tol=1e-6),
             'basis direction must be finite and unit length')
    return basis
