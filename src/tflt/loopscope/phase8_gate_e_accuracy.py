"""Separate, post-outcome exploratory four-cell K3 supplement."""
from pathlib import Path
from .phase8_pool import DATASET_REPO, DATASET_REVISION
from .phase8_accuracy_stats import (cell_accuracy, score_predictions, paired_contrast,
                                     holm_adjust, BOOTSTRAP_REPLICATES)

MODEL = 'Qwen/Qwen3-4B-Base'
REVISION = '906bfd4b4dc7f14ee4320094d8b41684abff8539'
SCHEMA = 'loopscope.phase8.gate_e.accuracy_manifest.v1'


def panel(basis_root):
    cells = []
    for index, window in enumerate(([12, 15], [13, 16])):
        for arm in ('Loop', 'Spectral'):
            cells.append(dict(model_index=0, model=MODEL, revision=REVISION,
                              dtype='bfloat16', cache_strategy='first',
                              cell_id=f'q4-w{window[0]}-{window[1]}-k3-{arm.lower()}',
                              window=window, k=3, arm=arm,
                              basis_path=str(Path(basis_root)/f'cell-{index}/basis.json')
                              if arm == 'Spectral' else None))
    return cells


def validate_manifest(manifest, pool, pool_path):
    if (manifest.get('schema') != SCHEMA or manifest.get('scope') not in
            ('PREFLIGHT_ONLY', 'FORMAL_TEST') or
            manifest['dataset'] != pool['dataset'] or manifest['pool'] != str(pool_path)):
        raise ValueError('Gate E manifest schema/scope/dataset/pool mismatch')
    if manifest['cells'] != panel(manifest['basis_root']):
        raise ValueError('Gate E requires frozen four-cell K3 panel and two basis mappings')
    return {cell['cell_id']: cell for cell in manifest['cells']}


def checked_basis(cell, scope):
    from .phase8_runtime import load_basis
    basis = load_basis(cell['basis_path'])
    expected = dict(scope='FORMAL_CALIBRATION' if scope == 'FORMAL_TEST' else 'PREFLIGHT_ONLY',
                    model=MODEL, revision=REVISION, window=cell['window'], k=3,
                    dtype='bfloat16', cache_strategy='first', alpha=1.0,
                    dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'))
    if (any(basis['metadata'].get(key) != value for key, value in expected.items()) or
            basis['rank'] != 1 or basis['lambda'] != .5 or
            (scope == 'FORMAL_TEST' and basis['row_count'] != 512)):
        raise ValueError('Gate E basis differs from K3 calibration/scope recipe')
    return basis


def analyze_panel(cells, gold, subjects, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    expected = {(tuple(c['window']), c['arm']) for c in panel('/unused')}
    actual = [(tuple(c['window']), c['arm']) for c in cells]
    if (len(cells) != 4 or set(actual) != expected or
            any(c['model'] != MODEL or c['k'] != 3 for c in cells)):
        raise ValueError('K3_EXPLORATORY requires exactly the four Gate E cells')
    accuracy, correctness, by_key = [], {}, {}
    for cell in cells:
        cid = cell['cell_id']
        by_key[(tuple(cell['window']), cell['arm'])] = cid
        accuracy.append({key: cell[key] for key in ('cell_id','model','window','k','arm')} |
                        cell_accuracy(cell['scores'], gold, subjects))
        correctness[cid] = [int(p == g) for p,g in zip(score_predictions(cell['scores']),gold)]
    contrasts = []
    for window in ((12,15), (13,16)):
        reference, treatment = (by_key[(window, arm)] for arm in ('Loop','Spectral'))
        contrasts.append(dict(reference=reference, treatment=treatment,
                              family='K3_EXPLORATORY', comparison='Spectral-Loop',
                              **paired_contrast(correctness[reference],correctness[treatment],
                                                subjects,bootstrap_replicates)))
    for row, adjusted in zip(contrasts, holm_adjust([r['mcnemar_exact_p'] for r in contrasts])):
        row.update(holm_adjusted_p=adjusted,holm_reject_alpha_0_05=adjusted <= .05,
                   holm_family_size=2)
    return dict(cells=accuracy, contrasts=contrasts, interpretation=dict(
        domain='post_outcome_exploratory_supplement', prior_K2_K4_outcomes_known=True,
        family='K3_EXPLORATORY; two contrasts separate from original families',
        ci='nominal; not a multiplicity-adjusted significance decision',
        limitation='No matched-norm control; does not isolate direction specificity',
        k='K3 fixed horizon 1 with two newly fitted K-specific bases'))
