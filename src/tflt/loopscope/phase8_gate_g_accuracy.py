"""Frozen Gate G acquisition panel and exploratory three-arm paired analysis."""
from pathlib import Path
from .phase8_pool import DATASET_REPO, DATASET_REVISION
from .phase8_accuracy_stats import (cell_accuracy, score_predictions, paired_contrast,
                                     holm_adjust, BOOTSTRAP_REPLICATES)

SCHEMA = 'loopscope.phase8.gate_g.accuracy_manifest.v1'
FAMILIES = ('T1_MINUS_LOOP_EXPLORATORY', 'T0_MINUS_LOOP_EXPLORATORY', 'T0_MINUS_T1_EXPLORATORY')
MODELS = (
    ('Qwen/Qwen3-4B-Base', '906bfd4b4dc7f14ee4320094d8b41684abff8539',
     'bfloat16', 'first', ((15, 18),), 'q4'),
)

def logical_panel():
    """All 9 scientific cells, with explicit fit time and unique stable IDs."""
    cells = []
    for mi, (model, revision, dtype, cache, windows, prefix) in enumerate(MODELS):
        for window in windows:
            for k in (2, 3, 4):
                for arm, fit_t in (('Loop', None), ('Spectral', 1), ('Spectral', 0)):
                    suffix = arm.lower() + ('-t0' if fit_t == 0 else '')
                    cells.append(dict(model_index=mi, model=model, revision=revision,
                        dtype=dtype, cache_strategy=cache, window=list(window), k=k,
                        arm=arm, fit_t=fit_t,
                        cell_id=f'{prefix}-w{window[0]}-{window[1]}-k{k}-{suffix}'))
    return cells


def panel(basis_root):
    """Nine new cells using three K-specific t1 and one shared t0 basis."""
    cells = logical_panel()
    for cell in cells:
        relative = 't0' if cell['fit_t'] == 0 else f'k{cell["k"]}'
        cell['basis_path'] = (str(Path(basis_root) / relative / 'basis.json')
                              if cell['arm'] == 'Spectral' else None)
    return cells


def validate_manifest(manifest, pool, pool_path):
    if (manifest.get('schema') != SCHEMA or manifest.get('scope') not in
            ('PREFLIGHT_ONLY', 'FORMAL_TEST') or
            manifest.get('dataset') != pool['dataset'] or manifest.get('pool') != str(pool_path)):
        raise ValueError('Gate G manifest schema/scope/dataset/pool mismatch')
    if manifest['cells'] != panel(manifest['basis_root']):
        raise ValueError('Gate G requires frozen 9-cell panel and four bases')
    return {cell['cell_id']: cell for cell in manifest['cells']}


def checked_basis(cell, scope):
    basis_scope = 'FORMAL_CALIBRATION' if scope == 'FORMAL_TEST' else 'PREFLIGHT_ONLY'
    if cell.get('fit_t') == 0:
        from .phase8_gate_g_calibration import load_t0_basis
        return load_t0_basis(cell['basis_path'], cell, basis_scope)
    from .phase8_runtime import load_basis
    basis = load_basis(cell['basis_path'])
    expected = dict(scope=basis_scope, model=cell['model'], revision=cell['revision'],
        window=cell['window'], k=cell['k'], dtype=cell['dtype'],
        cache_strategy=cell['cache_strategy'], alpha=1.0,
        dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'))
    if (cell.get('fit_t') != 1 or basis.get('fit_t') != 1 or
            any(basis['metadata'].get(key) != value for key, value in expected.items()) or
            basis['rank'] != 1 or basis['lambda'] != .5 or
            (scope == 'FORMAL_TEST' and basis['row_count'] != 512)):
        raise ValueError('Gate G t1 basis differs from K-specific calibration/scope recipe')
    return basis


def scientific_key(cell):
    fit_t = cell.get('fit_t', 1 if cell['arm'] == 'Spectral' else None)
    return (cell['model'], tuple(cell['window']), cell['k'], cell['arm'], fit_t)


def analyze_panel(cells, gold, subjects, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    """Call only after the gold-free nine-cell identity closure."""
    expected = {scientific_key(c): c for c in logical_panel()}
    keys = [scientific_key(c) for c in cells]
    if len(cells) != 9 or len(set(keys)) != 9 or set(keys) != set(expected):
        raise ValueError('Gate G requires exactly 9 unique logical cells')
    if len({c['cell_id'] for c in cells}) != 9:
        raise ValueError('Gate G requires unique cell IDs')
    by_key, accuracy, correctness = {}, [], {}
    for cell, key in zip(cells, keys):
        frozen = expected[key]
        if any(cell.get(field) != frozen[field] for field in ('revision', 'dtype', 'cache_strategy')):
            raise ValueError('Gate G new cell recipe mismatch')
        cid = cell['cell_id']
        by_key[key] = cid
        accuracy.append({field: cell[field] for field in ('cell_id', 'model', 'window', 'k', 'arm')} |
                        dict(fit_t=key[-1]) | cell_accuracy(cell['scores'], gold, subjects))
        correctness[cid] = [int(p == g) for p, g in zip(score_predictions(cell['scores']), gold)]
    contrasts = []
    for model, _, _, _, windows, _ in MODELS:
        for window in windows:
            for k in (2, 3, 4):
                ids = {name: by_key[(model, window, k, arm, fit_t)] for name, arm, fit_t in
                       (('Loop', 'Loop', None), ('t1', 'Spectral', 1), ('t0', 'Spectral', 0))}
                for ref, treatment, family in (
                        ('t1', 't0', 'T0_MINUS_T1_EXPLORATORY'),
                        ('Loop', 't0', 'T0_MINUS_LOOP_EXPLORATORY'),
                        ('Loop', 't1', 'T1_MINUS_LOOP_EXPLORATORY')):
                    contrasts.append(dict(model=model, window=list(window), k=k,
                        reference=ids[ref], treatment=ids[treatment], family=family,
                        comparison=f'{treatment}-{ref}',
                        **paired_contrast(correctness[ids[ref]], correctness[ids[treatment]],
                                          subjects, bootstrap_replicates),
                        holm_adjusted_p=None, holm_reject_alpha_0_05=None,
                        holm_family_size=None))
    for family in FAMILIES:
        selected = [r for r in contrasts if r['family'] == family]
        for row, pvalue in zip(selected, holm_adjust([r['mcnemar_exact_p'] for r in selected])):
            row.update(holm_adjusted_p=pvalue, holm_reject_alpha_0_05=pvalue <= .05,
                       holm_family_size=3)
    return dict(cells=accuracy, contrasts=contrasts, native_context=dict(
        source='Gate D q4-native', correct=10262, n=14042,
        micro_accuracy=10262/14042, use='descriptive_only_no_new_significance_test'),
        interpretation=dict(domain='post_outcome_exploratory_window_supplement',
        prior_D_E_F_outcomes_known=True,
        families='Three separate exploratory Holm families, three K comparisons each; not joint nine-test control',
        ci='nominal 95% intervals; not simultaneous or multiplicity-adjusted intervals',
        limitation='No confirmatory, universal K, or best-K claim',
        basis='one t0 basis shared across K2/3/4; three K-specific t1 bases'))
