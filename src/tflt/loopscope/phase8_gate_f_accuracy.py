"""Frozen Gate F acquisition panel and exploratory three-arm paired analysis."""
from pathlib import Path
from .phase8_pool import DATASET_REPO, DATASET_REVISION
from .phase8_accuracy_stats import (cell_accuracy, score_predictions, paired_contrast,
                                     holm_adjust, BOOTSTRAP_REPLICATES)

SCHEMA = 'loopscope.phase8.gate_f.accuracy_manifest.v1'
MODELS = (
    ('Qwen/Qwen3-4B-Base', '906bfd4b4dc7f14ee4320094d8b41684abff8539',
     'bfloat16', 'first', ((12, 15), (13, 16)), 'q4'),
    ('Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1',
     'float16', 'last', ((12, 15), (6, 9)), 'q17'),
)


def logical_panel():
    """All 36 scientific cells, with explicit fit time and unique stable IDs."""
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


def panel(t0_basis_root, t1_basis_root):
    """Twelve shared-t0 arms followed by four new q17 K3 comparators."""
    logical = logical_panel()
    cells = [c for c in logical if c['fit_t'] == 0]
    cells += [c for c in logical if c['model_index'] == 1 and c['k'] == 3
              and c['fit_t'] != 0]
    for cell in cells:
        wi = MODELS[cell['model_index']][4].index(tuple(cell['window']))
        if cell['fit_t'] == 0:
            cell['basis_path'] = str(Path(t0_basis_root) / f'cell-{cell["model_index"]*2+wi}' / 'basis.json')
        elif cell['fit_t'] == 1:
            cell['basis_path'] = str(Path(t1_basis_root) / f'cell-{wi}' / 'basis.json')
        else:
            cell['basis_path'] = None
    return cells


def validate_manifest(manifest, pool, pool_path):
    if (manifest.get('schema') != SCHEMA or manifest.get('scope') not in
            ('PREFLIGHT_ONLY', 'FORMAL_TEST') or
            manifest.get('dataset') != pool['dataset'] or manifest.get('pool') != str(pool_path)):
        raise ValueError('Gate F manifest schema/scope/dataset/pool mismatch')
    if manifest['cells'] != panel(manifest['t0_basis_root'], manifest['t1_basis_root']):
        raise ValueError('Gate F requires frozen 16-cell panel and four shared t0/two t1 bases')
    return {cell['cell_id']: cell for cell in manifest['cells']}


def checked_basis(cell, scope):
    if cell.get('fit_t') == 0:
        from .phase8_gate_f_basis import load_t0_basis
        basis_scope = 'FORMAL_CALIBRATION' if scope == 'FORMAL_TEST' else 'PREFLIGHT_ONLY'
        return load_t0_basis(cell['basis_path'], cell, basis_scope)
    from .phase8_runtime import load_basis
    basis = load_basis(cell['basis_path'])
    expected = dict(scope='FORMAL_CALIBRATION' if scope == 'FORMAL_TEST' else 'PREFLIGHT_ONLY',
        model=cell['model'], revision=cell['revision'], window=cell['window'], k=cell['k'],
        dtype=cell['dtype'], cache_strategy=cell['cache_strategy'], alpha=1.0,
        dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'))
    if (cell.get('fit_t') != 1 or cell['model_index'] != 1 or cell['k'] != 3 or
            any(basis['metadata'].get(key) != value for key, value in expected.items()) or
            basis['metadata'].get('fit_t', 1) != 1 or basis['rank'] != 1 or basis['lambda'] != .5 or
            (scope == 'FORMAL_TEST' and basis['row_count'] != 512)):
        raise ValueError('Gate F t1 basis differs from K3 calibration/scope recipe')
    return basis


def scientific_key(cell):
    fit_t = cell.get('fit_t', 1 if cell['arm'] == 'Spectral' else None)
    return (cell['model'], tuple(cell['window']), cell['k'], cell['arm'], fit_t)


def analyze_panel(cells, gold, subjects, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    """Call only after the gold-free 16-new/20-reused identity closure."""
    expected = {scientific_key(c): c for c in logical_panel()}
    keys = [scientific_key(c) for c in cells]
    if len(cells) != 36 or len(set(keys)) != 36 or set(keys) != set(expected):
        raise ValueError('Gate F requires exactly 36 unique logical cells')
    if len({c['cell_id'] for c in cells}) != 36:
        raise ValueError('Gate F requires unique cell IDs')
    by_key, accuracy, correctness = {}, [], {}
    for cell, key in zip(cells, keys):
        frozen = expected[key]
        if any(cell.get(field) != frozen[field] for field in ('revision', 'dtype', 'cache_strategy')):
            raise ValueError('Gate F reused/new cell recipe mismatch')
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
                        ('Loop', 't1', 'T1_MINUS_LOOP_DESCRIPTIVE')):
                    contrasts.append(dict(model=model, window=list(window), k=k,
                        reference=ids[ref], treatment=ids[treatment], family=family,
                        comparison=f'{treatment}-{ref}',
                        **paired_contrast(correctness[ids[ref]], correctness[ids[treatment]],
                                          subjects, bootstrap_replicates),
                        holm_adjusted_p=None, holm_reject_alpha_0_05=None,
                        holm_family_size=None))
    for family in ('T0_MINUS_T1_EXPLORATORY', 'T0_MINUS_LOOP_EXPLORATORY'):
        selected = [r for r in contrasts if r['family'] == family]
        for row, pvalue in zip(selected, holm_adjust([r['mcnemar_exact_p'] for r in selected])):
            row.update(holm_adjusted_p=pvalue, holm_reject_alpha_0_05=pvalue <= .05,
                       holm_family_size=12)
    return dict(cells=accuracy, contrasts=contrasts, interpretation=dict(
        domain='post_outcome_exploratory_supplement', prior_D_outcomes_known=True,
        families='t0-t1 and t0-Loop are separate twelve-contrast exploratory Holm families',
        descriptive='t1-Loop does not rewrite original D/E families',
        ci='nominal 95% intervals; not simultaneous or multiplicity-adjusted intervals',
        limitation='No universal cross-model or confirmatory best-configuration claim',
        basis='four t0 bases shared across K2/3/4; t1 remains K-specific'))
