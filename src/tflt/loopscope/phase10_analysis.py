"""Frozen Phase 10 statistics and outcome-independent transfer selection rules.

Imports neither models nor datasets. Caller must close and join the complete
raw-score panel before loading gold; Gate A exercises this module synthetically.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .phase8_accuracy_stats import exact_mcnemar_p, holm_adjust, paired_contrast
from .phase10_panel import DATASETS, STRENGTHS, build_panel, strategy_id, write_manifest

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261002
SCAN_FAMILY = 'LAMBDA_SCAN_126'
POLICY_FAMILY = 'POLICY_COMPARISONS_45'
DIRECTION_FAMILY = POLICY_FAMILY
TRANSFER_FAMILY = 'ARC_DIRECT_TRANSFER_14'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def score_predictions(scores: Sequence[Sequence[float]], choice_lengths=None) -> list[int]:
    """Full-continuation argmax, optionally harness acc_norm character lengths.

    Ties follow the original candidate order. Character lengths refer to original
    answer text, excluding the harness scoring delimiter; never token lengths.
    """
    if choice_lengths is not None:
        _require(len(choice_lengths) == len(scores), 'normalization lengths must align with score rows')
    result = []
    for index, row in enumerate(scores):
        _require(len(row) >= 2 and all(math.isfinite(float(value)) for value in row), 'score rows require finite full-candidate scores')
        lengths = choice_lengths[index] if choice_lengths is not None else [1] * len(row)
        _require(len(lengths) == len(row) and all(type(value) is int and value > 0 for value in lengths),
                 'acc_norm requires positive answer-text character lengths for every candidate')
        result.append(max(range(len(row)), key=lambda choice: float(row[choice]) / lengths[choice]))
    return result


def _cell_stats(correctness: Sequence[int], subjects: Sequence[str] | None) -> dict[str, Any]:
    n = len(correctness)
    result = {'correct': sum(correctness), 'n': n, 'micro_accuracy': sum(correctness) / n}
    if subjects is not None:
        grouped = defaultdict(list)
        for subject, value in zip(subjects, correctness):
            grouped[subject].append(value)
        subject_accuracy = {subject: {'correct': sum(values), 'n': len(values), 'accuracy': sum(values) / len(values)}
                            for subject, values in sorted(grouped.items())}
        result.update(subject_macro_accuracy=math.fsum(row['accuracy'] for row in subject_accuracy.values()) / len(grouped),
                      subject_accuracy=subject_accuracy)
    return result


def select_transfer_lambdas(analysis_or_cells: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seven independent Online strategies: highest MMLU correct count, smaller lambda on exact ties."""
    if isinstance(analysis_or_cells, Mapping):
        _require(analysis_or_cells.get('dataset') == 'mmlu', 'transfer lambdas must be selected on MMLU')
        stats = analysis_or_cells['cells']
    else:
        stats = analysis_or_cells
    expected = build_panel('mmlu')['cells']
    expected_online = {cell['cell_id']: cell for cell in expected if cell['arm'] == 'Online'}
    online = [cell for cell in stats if cell.get('arm') == 'Online']
    _require(len(online) == 63 and {cell.get('cell_id') for cell in online} == set(expected_online),
             'lambda selection requires all 63 independent Online cells')
    grouped = defaultdict(list)
    for cell in online:
        _require(cell.get('dataset') == 'mmlu', 'transfer lambdas may only use MMLU cell statistics')
        canonical = expected_online[cell['cell_id']]
        _require(all(cell.get(key) == canonical[key] for key in ('model', 'window', 'k', 'direction_policy', 'strength')),
                 'lambda selector metadata differs from frozen independent strategy')
        _require(type(cell.get('correct')) is int and type(cell.get('n')) is int and
                 0 <= cell['correct'] <= cell['n'] and cell['n'] > 0, 'selection requires exact correct/N')
        grouped[strategy_id(cell)].append(cell)
    _require(len(grouped) == 7, 'transfer selector requires seven nonduplicate strategies')
    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        _require(len(candidates) == 9 and {cell['strength'] for cell in candidates} == set(STRENGTHS), 'each strategy requires all nine lambdas')
        _require(len({cell['n'] for cell in candidates}) == 1, 'selection denominators must match within strategy')
        best = min(candidates, key=lambda cell: (-cell['correct'], cell['strength']))
        selected.append({'strategy_id': key, 'cell_id': best['cell_id'], 'model': best['model'],
                         'window': best['window'], 'k': best['k'], 'direction_policy': best['direction_policy'],
                         'strength': best['strength'], 'mmlu_correct': best['correct'], 'mmlu_n': best['n']})
    return {'schema': 'loopscope.phase10.transfer_lambdas.v1', 'source_dataset': 'mmlu',
            'status': 'MMLU_SELECTED_ARC_OUTCOMES_UNREAD', 'arc_target_gold_loaded': False,
            'selection_rule': 'max MMLU micro accuracy; exact ties choose smaller lambda; no Loop-gain threshold',
            'strategy_count': 7, 'selected': selected}


def validate_transfer_lambdas(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(value.get('schema') == 'loopscope.phase10.transfer_lambdas.v1' and
             value.get('source_dataset') == 'mmlu' and value.get('status') == 'MMLU_SELECTED_ARC_OUTCOMES_UNREAD' and
             value.get('arc_target_gold_loaded') is False and value.get('strategy_count') == 7,
             'transfer manifest must be fixed on MMLU before ARC outcomes')
    online = {cell['cell_id']: cell for cell in build_panel('mmlu')['cells'] if cell['arm'] == 'Online'}
    selected = value.get('selected', [])
    _require(len(selected) == 7, 'transfer manifest requires seven selections')
    by_strategy = {}
    for row in selected:
        _require(row.get('cell_id') in online, 'transfer selection outside frozen independent panel')
        cell = online[row['cell_id']]
        key = strategy_id(cell)
        _require(row.get('strategy_id') == key and key not in by_strategy and
                 all(row.get(name) == cell[name] for name in ('model', 'window', 'k', 'direction_policy', 'strength')),
                 'transfer selection metadata or duplicate strategy differs')
        by_strategy[key] = row
    _require(set(by_strategy) == {strategy_id(cell) for cell in online.values()}, 'transfer manifest misses a strategy')
    return by_strategy


def analyze_panel(dataset: str, cells: Sequence[Mapping[str, Any]], gold: Sequence[int],
                  subjects: Sequence[str] | None = None, choice_lengths=None,
                  transfer_lambdas: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze aligned independent cells containing `scores`, after score closure.

    ARC also accepts per-cell `choice_text_lengths`; supplied normalization lengths
    must agree across all methods. `acc` is descriptive, all ARC inference uses
    `acc_norm`. MMLU inference uses micro accuracy and resamples within subjects.
    """
    _require(dataset in DATASETS, 'unexpected analysis dataset')
    panel = build_panel(dataset)
    expected = {cell['cell_id']: cell for cell in panel['cells']}
    _require(len(cells) == 68 and len({cell.get('cell_id') for cell in cells}) == 68 and
             {cell.get('cell_id') for cell in cells} == set(expected), 'analysis requires the complete 68 independent cells')
    _require(len(gold) > 0 and all(type(label) is int and label >= 0 for label in gold), 'gold must be aligned candidate indices')
    if dataset == 'mmlu':
        _require(subjects is not None and len(subjects) == len(gold) and all(isinstance(s, str) and s for s in subjects),
                 'MMLU bootstrap requires an aligned subject for every item')
        strata = list(subjects)
    else:
        strata = ['ARC_ITEMS'] * len(gold)
        _require(transfer_lambdas is not None, 'ARC analysis requires the pre-outcome MMLU transfer manifest')
        validate_transfer_lambdas(transfer_lambdas)
    by_id, accuracy, correctness, raw_correctness = {}, [], {}, {}
    frozen_lengths = choice_lengths
    for cell in cells:
        canonical = expected[cell['cell_id']]
        _require(all(cell.get(key) == value for key, value in canonical.items()), 'analysis cell metadata differs from panel')
        scores = cell.get('scores', [])
        _require(len(scores) == len(gold), 'every cell must cover the same gold identities')
        _require(all(label < len(row) for label, row in zip(gold, scores)), 'gold index exceeds actual candidate count')
        if dataset == 'mmlu':
            _require(all(len(row) == 4 for row in scores), 'MMLU requires original four answer-letter candidates')
        lengths = None
        if dataset == 'arc_challenge':
            own_lengths = cell.get('choice_text_lengths', choice_lengths)
            _require(own_lengths is not None, 'ARC acc_norm requires original full answer-text lengths')
            if frozen_lengths is None:
                frozen_lengths = own_lengths
            _require(own_lengths == frozen_lengths, 'ARC answer-text normalization differs across cells')
            lengths = own_lengths
        pred = score_predictions(scores, lengths)
        correct = [int(p == g) for p, g in zip(pred, gold)]
        correctness[cell['cell_id']] = correct
        raw_pred = score_predictions(scores)
        raw_correct = [int(p == g) for p, g in zip(raw_pred, gold)]
        raw_correctness[cell['cell_id']] = raw_correct
        row = {**canonical, **_cell_stats(correct, subjects if dataset == 'mmlu' else None),
               'primary_metric': 'acc' if dataset == 'mmlu' else 'acc_norm'}
        if dataset == 'arc_challenge':
            row.update(acc_correct=sum(raw_correct), acc=sum(raw_correct) / len(gold),
                       acc_norm=sum(correct) / len(gold))
        if 'resource_summary' in cell:
            row['resource_summary'] = cell['resource_summary']
        accuracy.append(row)
        by_id[cell['cell_id']] = canonical
    contrasts = []

    def add(reference: str, treatment: str, family: str) -> None:
        stats = paired_contrast(correctness[reference], correctness[treatment], strata,
                                bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        if dataset == 'arc_challenge':
            stats['bootstrap'].update(method='item_paired_bootstrap', sampling='item_multinomial_paired_difference_counts')
            stats.pop('subject_macro_delta_pp')
            stats['bootstrap'].pop('subject_count')
        cell = by_id[treatment]
        contrasts.append({'reference': reference, 'treatment': treatment, 'family': family,
                          'comparison': ('Online-' + by_id[reference]['arm'] if by_id[reference]['arm'] != 'Online'
                                         else cell['direction_policy'] + '-' + by_id[reference]['direction_policy']),
                          'model': cell['model'], 'window': cell['window'], 'k': cell['k'],
                          'direction_policy': cell['direction_policy'], 'strength': cell['strength'],
                          **stats, 'net_corrections': stats['wrong_to_right'] - stats['right_to_wrong']})

    online = [cell for cell in panel['cells'] if cell['arm'] == 'Online']
    for cell in online:
        stem = cell['cell_id'].split('-online-')[0]
        prefix = ('q4', 'q17')[cell['model_index']]
        add(stem + '-loop', cell['cell_id'], SCAN_FAMILY)
        add(prefix + '-native', cell['cell_id'], SCAN_FAMILY)
        if cell['k'] == 3 and cell['direction_policy'] == 'lag1':
            add(cell['cell_id'].replace('-online-lag1-', '-online-fixed-t0-'), cell['cell_id'], POLICY_FAMILY)
        if cell['direction_policy'] == 'current_t':
            add(cell['cell_id'].replace('-online-current-t-', '-online-fixed-t0-'), cell['cell_id'], POLICY_FAMILY)
            if cell['k'] == 3:
                add(cell['cell_id'].replace('-online-current-t-', '-online-lag1-'), cell['cell_id'], POLICY_FAMILY)
    if dataset == 'arc_challenge':
        for selection in transfer_lambdas['selected']:
            cell = by_id[selection['cell_id']]
            add(cell['cell_id'].split('-online-')[0] + '-loop', cell['cell_id'], TRANSFER_FAMILY)
            add(('q4', 'q17')[cell['model_index']] + '-native', cell['cell_id'], TRANSFER_FAMILY)
    family_sizes = {SCAN_FAMILY: 126, POLICY_FAMILY: 45}
    if dataset == 'arc_challenge':
        family_sizes[TRANSFER_FAMILY] = 14
    families = {}
    for family, size in family_sizes.items():
        rows = [row for row in contrasts if row['family'] == family]
        _require(len(rows) == size, f'{family} must contain exactly {size} comparisons')
        adjusted = holm_adjust([row['mcnemar_exact_p'] for row in rows])
        for row, pvalue in zip(rows, adjusted):
            row.update(holm_adjusted_p=pvalue, holm_reject_alpha_0_05=pvalue <= 0.05,
                       holm_family_size=size, supports_positive_gain=pvalue <= 0.05 and row['delta_pp'] > 0)
        families[family] = {'size': size, 'alpha': 0.05, 'test': 'exact two-sided McNemar', 'adjustment': 'one Holm family'}
    reported = {row['cell_id']: row for row in accuracy}
    display = [{**reported[row['result_cell_id']], 'cell_id': row['cell_id'],
                'direction_policy': row['direction_policy'], 'result_cell_id': row['result_cell_id'],
                'alias_of': row['alias_of'], 'independent_result': row['independent_result']}
               for row in panel['display_rows']]
    result = {'schema': 'loopscope.phase10.analysis.v1', 'dataset': dataset,
              'status': 'ANALYZED_COMPLETE_PANEL', 'target_gold_loaded': True,
              'primary_metric': panel['dataset_recipe']['primary_metric'], 'independent_cell_count': 68,
              'display_row_count': 86, 'alias_count': 18, 'n_per_cell': len(gold),
              'cells': accuracy, 'display_rows': display, 'contrasts': contrasts, 'families': families,
              'bootstrap': {'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED, 'ci_level': 0.95,
                            'method': 'within_subject_paired_bootstrap' if dataset == 'mmlu' else 'item_paired_bootstrap',
                            'statistic': 'question_weighted_micro_accuracy_difference', 'ci_kind': 'nominal_percentile_linear'},
              'interpretation': {'mmlu': 'development-domain exploration; best grid value is selected',
                                 'arc_transfer': 'fixed MMLU lambdas compared separately from ARC grid exploration',
                                 'native': 'Online>Native supports the total scheme',
                                 'loop': 'Online>Loop supports added intervention benefit',
                                 'direction': 'policy differences require the direct 45-comparison family',
                                 'current_t': 'current_t changes intervention onset and direction timing jointly; no isolated t0 or timing attribution',
                                 'mechanism': 'No matched-norm control; no direction-specific mechanism conclusion',
                                 'ci': 'nominal intervals do not replace multiplicity-corrected decisions',
                                 'secondary_arc_acc': 'descriptive only'}}
    result['transfer_lambdas'] = select_transfer_lambdas(result) if dataset == 'mmlu' else dict(transfer_lambdas)
    return result


def write_tables(output_dir: Path, result: Mapping[str, Any]) -> None:
    """Write complete independent and alias tables, and all declared contrasts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(output_dir / 'analysis.json', result)
    write_manifest(output_dir / 'transfer_lambdas.json', result['transfer_lambdas'])
    for filename, rows in [('independent_cells.csv', result['cells']), ('display_cells.csv', result['display_rows']),
                           ('contrasts.csv', result['contrasts'])]:
        flat = [{**row, **row.get('bootstrap', {})} for row in rows]
        fields = list(dict.fromkeys(key for row in flat for key in row if key not in ('bootstrap', 'subject_accuracy', 'resource_summary')))
        with (output_dir / filename).open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(flat)


def write_report(path: Path, result: Mapping[str, Any], resource_summary: Mapping[str, Any] | None = None) -> None:
    """Write a concise complete Chinese report; human artifact location chosen by caller."""
    lines = ['# LoopScope 第十阶段：' + result['dataset'], '',
             '完整面板包含68个独立配置和86个展示行。18行K2 lag1为fixed_t0共享结果，不能视为独立重复实验。', '',
             '主指标：' + result['primary_metric'] + '。MMLU为开发域强度探索；ARC的14项直接迁移比较与126项网格探索各自作一个Holm family，策略比较为独立45项family（K2共18、K3共27）。',
             'exact双侧McNemar；配对bootstrap10000次，seed20261002，nominal 95% CI。MMLU在subject内重采样并按题加权，ARC按题重采样。', '',
             '| 配置 | 共享结果 | 正确数/N | 主准确率 |', '|---|---|---:|---:|']
    for row in result['display_rows']:
        lines.append(f"| {row['cell_id']} | {row['alias_of'] or ''} | {row['correct']}/{row['n']} | {100 * row['micro_accuracy']:.4f}% |")
    lines.extend(['', 'Online>Native表示总方案收益；Online>Loop才表示额外干预收益。current_t同时改变干预起点和方向时序，不能单独归因于t0或当轮方向。无Matched-norm对照，不能作方向特异性机制结论；不显著不表示等效。', '',
                  '所有负结果和逐项比较均保留在完整CSV；ARC当地最优lambda只作探索，不替代MMLU预选迁移lambda。', ''])
    if resource_summary is not None:
        import json
        lines.extend(['资源账本：', json.dumps(resource_summary, ensure_ascii=False, sort_keys=True), ''])
    else:
        lines.append('资源开销尚未提供；正式交付须补入验证过的运行成本。')
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
