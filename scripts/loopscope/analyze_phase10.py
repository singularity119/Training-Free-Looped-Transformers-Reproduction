#!/usr/bin/env python3
"""Analyze the normal Phase 10 verifier closure after fresh raw-score closure.

Reloads closure roots through verify_phase10_scores, including canonical MMLU
history reuse, and consumes its aligned output directly. Gold is a separate
identity-bound JSON object with
{identities:[...],gold:[candidate indices]}; never opened until complete closure
and the ARC preselected-lambda checks pass. Gate A runs synthetic data only.
"""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase10_analysis import analyze_panel, validate_transfer_lambdas, write_report, write_tables
from tflt.loopscope.phase10_panel import DATASETS, build_panel


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_closed_scores(closure, config_path=None):
    dataset = closure.get('dataset')
    if (closure.get('schema') != 'loopscope.phase10.score_verification.v1' or
            closure.get('status') != 'FULL_PHASE10_SCORE_PANEL_CLOSED' or
            closure.get('target_gold_loaded') is not False or dataset not in DATASETS or
            closure.get('scope') != 'FORMAL_TEST'):
        raise ValueError('complete dataset-matched gold-free formal closure is required before gold')
    panel = build_panel(dataset)
    expected_ids = {cell['cell_id'] for cell in panel['cells']}
    n = panel['sample_count_per_cell']
    if (closure.get('cell_count') != len(expected_ids) or closure.get('counts_by_cell') !=
            {cell_id: n for cell_id in expected_ids} or closure.get('sample_count') != len(expected_ids) * n):
        raise ValueError('closure does not cover every independent cell and canonical item')
    # Same loader/identity verifier as the score producer's normal closure path.
    from verify_phase10_scores import verify
    new_ids = {cell['cell_id'] for cell in panel['cells'] if cell['new_in_phase10']}
    roots = [Path(row['root']) for row in closure['roots'] if row['cell_id'] in new_ids]
    fresh, aligned = verify(roots, Path(closure['manifest']), Path(closure['pool']),
                            'FORMAL_TEST', config_path, full_panel=True,
                            include_reuse=dataset == 'mmlu')
    if fresh['status'] != 'FULL_PHASE10_SCORE_PANEL_CLOSED' or fresh['counts_by_cell'] != closure['counts_by_cell']:
        raise ValueError('fresh normal raw-score verification failed to close')
    if (aligned.get('schema') != 'loopscope.phase10.aligned_scores.v1' or
            aligned.get('dataset') != dataset or aligned.get('target_gold_loaded') is not False or
            set(aligned.get('cells', {})) != expected_ids or len(aligned.get('identities', [])) != n or
            any(len(scores) != n for scores in aligned['cells'].values())):
        raise ValueError('expected complete official verifier aligned-score artifact')
    cells = [{**cell, 'scores': aligned['cells'][cell['cell_id']]} for cell in panel['cells']]
    return cells, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--closure', type=Path, required=True, help='normal complete verifier closure artifact')
    parser.add_argument('--config', type=Path, help='accepted Phase 10 data-bound config')
    parser.add_argument('--gold', type=Path, required=True, help='separate identity-bound JSON labels; opened only after closure')
    parser.add_argument('--transfer-lambdas', type=Path, help='required ARC: frozen seven MMLU selections')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-path', type=Path)
    parser.add_argument('--resource-summary', type=Path)
    args = parser.parse_args()
    closure = read_json(args.closure)
    if closure.get('dataset') != args.dataset:
        raise ValueError('requested dataset differs from closure')
    transfer = read_json(args.transfer_lambdas) if args.transfer_lambdas else None
    if args.dataset == 'arc_challenge':
        if transfer is None:
            raise ValueError('ARC analysis requires a pre-outcome MMLU transfer manifest')
        validate_transfer_lambdas(transfer)
    cells, aligned = load_closed_scores(closure, args.config)
    gold_value = read_json(args.gold)
    if not isinstance(gold_value, dict) or gold_value.get('identities') != aligned['identities']:
        raise ValueError('gold identities must match the verified canonical order exactly')
    result = analyze_panel(args.dataset, cells, gold_value['gold'], aligned.get('subjects'),
                           aligned.get('choice_lengths'), transfer)
    result['source_closure'] = str(args.closure)
    result['fresh_raw_score_verification'] = True
    result['source_manifest'] = closure['manifest']
    result['source_pool'] = closure['pool']
    result['source_transfer_manifest'] = str(args.transfer_lambdas) if args.transfer_lambdas else None
    write_tables(args.output_dir, result)
    if args.report_path:
        write_report(args.report_path, result, read_json(args.resource_summary) if args.resource_summary else None)
    print(json.dumps({'dataset': args.dataset, 'status': result['status'], 'families': result['families'],
                      'n_per_cell': result['n_per_cell'], 'output_dir': str(args.output_dir)}, sort_keys=True))


if __name__ == '__main__':
    main()
