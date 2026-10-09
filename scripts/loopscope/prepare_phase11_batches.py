#!/usr/bin/env python3
"""Prepare Gate D gold-free ARC/GPQA worklists, retaining Gate C identities."""
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase11_accuracy import load_pool
from tflt.loopscope.phase11_panel import require, validate_score_manifest, write_json_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('arc_challenge', 'gpqa_main'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--canary-closure', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--runs-prefix', required=True)
    parser.add_argument('--trial-worker-count', type=int, default=24)
    args = parser.parse_args()
    panel = json.loads(args.manifest.read_text())
    cells = validate_score_manifest(panel, pool_path=args.pool, scope='FORMAL_TEST')
    pool = load_pool(args.pool, args.dataset, 'FORMAL_TEST')
    require(panel['dataset'] == args.dataset, 'worklist task differs')
    canary = json.loads(args.canary_closure.read_text())
    require(canary['status'] == 'CANARY_CLOSED_GOLD_UNREAD' and
            canary['target_gold_loaded'] is False, 'canary reuse lacks gold-free closure')
    retained = {cell: {} for cell in cells}
    for fact in canary['completed']:
        if fact['dataset'] != args.dataset:
            continue
        cell, index = fact['cell_id'], fact['canonical_index']
        require(cell in retained and index not in retained[cell] and
                pool['test_identities'][index] == fact['identity'], 'canary identity differs')
        retained[cell][index] = fact['run_root']
    require(all(len(values) == 4 for values in retained.values()), 'expected four retained identities per cell')
    rows, n = pool['rows'], len(pool['rows'])
    remaining = {cell: [i for i in range(n) if i not in retained[cell]] for cell in cells}
    selected = next(cell for cell in panel['cells'] if cell['arm'] == 'Online' and
                    cell['k'] == 3 and cell['direction_policy'] == 'current_t' and cell['strength'] == .9)
    selected_id = selected['cell_id']

    def length(index):
        row = rows[index]
        if args.dataset == 'arc_challenge':
            return max(item['input_length'] for item in row['tokenization'][pool['model']]['candidates'])
        return row['prompt_token_length']

    # Twelve equal-size lanes interleave descending input lengths. Distinct trial
    # records therefore cover nearly matched lengths without repeating any result.
    count = 12 * args.trial_worker_count
    require(0 < count <= len(remaining[selected_id]), 'trial records exceed remaining frozen population')
    longest = sorted(remaining[selected_id], key=lambda i: (-length(i), i))[:count]
    lanes = [sorted(longest[lane::12]) for lane in range(12)]
    allocation = {2: [0, 6], 4: [1, 4, 7, 10], 6: [2, 3, 5, 8, 9, 11]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for packing, lane_ids in allocation.items():
        jobs = []
        for position, lane in enumerate(lane_ids):
            indices = lanes[lane]
            jobs.append({'dataset': args.dataset, 'cell_id': selected_id,
                'manifest': str(args.manifest), 'pool': str(args.pool),
                'population_count': n, 'indices': indices,
                'run_root': f'{args.runs_prefix}-{args.dataset}-trial-p{packing}-w{position}',
                'input_length_sum': sum(length(i) for i in indices),
                'max_input_length': max(length(i) for i in indices)})
        write_json_once(args.output_dir / f'trial-p{packing}.json', {'jobs': jobs, 'packing': packing,
            'selection': 'longest unmatched inputs; twelve length-interleaved lanes; same K3 current_t lambda0.9',
            'target_gold_loaded': False})
    trial_set = set(longest)
    remaining[selected_id] = [i for i in remaining[selected_id] if i not in trial_set]
    write_json_once(args.output_dir / 'plan.json', {'dataset': args.dataset, 'population_count': n,
        'retained': {cell: [{'canonical_index': i, 'run_root': root} for i, root in sorted(facts.items())]
                     for cell, facts in retained.items()},
        'trial_count': count, 'trial_indices': sorted(longest), 'trial_cell_id': selected_id,
        'remaining_after_trials': remaining, 'target_gold_loaded': False})
    print(json.dumps({'dataset': args.dataset, 'retained_count': sum(map(len, retained.values())),
        'trial_count': count, 'remaining_after_trials_count': sum(map(len, remaining.values())),
        'target_gold_loaded': False}))


if __name__ == '__main__':
    main()
