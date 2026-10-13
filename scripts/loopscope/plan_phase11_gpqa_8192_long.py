#!/usr/bin/env python3
"""Freeze exact unsubmitted D-4 pairs from submitted worklists and receipts."""
import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def run(args):
    ledger = read(args.ledger)
    frozen = read(args.membership)
    formal = [j for j in ledger['jobs'] if j['class'] == 'formal']
    assert len({j['job_id'] for j in formal}) == len(formal)
    submitted, roots = set(), defaultdict(list)
    submitted_children = []
    for parent in formal:
        if 'number' in parent:
            receipt = read(Path(parent['worklist']).parent / 'submission-receipts' / f"batch-{parent['number']:03d}.json")
            assert receipt['job_id'] == parent['job_id'] and receipt['worklist'] == parent['worklist']
        for child in read(parent['worklist'])['jobs']:
            pairs = {(child['cell_id'], i) for i in child['indices']}
            assert len(pairs) == len(child['indices']) and not pairs.intersection(submitted)
            submitted.update(pairs)
            roots[child['cell_id']].append(child['run_root'])
            submitted_children.append(child)
    cells = {r['cell']['cell_id']: r['cell'] for r in frozen['cells']}
    target = {(cid, i) for cid in cells for i in range(448)}
    assert len(target) == 8064 and submitted.issubset(target)
    assert len(submitted) == ledger['formal_submitted_records']
    # Independently confirm no additional remote receipt was omitted from the ledger.
    plan = read(ledger['formal_plan'])
    remote_receipts = list((Path(ledger['formal_plan']).parent / 'submission-receipts').glob('batch-*.json'))
    receipt_ids = {read(p)['job_id'] for p in remote_receipts}
    ledger_plan_ids = {j['job_id'] for j in formal if Path(j['worklist']).parent == Path(ledger['formal_plan']).parent}
    assert receipt_ids == ledger_plan_ids
    remaining = target - submitted
    assert remaining
    by_cell = {cid: sorted(i for c, i in remaining if c == cid) for cid in cells}
    by_cell = {cid: indices for cid, indices in by_cell.items() if indices}
    slots = args.task_count * 5
    assert len(by_cell) <= slots <= len(remaining)
    weight = {cid: (1 if cell['arm'] == 'Native' else cell['k']) for cid, cell in cells.items()}
    divisions = {cid: 1 for cid in by_cell}
    while sum(divisions.values()) < slots:
        cid = max((c for c in by_cell if divisions[c] < len(by_cell[c])),
                  key=lambda c: len(by_cell[c]) * weight[c] / divisions[c])
        divisions[cid] += 1
    chunks = []
    for cid, indices in by_cell.items():
        for part in range(divisions[cid]):
            chunk = indices[part::divisions[cid]]
            chunks.append((len(chunk) * weight[cid], cid, chunk))
    assignments = [[] for _ in range(args.task_count)]
    loads = [0] * args.task_count
    for cost, cid, indices in sorted(chunks, reverse=True):
        task = min((n for n in range(args.task_count) if len(assignments[n]) < 5), key=lambda n: loads[n])
        assignments[task].append((cid, indices))
        loads[task] += cost
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=False)
    tasks, new_children = [], []
    for number, groups in enumerate(assignments):
        jobs = []
        for child_number, (cid, indices) in enumerate(groups):
            root = str(Path(args.runs) / (stage.name + f'-t{number:02d}-w{child_number:02d}'))
            assert not Path(root).exists()
            jobs.append({'cell_id': cid, 'indices': indices, 'run_root': root,
                'membership': args.membership, 'pool': frozen['pool'], 'synthetic': False})
            roots[cid].append(root)
        worklist = stage / f'task-{number:02d}.json'
        write(worklist, {'jobs': jobs, 'target_gold_loaded': False})
        new_children.extend(jobs)
        tasks.append({'number': number, 'worklist': str(worklist),
            'batch_root': str(Path(args.runs) / (stage.name + f'-t{number:02d}-batch')),
            'packing': 5, 'synthetic': False, 'partition': 'emergency_gpu',
            'records': sum(len(j['indices']) for j in jobs), 'relative_work_weight': loads[number]})
    assert {(j['cell_id'], i) for j in new_children for i in j['indices']} == remaining
    value = {'gate': 'D-4', 'scope': 'FORMAL_TEST', 'tasks': tasks,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'submitted_records_retained': len(submitted), 'unsubmitted_records_frozen': len(remaining),
        'remaining_pairs': [[c, i] for c, i in sorted(remaining)],
        'submitted_job_ids_retained': [j['job_id'] for j in formal],
        'original_plan': ledger['formal_plan'], 'old_next_remaining_batch': ledger['next_remaining_batch'],
        'full_membership': args.membership, 'checkpoint_mode': 'question_boundary',
        'target_gold_loaded': False, 'packing': 5, 'max_concurrent_a800': 8,
        'time_limit': '14-00:00:00', 'partition': 'emergency_gpu', 'qos': 'emergency_gpu'}
    write(stage / 'plan.json', value)
    write(stage / 'cell-roots.json', dict(roots))
    write(stage / 'submitted-children.json', {'jobs': submitted_children})
    print(json.dumps({k: value[k] for k in ('submitted_records_retained','unsubmitted_records_frozen','old_next_remaining_batch')}))
    print(json.dumps({'task_count': len(tasks), 'records_per_task': [t['records'] for t in tasks],
        'relative_weight_per_task': loads, 'plan': str(stage / 'plan.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ledger','membership','stage','runs'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--task-count', type=int, default=8)
    run(p.parse_args())
