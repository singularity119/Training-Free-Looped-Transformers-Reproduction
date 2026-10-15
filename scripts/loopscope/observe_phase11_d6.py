#!/usr/bin/env python3
"""One bounded, gold-free observation of the exact D6 jobset."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess

from observe_phase11_gpqa_8192 import accounting_ids


def observe(jobset_path):
    jobset = json.loads(Path(jobset_path).read_text())
    jobs = jobset['jobs']
    ids = [j['job_id'] for j in jobs]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('exact nonduplicate job IDs required')
    queue = subprocess.run(['/opt/slurm/bin/squeue', '-h', '-r', '-j', ','.join(ids),
                            '-o', '%i|%T|%R'], capture_output=True, text=True, timeout=20)
    account = subprocess.run(['/opt/slurm/bin/sacct', '-X', '-n', '-P', '-j',
        ','.join(dict.fromkeys(i.split('_')[0] for i in ids)),
        '-o', 'JobID,State,ExitCode,ElapsedRaw,AllocTRES'], capture_output=True, text=True, timeout=20)
    if queue.returncode or account.returncode:
        raise RuntimeError((queue.returncode, queue.stderr, account.returncode, account.stderr))
    qrows = [r.split('|') for r in queue.stdout.splitlines() if r.strip()]
    facts = {}
    for line in account.stdout.splitlines():
        row = line.split('|')
        for job_id in accounting_ids(row[0], ids):
            facts[job_id] = dict(state=row[1], exit_code=row[2], allocation_seconds=int(row[3]), alloc_tres=row[4])
    terminal = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
    failed = [{'job_id': i, **v} for i, v in facts.items()
              if v['state'].split()[0] in terminal and
              (v['state'] != 'COMPLETED' or v['exit_code'] != '0:0')]
    progress, batch_states = [], []
    for job in jobs:
        plan = json.loads(Path(job['plan']).read_text())
        task = next(t for t in plan['tasks'] if t['number'] == job['task_index'])
        summary = Path(task['batch_root']) / 'batch_summary.json'
        if summary.exists():
            batch_states.append({'job_id': job['job_id'], 'status': json.loads(summary.read_text())['status']})
        for child in task['jobs']:
            root = Path(child['run_root'])
            completed = len(list((root / 'complete').glob('*.json')))
            progress.append({'cell_key': child['cell_key'], 'run_root': str(root),
                'done': completed, 'target': len(child['indices']), 'summary_exists': (root / 'summary.json').exists()})
    missing = sorted(set(ids) - set(facts))
    all_terminal = not qrows and not missing and all(v['state'].split()[0] in terminal for v in facts.values())
    done, target = sum(v['done'] for v in progress), sum(v['target'] for v in progress)
    return {'gate': 'D-6', 'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'executor': jobset['executor'], 'jobset': str(jobset_path), 'job_ids': ids,
        'scheduler': facts, 'queue': qrows, 'queue_state_counts': dict(Counter(v[1] for v in qrows)),
        'missing_job_ids': missing, 'all_terminal': all_terminal, 'terminal_failures': failed,
        'batch_states': batch_states, 'done_record_count': done, 'target_count': target,
        'all_success': all_terminal and not failed and done == target and all(v['summary_exists'] for v in progress),
        'trigger_resume': bool(failed or all_terminal or any(v['status'] == 'FAILED' for v in batch_states)),
        'allocation_gpuh': sum(v['allocation_seconds'] for v in facts.values()) / 3600,
        'progress': progress, 'target_gold_loaded': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobset', required=True)
    print(json.dumps(observe(p.parse_args().jobset)))
