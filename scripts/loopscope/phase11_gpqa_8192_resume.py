#!/usr/bin/env python3
"""D-4 question-boundary resume around the unchanged frozen acquisition function."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

from tflt.loopscope.phase11_checkpoint import RecordStore
from tflt.loopscope.phase11_accuracy import load_pool, validate_record
from tflt.loopscope.phase11_panel import load_config, require, write_json_once
from phase11_gpqa_8192 import membership, read, DATASET, LIMIT, SCHEMA
from run_phase11_accuracy import acquire_row, load_runtime, source_provenance, ForwardTiming


def validator_for(pool, cell):
    def validate(index, record, fact):
        validate_record(record, DATASET)
        row = pool['rows'][index]
        require(record['canonical_index'] == index and record['identity'] == row['identity']
                and record['category'] == row['subject']
                and record['candidate_labels'] == row['choice_labels']
                and record['max_new_tokens'] == LIMIT, 'checkpoint identity/input/length differs')
        require(fact['canonical_index'] == index and fact['identity'] == record['identity']
                and fact['cell_id'] == cell['cell_id'] and fact['cell'] == cell
                and fact['prefill_count'] == 1
                and fact['decode_count'] == record['generated_token_count'] - 1
                and fact['cache_growth_closed'] is True, 'checkpoint cell/cache lifecycle differs')
        if cell['arm'] == 'Online':
            require(fact['online']['decode_fit_count'] == 0
                    and fact['online']['directions_released'] is True, 'checkpoint direction lifecycle differs')
    return validate


def acquire(args):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == args.partition
            and (args.partition == 'debug' if args.synthetic else args.partition != 'debug'),
            'resume producer requires the admitted allocation')
    frozen = read(args.membership)
    require(all(frozen.get(k) == v for k, v in membership().items()), 'full membership differs')
    cells = {item['cell']['cell_id']: item for item in frozen['cells']}
    require(args.cell_id in cells, 'cell outside D-4')
    cell = cells[args.cell_id]['cell']
    scope = 'PREFLIGHT_ONLY' if args.synthetic else 'FORMAL_TEST'
    pool = load_pool(args.pool, DATASET, scope)
    indices = [0] if args.synthetic else [int(i) for i in args.indices.split(',')]
    require(indices and len(indices) == len(set(indices))
            and all(i in cells[args.cell_id]['indices'] for i in indices), 'indices outside full panel')
    if not args.synthetic:
        require(frozen['pool'] == str(args.pool), 'formal pool differs')
    source = source_provenance(args.commit)
    args.run_root.mkdir(parents=True, exist_ok=True)
    # A scheduler restart must never race another writer at the same question root.
    with (args.run_root / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = {'schema': SCHEMA, 'cell': cell, 'indices': indices, 'scope': scope,
            'max_new_tokens': LIMIT, 'synthetic_min_new_tokens': 3 if args.synthetic else None,
            'target_gold_loaded': False, 'source_commit': args.commit,
            'pool': str(args.pool), 'membership': str(args.membership), **source}
        metadata_path = args.run_root / 'command_args.json'
        if metadata_path.exists():
            metadata = read(metadata_path)
            require({k: metadata.get(k) for k in contract} == contract, 'resume contract/source differs')
        else:
            metadata = {**contract, 'argv': sys.argv, 'checkpoint_mode': 'question_boundary'}
            write_json_once(metadata_path, metadata)
        store = RecordStore(args.run_root, indices, validator_for(pool, cell))
        if (args.run_root / 'summary.json').exists():
            summary = read(args.run_root / 'summary.json')
            require(not store.pending and summary['metadata'] == metadata
                    and summary['count'] == len(indices), 'existing summary differs from checkpoint closure')
            store.finalize_records()
            print(json.dumps({'status': 'REUSED_COMPLETE', 'count': len(indices)}), flush=True)
            return
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex
        attempt = args.run_root / 'attempts' / tag
        attempt.mkdir(parents=True, exist_ok=False)
        write_json_once(attempt / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'],
            'recovered_done': list(store.done), 'pending': store.pending, 'target_gold_loaded': False})
        start = time.monotonic()
        timing = None
        try:
            if store.pending:
                config = {**load_config(), 'max_new_tokens': LIMIT}
                torch, model, tokenizer, adapter, eos_ids, env = load_runtime(cell, config, DATASET)
                capacity = model.config.max_position_embeddings
                require(capacity == config['max_length']
                        and all(len(pool['rows'][i]['input_ids']) + LIMIT <= capacity for i in indices),
                        'original full input/context differs')
                environment = {**env, **source, 'source_commit': args.commit,
                    'native_context_capacity': capacity, 'target_gold_loaded': False}
                write_json_once(attempt / 'env.json', environment)
                if not (args.run_root / 'env.json').exists():
                    write_json_once(args.run_root / 'env.json', environment)
                torch.set_grad_enabled(False)
                torch.manual_seed(20261008)
                torch.cuda.reset_peak_memory_stats()
                timing = ForwardTiming(torch, model, DATASET)
                for index in store.pending:
                    store.mark_in_progress(index)
                    torch.cuda.synchronize()
                    began = time.monotonic()
                    record, fact = acquire_row(torch, model, tokenizer, adapter, eos_ids,
                        pool['rows'][index], index, cell, config, scope, LIMIT, 3 if args.synthetic else None)
                    torch.cuda.synchronize()
                    fact.update(timing.summary())
                    fact.update(canonical_index=index, identity=record['identity'],
                        cell_id=cell['cell_id'], cell=cell, elapsed_seconds=time.monotonic() - began,
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                    store.commit(index, record, fact)
                    print(json.dumps({'cell_id': cell['cell_id'], 'canonical_index': index,
                        'completed': len(store.done), 'target_count': len(indices)}), flush=True)
            store.finalize_records()
            bundles = list(store.done.values())
            write_json_once(args.run_root / 'summary.json', {'schema': SCHEMA,
                'status': 'GPQA_8192_ATTEMPT_COMPLETE', 'metadata': metadata, 'count': len(bundles),
                'generated_token_count': sum(b['record']['generated_token_count'] for b in bundles),
                'elapsed_seconds': sum(b['telemetry']['elapsed_seconds'] for b in bundles),
                'peak_allocated_bytes': max(b['telemetry']['peak_allocated_bytes'] for b in bundles),
                'peak_reserved_bytes': max(b['telemetry']['peak_reserved_bytes'] for b in bundles),
                'target_gold_loaded': False, 'checkpoint_mode': 'question_boundary'})
            write_json_once(attempt / 'complete.json', {'status': 'COMPLETE', 'count': len(bundles),
                'elapsed_seconds': time.monotonic() - start, 'target_gold_loaded': False})
        except Exception as exc:
            write_json_once(attempt / 'failure.json', {'status': 'ATTEMPT_FAILED',
                'exception_type': type(exc).__name__, 'completed': len(store.done),
                'elapsed_seconds': time.monotonic() - start, 'target_gold_loaded': False})
            raise
        finally:
            if timing is not None:
                timing.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('acquire')
    for name in ('membership', 'pool', 'run-root'):
        a.add_argument('--' + name, type=Path, required=True)
    for name in ('cell-id', 'commit', 'partition'):
        a.add_argument('--' + name, required=True)
    a.add_argument('--indices', default='0')
    a.add_argument('--synthetic', action='store_true')
    acquire(p.parse_args())
