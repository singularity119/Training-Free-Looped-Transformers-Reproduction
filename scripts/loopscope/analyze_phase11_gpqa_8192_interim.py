#!/usr/bin/env python3
"""Single authorized D-4 report snapshot; acquisition and final closure stay unchanged."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path

from tflt.loopscope.phase11_accuracy import load_pool, validate_record, extract_final_answer
from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_panel import build_panel, require, write_json_once
from tflt.loopscope.phase11_posthoc import extract_s1, S1_PATTERN

SCHEMA = 'loopscope.phase11.gpqa_8192.v1'
APPROVED = {'5fc00f067773b2a20e355bf4def462781214b99f', 'ba218fc60cc7eb3bd0e45a7e30002549b4404357'}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def complete_lines(path):
    """Capture bytes once; only fully flushed newline-delimited rows are eligible."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b'\n')
    return [json.loads(line) for line in data[:end + 1].splitlines() if line.strip()]


def root_bundles(root):
    complete = root / 'complete'
    if complete.exists():
        # Capture this root's published file list once. New files stay outside this snapshot.
        paths = sorted(complete.glob('*.json'))
        for path in paths:
            bundle = read(path)
            require(set(bundle) == {'record', 'telemetry'}, 'incomplete committed bundle')
            require(path.stem == f"{bundle['record']['canonical_index']:06d}", 'bundle filename differs')
            yield bundle['record'], bundle['telemetry']
    else:
        records = complete_lines(root / 'records.jsonl')
        facts = complete_lines(root / 'telemetry.jsonl')
        require(len(facts) <= len(records), 'telemetry is ahead of records')
        for record, fact in zip(records, facts):
            yield record, fact


def validate(record, fact, row, cell, declared):
    validate_record(record, 'gpqa_main')
    i = record['canonical_index']
    require(i in declared and record['identity'] == row['identity']
            and record['category'] == row['subject'] and record['candidate_labels'] == row['choice_labels']
            and record['max_new_tokens'] == 8192, 'snapshot record identity/input/scope differs')
    require(fact['canonical_index'] == i and fact['identity'] == record['identity']
            and fact['prefill_count'] == 1 and fact['decode_count'] == record['generated_token_count'] - 1
            and fact['cache_growth_closed'] is True, 'snapshot cache lifecycle differs')
    if 'cell' in fact:
        require(fact['cell'] == cell and fact['cell_id'] == cell['cell_id'], 'checkpoint cell differs')
    if cell['arm'] == 'Online':
        require(fact['online']['decode_fit_count'] == 0
                and fact['online']['directions_released'] is True, 'snapshot direction lifecycle differs')


def freeze(args):
    output = Path(args.output_dir)
    require(not output.exists(), 'snapshot output must be fresh')
    frozen, roots = read(args.membership), read(args.cell_roots)
    panel = build_panel('gpqa_main')
    require(frozen['schema'] == SCHEMA and frozen['record_count'] == 8064
            and frozen['cells'] == [{'cell': c, 'indices': list(range(448))} for c in panel['cells']],
            'snapshot full membership differs')
    require(set(roots) == {c['cell_id'] for c in panel['cells']}, 'snapshot roots lack exact cells')
    pool = load_pool(Path(frozen['pool']), 'gpqa_main', 'FORMAL_TEST')
    require(frozen['samples'] == [{'canonical_index': i, 'identity': r['identity'],
        'prompt_token_length': len(r['input_ids'])} for i, r in enumerate(pool['rows'])], 'input membership differs')
    output.mkdir(parents=True, exist_ok=False)
    begun, members, source_facts = now(), {}, []
    snapshot = output / 'raw_snapshot.jsonl'
    with snapshot.open('x', encoding='utf-8') as handle:
        for cell in panel['cells']:
            cid, seen = cell['cell_id'], {}
            for directory in roots[cid]:
                root = Path(directory)
                if not (root / 'command_args.json').exists():
                    require(not (root / 'records.jsonl').exists() and not (root / 'complete').exists(),
                            'records exist without source metadata')
                    continue
                meta = read(root / 'command_args.json')
                require(meta['schema'] == SCHEMA and meta['cell'] == cell and meta['scope'] == 'FORMAL_TEST'
                        and meta['max_new_tokens'] == 8192 and meta['target_gold_loaded'] is False
                        and meta['synthetic_min_new_tokens'] is None and meta['pool'] == frozen['pool']
                        and meta['source_commit'] in APPROVED and read(meta['membership']) == frozen,
                        'snapshot attempt science/provenance differs')
                declared = meta['indices']
                require(declared and len(set(declared)) == len(declared)
                        and all(type(i) is int and 0 <= i < 448 for i in declared), 'invalid attempt indices')
                count = 0
                for record, fact in root_bundles(root):
                    i = record['canonical_index']
                    require(type(i) is int and 0 <= i < 448 and i not in seen, 'snapshot duplicate/outside index')
                    validate(record, fact, pool['rows'][i], cell, declared)
                    seen[i] = record['identity']
                    handle.write(json.dumps({'cell_id': cid, 'source_root': str(root),
                        'source_commit': meta['source_commit'], 'record': record, 'telemetry': fact},
                        ensure_ascii=False, allow_nan=False) + '\n')
                    count += 1
                source_facts.append({'cell_id': cid, 'root': str(root), 'source_commit': meta['source_commit'],
                                     'snapshotted_records': count})
            members[cid] = [{'canonical_index': i, 'identity': seen[i]} for i in sorted(seen)]
        handle.flush()
        os.fsync(handle.fileno())
    # This is a report snapshot, never a full-panel closure or Gate terminal.
    barrier = {'status': 'INTERIM_EXACT_SNAPSHOT_SAVED_GOLD_UNREAD', 'gold_read': False,
        'target_gold_loaded': False, 'started_at_utc': begun, 'frozen_at_utc': now(),
        'record_count': sum(len(v) for v in members.values()), 'cell_membership': members,
        'record_counts': {cid: len(v) for cid, v in members.items()}, 'source_roots': source_facts,
        'full_membership': str(args.membership), 'canonical_pool': frozen['pool'],
        'raw_snapshot': str(snapshot), 'selection': 'fully published valid records enumerated once per root',
        'selection_warning': 'completion depends on scheduling and output length; selected nonrandom subset',
        'scope': 'ONE_INTERIM_REPORT_SNAPSHOT_NOT_FINAL'}
    write_json_once(output / 'snapshot_barrier.json', barrier)
    print(json.dumps({'status': barrier['status'], 'record_count': barrier['record_count'],
                      'record_counts': barrier['record_counts'], 'snapshot_dir': str(output)}), flush=True)


def score(args):
    from tflt.loopscope.phase11_interim_stats import analyze_snapshot
    output = Path(args.snapshot_dir)
    require(not (output / 'analysis.json').exists() and not (output / 'gold_read_receipt.json').exists(),
            'single snapshot already scored; no repeated gold look')
    barrier = read(output / 'snapshot_barrier.json')
    require(barrier['status'] == 'INTERIM_EXACT_SNAPSHOT_SAVED_GOLD_UNREAD'
            and barrier['gold_read'] is False, 'snapshot must be closed before gold')
    panel = build_panel('gpqa_main')
    aligned = {c['cell_id']: [] for c in panel['cells']}
    with (output / 'raw_snapshot.jsonl').open(encoding='utf-8') as handle, \
         (output / 'predictions.jsonl').open('x', encoding='utf-8') as predictions:
        for line in handle:
            item = json.loads(line)
            record, cid = item['record'], item['cell_id']
            aligned[cid].append({k: record[k] for k in ('identity','canonical_index','generated_text',
                'generated_token_count','end_reason','candidate_labels')})
            predictions.write(json.dumps({'cell_id': cid, 'identity': record['identity'],
                'S1': extract_s1(record['generated_text']),
                'S0': extract_final_answer(record['generated_text'], record['candidate_labels'])}) + '\n')
        predictions.flush()
        os.fsync(predictions.fileno())
    for cid, rows in aligned.items():
        rows.sort(key=lambda r: r['canonical_index'])
        require([{'canonical_index': r['canonical_index'], 'identity': r['identity']} for r in rows]
                == barrier['cell_membership'][cid], 'frozen snapshot membership differs')
    write_json_once(output / 'extraction_barrier.json', {'status':'INTERIM_S1_S0_SAVED_GOLD_UNREAD',
        'timestamp': now(), 'record_count': barrier['record_count'], 'S1_pattern': S1_PATTERN,
        'multiple_matching_lines': 'failure, including duplicate identical answers', 'gold_read': False})
    frozen = read(barrier['full_membership'])
    pool = load_pool(Path(frozen['pool']), 'gpqa_main', 'FORMAL_TEST')
    identities = [r['identity'] for r in pool['rows']]
    # First gold access happens after both immutable report barriers are saved.
    gold_rows = complete_lines(Path(args.gold))
    gold_time = now()
    gold = gold_indices(gold_rows, identities, {r['identity']: r['choice_labels'] for r in pool['rows']},
                        {r['identity']: r['subject'] for r in pool['rows']})
    write_json_once(output / 'gold_read_receipt.json', {'gold_read_timestamp': gold_time,
        'scope':'USER_AUTHORIZED_SINGLE_INTERIM_REPORT_SNAPSHOT', 'snapshot_barrier':str(output/'snapshot_barrier.json'),
        'extraction_barrier':str(output/'extraction_barrier.json'), 'canonical_gold_count':len(gold)})
    report = analyze_snapshot(panel, aligned, dict(zip(identities, gold)))
    report.update(status='INTERIM_EXPLORATORY_SNAPSHOT_ANALYSIS', gate='D-4', dataset='gpqa_main',
        record_count=barrier['record_count'], snapshot_started_at_utc=barrier['started_at_utc'],
        snapshot_frozen_at_utc=barrier['frozen_at_utc'], gold_read_timestamp=gold_time,
        executor_commit=args.executor_commit, snapshot_dir=str(output), max_new_tokens=8192,
        interpretation={'not_final':True,'completion_selection_bias':barrier['selection_warning'],
            'own_cell_results':'actual completed n; absent records excluded, extraction failures wrong',
            'fair_comparisons':'same all18-cell completed identity intersection; still completion-selected exploratory',
            'no_outcome_feedback':True,'final_full8064_closure_still_required':True},
        acquisition_changed=False, gpu_jobs_submitted=False, mmlu_accessed=False, gate_pass_claimed=False)
    write_json_once(output / 'analysis.json', report)
    with (output / 'cell_table.csv').open('x', encoding='utf-8', newline='') as handle:
        fields=['cell_id','arm','k','direction_policy','strength','n','S1_correct','S1_accuracy',
                'S1_extraction_success_rate','truncated_rate','S0_correct','S0_accuracy',
                'consistent_multiple','conflicting_multiple']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        other={r['cell_id']:r for r in report['s0']['own_completion']}
        for r in report['s1']['own_completion']:
            diagnostics=r['s1_duplicate_line_counts']
            writer.writerow({**{k:r[k] for k in fields if k in r},'S1_correct':r['correct'],
                'S1_accuracy':r['accuracy'],'S1_extraction_success_rate':r['extraction_success_rate'],
                'S0_correct':other[r['cell_id']]['correct'],'S0_accuracy':other[r['cell_id']]['accuracy'],
                **{k:diagnostics[k] for k in ('consistent_multiple','conflicting_multiple')}})
    print(json.dumps({'status':report['status'],'record_count':report['record_count'],
        'common_intersection_n':report['s1']['common_intersection']['n'],'output':str(output)}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for name in ('membership','cell-roots','output-dir'):f.add_argument('--'+name,required=True)
    f.set_defaults(func=freeze)
    s=sub.add_parser('score')
    for name in ('snapshot-dir','gold','executor-commit'):s.add_argument('--'+name,required=True)
    s.set_defaults(func=score)
    args=p.parse_args();args.func(args)
