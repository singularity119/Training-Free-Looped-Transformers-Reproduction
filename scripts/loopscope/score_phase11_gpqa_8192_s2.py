#!/usr/bin/env python3
"""User-authorized S2 rescore of the single existing 7032-record interim snapshot."""
import argparse
import csv
import json
import os
from pathlib import Path

from analyze_phase11_gpqa_8192_interim import read, now, complete_lines
from tflt.loopscope.phase11_accuracy import load_pool
from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_panel import build_panel, require, write_json_once
from tflt.loopscope.phase11_posthoc import S1_PATTERN
from tflt.loopscope.phase11_s2 import extract_s2, analyze_s2_snapshot


def run(args):
    snapshot, output = Path(args.snapshot_dir), Path(args.output_dir)
    require(not output.exists(), 'S2 output root must be fresh; preserve previous scores')
    barrier = read(snapshot / 'snapshot_barrier.json')
    require(barrier['status'] == 'INTERIM_EXACT_SNAPSHOT_SAVED_GOLD_UNREAD'
            and barrier['record_count'] == 7032 and barrier['scope'] == 'ONE_INTERIM_REPORT_SNAPSHOT_NOT_FINAL',
            'S2 is restricted to the previously frozen 7032-record snapshot')
    output.mkdir(parents=True, exist_ok=False)
    spec = {'schema': 'loopscope.phase11.s2.v1', 'parser': 'S2', 'frozen_at_utc': now(),
        'valid_line_pattern': S1_PATTERN, 'flags': 're.IGNORECASE',
        'line_processing': 'splitlines then strip then fullmatch',
        'normalization': 'matched A-D letter converted to uppercase',
        'success': 'at least one valid line and exactly one distinct normalized letter',
        'consistent_multiple': 'use the shared letter as prediction; gold determines correct or incorrect',
        'conflicting_multiple': 'two or more distinct letters are extraction failure and wrong',
        'no_match': 'extraction failure and wrong; do not infer prose answers',
        'snapshot_dir': str(snapshot), 'record_count': 7032, 'gold_read': False,
        'scope': 'USER_AUTHORIZED_POSTHOC_EXPLORATORY_S2_REPORT'}
    write_json_once(output / 's2_spec.json', spec)
    panel = build_panel('gpqa_main')
    aligned = {c['cell_id']: [] for c in panel['cells']}
    counts = {cid: 0 for cid in aligned}
    with (snapshot / 'raw_snapshot.jsonl').open(encoding='utf-8') as source, \
         (output / 's2_predictions.jsonl').open('x', encoding='utf-8') as predictions:
        for line in source:
            item = json.loads(line)
            record, cid = item['record'], item['cell_id']
            require(cid in aligned, 'unexpected snapshot cell')
            aligned[cid].append({k: record[k] for k in ('identity','canonical_index','generated_text',
                'generated_token_count','end_reason','candidate_labels')})
            predictions.write(json.dumps({'cell_id': cid, 'identity': record['identity'],
                'canonical_index': record['canonical_index'], 'S2': extract_s2(record['generated_text'])}) + '\n')
            counts[cid] += 1
        predictions.flush()
        os.fsync(predictions.fileno())
    require(counts == barrier['record_counts'] and sum(counts.values()) == 7032, 'snapshot count differs')
    for cid, rows in aligned.items():
        rows.sort(key=lambda r: r['canonical_index'])
        require([{'canonical_index':r['canonical_index'],'identity':r['identity']} for r in rows]
                == barrier['cell_membership'][cid], 'snapshot identity/cell membership differs')
    common = set.intersection(*(set(r['identity'] for r in rows) for rows in aligned.values()))
    require(len(common) == 362, 'existing common362 identity intersection differs')
    write_json_once(output / 's2_extraction_barrier.json', {
        'status':'S2_SPEC_AND_ALL_PREDICTIONS_SAVED_BEFORE_GOLD', 'saved_at_utc':now(),
        'record_count':7032, 'record_counts':counts, 'common_intersection_n':362,
        's2_spec':str(output/'s2_spec.json'), 'predictions':str(output/'s2_predictions.jsonl'), 'gold_read':False})
    frozen = read(barrier['full_membership'])
    pool = load_pool(Path(frozen['pool']), 'gpqa_main', 'FORMAL_TEST')
    identities = [r['identity'] for r in pool['rows']]
    # S2 specification and complete extraction artifacts are persisted before this read.
    gold_rows = complete_lines(Path(args.gold))
    gold_time = now()
    gold = gold_indices(gold_rows, identities, {r['identity']:r['choice_labels'] for r in pool['rows']},
                        {r['identity']:r['subject'] for r in pool['rows']})
    write_json_once(output/'s2_gold_read_receipt.json', {'gold_read_timestamp':gold_time,
        's2_spec_frozen_at_utc':spec['frozen_at_utc'], 'extraction_barrier':str(output/'s2_extraction_barrier.json'),
        'scope':'USER_AUTHORIZED_S2_POSTHOC_RESCORE_SAME_FROZEN_SNAPSHOT', 'canonical_gold_count':len(gold)})
    report = analyze_s2_snapshot(panel, aligned, dict(zip(identities,gold)))
    require(report['s2']['common_intersection']['n'] == 362, 'paired population differs')
    report.update(status='S2_POSTHOC_EXPLORATORY_REPORT_SAME_FROZEN_SNAPSHOT', gate='D-4', dataset='gpqa_main',
        record_count=7032, max_new_tokens=8192, snapshot_dir=str(snapshot), s2_output_dir=str(output),
        snapshot_started_at_utc=barrier['started_at_utc'], snapshot_frozen_at_utc=barrier['frozen_at_utc'],
        s2_spec_frozen_at_utc=spec['frozen_at_utc'], s2_gold_read_timestamp=gold_time,
        executor_commit=args.executor_commit, primary_parser='S2', preserved_auxiliary_parsers=['S1','S0'],
        acquisition_changed=False, gpu_jobs_submitted=False, mmlu_accessed=False, gate_pass_claimed=False,
        interpretation={'scope':'user-requested posthoc S2 rescore; not final or confirmatory',
            'snapshot':'same frozen7032, no rolling additions; own completed n and common362 for comparisons',
            'selection_bias':barrier['selection_warning'],
            'multiplicity':'original30/12/2 families per S2; Holm does not remove posthoc/completion selection',
            'no_outcome_feedback':True, 'final_full8064_closure_still_required':True})
    write_json_once(output/'analysis_s2.json',report)
    fields=['cell_id','arm','k','direction_policy','strength','n','correct','accuracy',
        'extraction_success_rate','truncated_rate','consistent_multiple_count',
        'consistent_multiple_correct','consistent_multiple_incorrect','conflicting_multiple_count']
    with (output/'cell_table_s2.csv').open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in report['s2']['own_completion']:writer.writerow({k:row[k] for k in fields})
    print(json.dumps({'status':report['status'],'record_count':7032,'common_n':362,'output':str(output),
        'holm_rejections':sum(r['holm_reject_alpha_0_05'] for r in report['s2']['common_intersection']['contrasts'])}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot-dir','output-dir','gold','executor-commit'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
