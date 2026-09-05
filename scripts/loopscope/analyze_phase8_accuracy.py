#!/usr/bin/env python3
"""Unseal gold only after full Phase 8 panel closure and report frozen statistics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tflt.loopscope.phase8_accuracy_stats import analyze_panel, stratified_paired_bootstrap
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def load_frozen_gold(pool_rows, cache_dir):
    """Only called downstream of saved fresh pre-outcome closure."""
    for name in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[name] = '1'
    from datasets import DownloadConfig, DownloadMode, load_dataset
    by_subject = defaultdict(list)
    for row in pool_rows:
        by_subject[row['subject']].append(row)
    labels, evidence = {}, []
    for subject, rows in sorted(by_subject.items()):
        raw = load_dataset(DATASET_REPO, subject, revision=DATASET_REVISION,
                           split='test', cache_dir=cache_dir,
                           download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                           download_config=DownloadConfig(local_files_only=True))
        urls = sorted(str(url) for url in (getattr(raw.info, 'download_checksums', None) or {}))
        cache_files = [str(item['filename']) for item in raw.cache_files]
        if urls:
            if any(DATASET_REVISION not in url for url in urls):
                raise ValueError('loaded MMLU source URLs differ from frozen revision')
            source = 'source_urls'
        else:
            if not cache_files or any(Path(path).parent.name != DATASET_REVISION for path in cache_files):
                raise ValueError('loaded MMLU cache version differs from frozen revision')
            source = 'loaded_cache_version_directory'
        if len(raw) != len(rows):
            raise ValueError(f'test subject count differs: {subject}')
        for row in rows:
            doc = raw[row['doc_index']]
            if (doc['subject'] != subject or doc['question'] != row['question'] or
                    list(doc['choices']) != row['choices']):
                raise ValueError('gold source question/choice identity mismatch')
            answer = doc['answer']
            if answer not in (0, 1, 2, 3):
                raise ValueError('invalid original A/B/C/D gold index')
            labels[row['identity']] = int(answer)
        evidence.append({'subject': subject, 'split': 'test', 'count': len(raw),
                         'revision_evidence': source, 'source_urls': urls, 'cache_files': cache_files})
    return [labels[row['identity']] for row in pool_rows], evidence


def aligned_cells(closure, manifest, pool_rows):
    by_cell = defaultdict(dict)
    for root in closure['cell_roots']:
        root = Path(root)
        cell_id = read_json(root / 'command_args.json')['cell']['cell_id']
        with (root / 'scores.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['identity'] in by_cell[cell_id]:
                    raise ValueError('duplicate sample while aligning closed panel')
                by_cell[cell_id][row['identity']] = row['scores']
    return [dict(cell, scores=[by_cell[cell['cell_id']][row['identity']] for row in pool_rows])
            for cell in manifest['cells']]


def independent_count_check(cells, gold, subjects, result):
    """Fresh scalar recomputation from saved raw scores; no cached correctness."""
    from scipy.stats import binomtest
    correct = {}
    reported = {cell['cell_id']: cell for cell in result['cells']}
    for cell in cells:
        flags = []
        for scores, answer in zip(cell['scores'], gold):
            best = 0
            for index in range(1, 4):
                if scores[index] > scores[best]:
                    best = index
            flags.append(best == answer)
        groups = defaultdict(list)
        for subject, flag in zip(subjects, flags):
            groups[subject].append(flag)
        macro = math.fsum(sum(flags) / len(flags) for flags in groups.values()) / len(groups)
        if not math.isclose(macro, reported[cell['cell_id']]['subject_macro_accuracy'], abs_tol=1e-14):
            raise ValueError('fresh subject macro accuracy recomputation failed')
        correct[cell['cell_id']] = flags
        if sum(flags) != reported[cell['cell_id']]['correct']:
            raise ValueError('fresh accuracy recomputation failed')
        if sum(flags) / len(gold) != reported[cell['cell_id']]['micro_accuracy']:
            raise ValueError('fresh micro accuracy recomputation failed')
    fresh_pvalues = {}
    for row in result['contrasts']:
        pairs = list(zip(correct[row['reference']], correct[row['treatment']]))
        gain = sum(1 for reference, treatment in pairs if treatment and not reference)
        loss = sum(1 for reference, treatment in pairs if reference and not treatment)
        if (gain, loss) != (row['wrong_to_right'], row['right_to_wrong']):
            raise ValueError('fresh discordant-pair recomputation failed')
        if (gain - loss) / len(gold) * 100 != row['delta_pp']:
            raise ValueError('fresh paired pp identity failed')
        differences = [int(t) - int(r) for r, t in pairs]
        groups = defaultdict(list)
        for subject, difference in zip(subjects, differences):
            groups[subject].append(difference)
        macro_delta = math.fsum(sum(v) / len(v) for v in groups.values()) / len(groups) * 100
        if not math.isclose(macro_delta, row['subject_macro_delta_pp'], abs_tol=1e-12):
            raise ValueError('fresh subject macro delta recomputation failed')
        pvalue = float(binomtest(gain, gain + loss, p=.5, alternative='two-sided').pvalue) if gain + loss else 1.0
        if not math.isclose(pvalue, row['mcnemar_exact_p'], rel_tol=1e-8, abs_tol=1e-14):
            raise ValueError('independent exact McNemar recomputation failed')
        fresh_pvalues[(row['reference'], row['treatment'])] = pvalue
        # A fresh call from the independently reconstructed flags checks saved
        # CIs; this deliberately shares the already-tested bootstrap algorithm.
        bootstrap = stratified_paired_bootstrap(differences, subjects)
        for key in ('ci_low_pp', 'ci_high_pp'):
            if bootstrap[key] != row['bootstrap'][key]:
                raise ValueError('fresh bootstrap CI recomputation failed')
    for family in ('K2_PRIMARY', 'K4_SECONDARY'):
        rows = [r for r in result['contrasts'] if r['family'] == family]
        rows.sort(key=lambda r: fresh_pvalues[(r['reference'], r['treatment'])])
        running = 0.0
        for rank, row in enumerate(rows):
            running = min(1.0, max(running, (len(rows) - rank) *
                                  fresh_pvalues[(row['reference'], row['treatment'])]))
            if not math.isclose(running, row['holm_adjusted_p'], rel_tol=1e-8, abs_tol=1e-14):
                raise ValueError('fresh Holm recomputation failed')
            if (running <= .05) != row['holm_reject_alpha_0_05']:
                raise ValueError('fresh Holm significance decision differs')
    return {'status': 'VERIFIED', 'cell_count': len(cells),
            'contrast_count': len(result['contrasts']), 'n_per_cell': len(gold),
            'checks': ['raw-score argmax correct/N', 'wrong_to_right/right_to_wrong',
                       'delta_pp=(wrong_to_right-right_to_wrong)/N*100',
                       'subject macro accuracy and delta', 'scipy binomtest exact McNemar',
                       'independent sorted Holm',
                       'fresh bootstrap CI from raw flags using same tested algorithm']}


def write_tables(output_dir, result):
    cell_fields = ['cell_id', 'model', 'window', 'k', 'arm', 'correct', 'n',
                   'micro_accuracy', 'subject_macro_accuracy']
    contrast_fields = ['reference', 'treatment', 'family', 'comparison', 'n',
                       'reference_correct', 'treatment_correct', 'wrong_to_right',
                       'right_to_wrong', 'delta_pp', 'subject_macro_delta_pp',
                       'ci_low_pp', 'ci_high_pp', 'mcnemar_exact_p',
                       'holm_adjusted_p', 'holm_reject_alpha_0_05']
    for filename, rows, fields in [('cells.csv', result['cells'], cell_fields),
                                   ('contrasts.csv', result['contrasts'], contrast_fields)]:
        with (output_dir / filename).open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, **{key: row['bootstrap'][key] for key in
                                             ('ci_low_pp', 'ci_high_pp')}) if 'bootstrap' in row else row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', default=os.environ.get('HF_DATASETS_CACHE'))
    args = parser.parse_args()
    if not args.cache_dir:
        parser.error('--cache-dir or HF_DATASETS_CACHE is required')
    closure = read_json(args.closure)
    if closure.get('status') != 'FULL_PANEL_CLOSED' or closure.get('target_gold_loaded') is not False:
        raise ValueError('gold access requires saved full pre-outcome panel closure')
    from tflt.loopscope.phase8_accuracy_verify import verify
    fresh = verify(closure['cell_roots'], closure['pool'], closure['manifest'], full_panel=True)
    if fresh['status'] != 'FULL_PANEL_CLOSED' or fresh['target_gold_loaded'] is not False:
        raise ValueError('fresh full-panel verification did not close')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'pre_outcome_verification.json',
               dict(fresh, saved_at=now(), original_closure=str(args.closure)))
    manifest = read_json(closure['manifest'])
    pool_rows = read_json(closure['pool'])['rows']
    cells = aligned_cells(closure, manifest, pool_rows)
    gold_load_started = now()
    gold, sources = load_frozen_gold(pool_rows, args.cache_dir)
    result = analyze_panel(cells, gold, [row['subject'] for row in pool_rows])
    result.update(schema='loopscope.phase8.accuracy_analysis.v1',
                  closure=str(args.closure), manifest=closure['manifest'], pool=closure['pool'],
                  cell_roots=closure['cell_roots'], gold_load_started=gold_load_started,
                  completed_at=now(), dataset={'repo': DATASET_REPO, 'revision': DATASET_REVISION,
                                               'split': 'test'}, gold_source_evidence=sources)
    result['fresh_count_verification'] = independent_count_check(cells, gold, [row['subject'] for row in pool_rows], result)
    write_json(args.output_dir / 'analysis.json', result)
    write_tables(args.output_dir, result)
    print(json.dumps({'status': 'ANALYZED', 'output_dir': str(args.output_dir),
                      'cells': len(result['cells']), 'contrasts': len(result['contrasts'])}))


if __name__ == '__main__':
    main()
