"""Outcome-free closure of Phase 8 choice-score shards before gold unsealing."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .phase8_test_pool import validate_test_pool
from .phase8_accuracy import panel as frozen_panel, select_rows


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest, pool, pool_path):
    _require(manifest.get('schema') == 'loopscope.phase8.accuracy_manifest.v1', 'manifest schema mismatch')
    _require(manifest['dataset'] == pool['dataset'] and manifest['pool'] == str(pool_path),
             'manifest dataset/pool mismatch')
    config = _read(Path(__file__).resolve().parents[3] / 'configs/loopscope/phase8_model_windows.json')
    expected = frozen_panel(config)
    _require(len(manifest['cells']) == 18, 'manifest requires frozen 18-cell panel')
    actual = {cell['cell_id']: cell for cell in manifest['cells']}
    _require(actual == {cell['cell_id']: cell for cell in expected},
             'manifest differs from frozen model/window/K/cache/basis panel')
    return {cell['cell_id']: cell for cell in manifest['cells']}


def verify(cell_roots, pool, manifest, full_panel=False, *, gate_e=False):
    """Verify shards without computing predictions, correctness or accuracy."""
    pool_path, manifest_path = Path(pool), Path(manifest)
    bundle, panel = _read(pool_path), _read(manifest_path)
    validate_test_pool(bundle)
    if gate_e:
        from .phase8_gate_e_accuracy import validate_manifest as validate_e
        cells = validate_e(panel, bundle, pool_path)
    else:
        cells = validate_manifest(panel, bundle, pool_path)
    canonical = bundle['rows']
    covered, records, commits, scopes = {}, [], set(), set()
    for root in map(Path, cell_roots):
        metadata = _read(root / 'command_args.json')
        summary = _read(root / 'summary.json')
        _require(summary['status'] == 'SCORES_COMPLETE', 'incomplete shard')
        _require(summary['metadata'] == metadata, 'summary/command metadata mismatch')
        _require(summary['target_gold_loaded'] is False, 'target gold loaded before closure')
        cell = metadata['cell']
        cell_id = cell['cell_id']
        _require(cell_id in cells and cell == cells[cell_id], 'shard configuration differs from manifest')
        _require(metadata['pool'] == str(pool_path) and metadata['manifest'] == str(manifest_path),
                 'shard source paths mismatch')
        _require(bool(metadata['source_commit']), 'missing source commit')
        env = _read(root / 'env.json')
        _require(env['source_commit'] == metadata['source_commit']
                 and env['model_revision'] == cell['revision']
                 and env['model_dtype'] == 'torch.' + cell['dtype'],
                 'actual runtime source/revision/dtype mismatch')
        commits.add(metadata['source_commit'])
        scope = metadata['scope']
        _require(scope in ('PREFLIGHT_ONLY', 'FORMAL_TEST'), 'invalid shard scope')
        scopes.add(scope)
        if gate_e:
            _require(scope == panel['scope'], 'Gate E shard/manifest scope mismatch')
        if scope == 'FORMAL_TEST':
            start, end = metadata['start'], metadata['end']
            _require(type(start) is int and type(end) is int and 0 <= start < end <= len(canonical),
                     'invalid canonical shard interval')
            indices = list(range(start, end))
        else:
            selected = select_rows(bundle, cell, scope, metadata['start'], metadata['end'])
            positions = {row['identity']: i for i, row in enumerate(canonical)}
            indices = [positions[row['identity']] for row in selected]
        expected = [canonical[i] for i in indices]
        seen = covered.setdefault(cell_id, set())
        _require(not seen.intersection(indices), 'overlapping or duplicate cell shards')
        count = 0
        with (root / 'scores.jsonl').open(encoding='utf-8') as handle:
            for count, line in enumerate(handle, 1):
                _require(count <= len(expected), 'extra score rows')
                row = json.loads(line)
                _require(set(row) == {'identity', 'subject', 'doc_index', 'scores'},
                         'score row must contain only identity and raw scores; no outcomes')
                target = expected[count - 1]
                _require(all(row[key] == target[key] for key in ('identity', 'subject', 'doc_index')),
                         'score identity/order differs from canonical shard')
                scores = row['scores']
                _require(isinstance(scores, list) and len(scores) == 4
                         and all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                         'four finite raw scores required')
        _require(count == len(expected) == summary['count'], 'incomplete shard row count')
        seen.update(indices)
        records.append({'root': str(root), 'cell_id': cell_id, 'scope': scope,
                        'count': count, 'start': metadata.get('start'), 'end': metadata.get('end')})
    _require(bool(records), 'no shards supplied')
    _require(len(scopes) == 1, 'preflight and formal shards cannot be mixed')
    if full_panel:
        _require(scopes == {'FORMAL_TEST'}, 'full closure requires formal test shards')
        _require(set(covered) == set(cells), f'full closure requires all {len(cells)} cells')
        _require(all(indices == set(range(14042)) for indices in covered.values()),
                 'every cell requires all 14042 canonical identities')
    return {'status': 'FULL_PANEL_CLOSED' if full_panel else 'SHARDS_CLOSED',
            'target_gold_loaded': False, 'manifest': str(manifest_path), 'pool': str(pool_path),
            'cell_roots': [record['root'] for record in records], 'cell_count': len(covered),
            'shard_count': len(records), 'sample_count': sum(len(v) for v in covered.values()),
            'counts_by_cell': {key: len(value) for key, value in covered.items()},
            'subject_count': len({row['subject'] for row in canonical}),
            'source_commits': sorted(commits), 'shards': records}
