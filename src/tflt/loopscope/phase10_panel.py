"""Outcome-free Phase 10 panel, K2 display aliases and expected historical reuse."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

Cell = dict[str, Any]
SCHEMA = 'loopscope.phase10.panel.v1'
CONFIG_SCHEMA = 'loopscope.phase10.panel_config.v1'
SCORE_SCHEMA = 'loopscope.phase10.score_manifest.v1'
DATASETS = ('mmlu', 'arc_challenge')
DIRECTION_POLICIES = ('fixed_t0', 'lag1', 'current_t')
STRENGTHS = tuple(i / 10 for i in range(1, 10))
EXPECTED_CELL_COUNT = 68
EXPECTED_DISPLAY_COUNT = 86
EXPECTED_ALIAS_COUNT = 18


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / 'configs/loopscope/phase10_panel.json'


def load_config(path: Path | None = None) -> dict[str, Any]:
    value = json.loads((Path(path) if path else default_config_path()).read_text(encoding='utf-8'))
    validate_config(value)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    """Check the scientific membership against immutable Phase 9 configuration."""
    from .phase9_accuracy import load_config as load_phase9
    legacy = load_phase9()
    _require(config.get('schema') == CONFIG_SCHEMA and config.get('phase') == 10, 'unexpected Phase 10 config')
    _require(config.get('scientific_contract') == '.planning/phase10/loopscope_phase10_contract_v2.md', 'Phase 10 contract must be v2')
    _require(config.get('direction_policies') == list(DIRECTION_POLICIES), 'Phase 10 policies are frozen')
    _require(config.get('strengths') == list(STRENGTHS), 'Phase 10 lambda grid is frozen')
    for key, expected in {'strategy': 'damped_euler', 'iteration_mode': 'block', 'alpha': 1.0,
                          'beta': 0.0, 'decode_mode': 'bypass', 'batch_size': 1, 'rank': 1}.items():
        _require(config.get(key) == expected, f'Phase 10 {key} differs from contract')
    _require(config.get('models') == legacy['models'], 'Phase 10 must inherit Phase 9 model/tokenizer recipes')
    _require(config.get('cells') == [
        {'model_index': 0, 'window': [15, 18], 'boundaries': [15, 19], 'k': 2},
        {'model_index': 0, 'window': [15, 18], 'boundaries': [15, 19], 'k': 3},
        {'model_index': 1, 'window': [12, 15], 'boundaries': [12, 16], 'k': 2},
    ], 'Phase 10 requires the three frozen cells')
    datasets = config.get('datasets', {})
    _require(set(datasets) == set(DATASETS), 'Phase 10 dataset membership differs')
    mmlu = datasets['mmlu']
    _require((mmlu.get('repo'), mmlu.get('revision'), mmlu.get('split'), mmlu.get('num_fewshot'),
              mmlu.get('evaluation_count'), mmlu.get('primary_metric')) ==
             (legacy['dataset_repo'], legacy['dataset_revision'], 'test', 5, 14042, 'acc'), 'MMLU recipe differs')
    arc = datasets['arc_challenge']
    _require((arc.get('repo'), arc.get('subset'), arc.get('split'), arc.get('num_fewshot'),
              arc.get('evaluation_count'), arc.get('primary_metric'), arc.get('secondary_metric'),
              arc.get('fewshot_seed'), arc.get('revision'), arc.get('max_length')) ==
             ('allenai/ai2_arc', 'ARC-Challenge', 'test', 25, 1172, 'acc_norm', 'acc', 20261002,
              '210d026faf9955653af8916fad021475a3f00453', 32768), 'ARC recipe differs')
    _require(config.get('bootstrap') == {'replicates': 10000, 'seed': 20261002, 'ci_level': 0.95}, 'bootstrap recipe differs')
    reuse = config.get('mmlu_reuse_mapping', [])
    expected_reuse = {'q4-native', 'q17-native', 'q4-w15-18-k2-loop', 'q4-w15-18-k3-loop',
                      'q17-w12-15-k2-loop', 'q4-w15-18-k2-online-fixed-t0-lambda0.5',
                      'q4-w15-18-k3-online-fixed-t0-lambda0.5', 'q17-w12-15-k2-online-fixed-t0-lambda0.5',
                      'q4-w15-18-k3-online-lag1-lambda0.5'}
    _require(len(reuse) == 9 and {row.get('cell_id') for row in reuse} == expected_reuse, 'MMLU expected reuse must have nine cells')
    _require(all(row.get('roots') and row.get('source_manifest') and row.get('source_manifest_schema')
                 and row.get('source_closure_schema') and row.get('source_score_fields') ==
                 ['identity', 'subject', 'doc_index', 'scores'] and row.get('verification_status') ==
                 'EXPECTED_REUSE_NOT_RAW_SCORE_VERIFIED_IN_GATE_A' for row in reuse), 'reuse requires explicit paths/schema and pending verification')


def strategy_id(cell: Mapping[str, Any]) -> str:
    return f"{cell['cell_id'].split('-online-')[0]}-{cell['direction_policy'].replace('_', '-')}"


def _base(config: Mapping[str, Any], dataset: str, model_index: int) -> Cell:
    model = config['models'][model_index]
    return {'phase': 10, 'dataset': dataset, 'model_index': model_index, 'model': model['model'],
            'revision': model['revision'], 'tokenizer_revision': model['revision'], 'dtype': model['dtype'],
            'cache_strategy': model['cache_strategy'], 'alpha': 1.0, 'beta': 0.0,
            'iteration_mode': 'block', 'strategy': 'damped_euler', 'decode_mode': 'bypass',
            'batch_size': 1, 'rank': 1}


def build_panel(dataset: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Expand 68 independent results and 86 display rows; reads no model/data/scores.

    K2 aliasing is the frozen intended execution plan. Its admission remains
    conditional on Gate B real numerical equivalence, recorded separately.
    """
    _require(dataset in DATASETS, 'dataset must be mmlu or arc_challenge')
    config = load_config() if config is None else config
    validate_config(config)
    reuse = copy.deepcopy(config['mmlu_reuse_mapping']) if dataset == 'mmlu' else []
    reuse_by_id = {row['cell_id']: row for row in reuse}
    cells, display = [], []

    def add(cell: Cell, alias_of: str | None = None) -> None:
        cell.update(source='expected_historical_reuse' if cell['cell_id'] in reuse_by_id else 'phase10',
                    new_in_phase10=cell['cell_id'] not in reuse_by_id)
        if alias_of is None:
            cells.append(cell)
        display.append({**cell, 'result_cell_id': alias_of or cell['cell_id'], 'alias_of': alias_of,
                        'independent_result': alias_of is None})

    for index in range(2):
        add({**_base(config, dataset, index), 'cell_id': ('q4', 'q17')[index] + '-native',
             'window': None, 'boundaries': None, 'k': None, 'arm': 'Native',
             'direction_policy': None, 'strength': 0.0, 'intervention_mode': None})
    for spec in config['cells']:
        base = {**_base(config, dataset, spec['model_index']),
                **{key: copy.deepcopy(spec[key]) for key in ('window', 'boundaries', 'k')}}
        stem = f"{('q4','q17')[spec['model_index']]}-w{spec['window'][0]}-{spec['window'][1]}-k{spec['k']}"
        add({**base, 'cell_id': stem + '-loop', 'arm': 'Loop', 'direction_policy': None,
             'strength': 0.0, 'intervention_mode': None})
        for policy in DIRECTION_POLICIES:
            for strength in STRENGTHS:
                cell_id = f"{stem}-online-{policy.replace('_','-')}-lambda{strength:.1f}"
                alias = f'{stem}-online-fixed-t0-lambda{strength:.1f}' if policy == 'lag1' and spec['k'] == 2 else None
                add({**base, 'cell_id': cell_id, 'arm': 'Online', 'direction_policy': policy,
                     'strength': strength, 'intervention_mode': 'spectral'}, alias)
    recipe = copy.deepcopy(config['datasets'][dataset])
    return {'schema': SCHEMA, 'phase': 10, 'scientific_contract': config['scientific_contract'],
            'dataset': dataset, 'dataset_recipe': recipe,
            'target_gold_loaded': False, 'cells': cells, 'display_rows': display,
            'cell_count': EXPECTED_CELL_COUNT, 'independent_cell_count': EXPECTED_CELL_COUNT,
            'display_row_count': EXPECTED_DISPLAY_COUNT, 'alias_count': EXPECTED_ALIAS_COUNT,
            'reuse_mapping': reuse, 'expected_reuse_cell_count': len(reuse), 'validated_reuse_cell_count': 0,
            'new_cell_count': EXPECTED_CELL_COUNT - len(reuse), 'sample_count_per_cell': recipe['evaluation_count'],
            'new_configuration_sample_records': (EXPECTED_CELL_COUNT - len(reuse)) * recipe['evaluation_count'],
            'total_configuration_sample_records': EXPECTED_CELL_COUNT * recipe['evaluation_count'],
            'k2_alias_admission': 'Gate A local and Gate B real path equivalence required before formal reuse',
            'closure': 'logical membership only; expected historical reuse is not raw-score verification'}


def validate_panel(value: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, int]:
    expected = build_panel(value.get('dataset'), config)
    _require(dict(value) == expected, 'Phase 10 panel differs from frozen membership/aliases/reuse')
    return {key: expected[key] for key in ('cell_count', 'display_row_count', 'alias_count', 'expected_reuse_cell_count', 'new_cell_count')}


def new_cells(dataset: str, config: Mapping[str, Any] | None = None) -> list[Cell]:
    return [cell for cell in build_panel(dataset, config)['cells'] if cell['new_in_phase10']]


def score_manifest(dataset: str, pool_path: Path | None = None, scope: str = 'FORMAL_TEST',
                   config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _require(scope in ('FORMAL_TEST', 'PREFLIGHT_ONLY'), 'unexpected score scope')
    panel = build_panel(dataset, config)
    return {**panel, 'schema': SCORE_SCHEMA, 'scope': scope,
            'pool': str(pool_path) if pool_path is not None else None,
            'score_cells': new_cells(dataset, config), 'score_cell_count': panel['new_cell_count'],
            'closure': 'new score identities only; expected historical cells remain separately mapped'}


def validate_score_manifest(value: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                            pool_path: Path | None = None, scope: str | None = None) -> dict[str, Cell]:
    expected = score_manifest(value.get('dataset'), Path(value['pool']) if value.get('pool') else None,
                              value.get('scope'), config)
    _require(dict(value) == expected, 'Phase 10 score manifest differs from frozen new-cell membership')
    if pool_path is not None:
        _require(value.get('pool') == str(pool_path), 'score manifest pool differs from requested pool')
    if scope is not None:
        _require(value.get('scope') == scope, 'score manifest scope differs from requested scope')
    return {cell['cell_id']: cell for cell in expected['cells']}


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
