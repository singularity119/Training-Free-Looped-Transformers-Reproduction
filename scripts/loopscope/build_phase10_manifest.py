#!/usr/bin/env python3
"""Build outcome-free Phase 10 independent/display or new-score manifests."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase10_panel import DATASETS, build_panel, load_config, score_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scores', action='store_true', help='emit full runtime cells, designate new score cells, and carry expected reuse map')
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--scope', choices=('FORMAL_TEST', 'PREFLIGHT_ONLY'), default='FORMAL_TEST')
    args = parser.parse_args()
    config = load_config(args.config)
    value = score_manifest(args.dataset, args.pool, args.scope, config) if args.scores else build_panel(args.dataset, config)
    write_manifest(args.output, value)
    print(json.dumps({key: value[key] for key in ('schema', 'dataset', 'cell_count', 'display_row_count',
          'expected_reuse_cell_count', 'validated_reuse_cell_count', 'new_cell_count', 'target_gold_loaded')}, sort_keys=True))


if __name__ == '__main__':
    main()
