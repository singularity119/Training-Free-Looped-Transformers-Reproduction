#!/usr/bin/env python3
"""Verify Phase 8 score closure without loading target outcomes."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_gate_f_verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--cell-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--full-panel', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.cell_roots, args.pool, args.manifest, args.full_panel)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
