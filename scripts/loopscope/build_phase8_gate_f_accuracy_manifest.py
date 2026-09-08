#!/usr/bin/env python3
"""Build frozen Gate F 16-cell score manifest, without target labels."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_gate_f_accuracy import panel, SCHEMA, validate_manifest
from tflt.loopscope.phase8_accuracy import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pool',type=Path,required=True)
p.add_argument('--t0-basis-root',required=True)
p.add_argument('--t1-basis-root',required=True)
p.add_argument('--scope',choices=['PREFLIGHT_ONLY','FORMAL_TEST'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
pool=json.loads(a.pool.read_text())
manifest=dict(schema=SCHEMA,scope=a.scope,pool=str(a.pool),dataset=pool['dataset'],t0_basis_root=a.t0_basis_root,t1_basis_root=a.t1_basis_root,cells=panel(a.t0_basis_root,a.t1_basis_root))
validate_manifest(manifest,pool,a.pool)
write_json(a.output,manifest)
