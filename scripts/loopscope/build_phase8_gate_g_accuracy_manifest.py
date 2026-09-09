#!/usr/bin/env python3
"""Build frozen Gate G 9-cell score manifest, without target labels."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_gate_g_accuracy import panel, SCHEMA, validate_manifest
from tflt.loopscope.phase8_accuracy import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pool',type=Path,required=True)
p.add_argument('--basis-root',required=True)
p.add_argument('--scope',choices=['PREFLIGHT_ONLY','FORMAL_TEST'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
pool=json.loads(a.pool.read_text())
manifest=dict(schema=SCHEMA,scope=a.scope,pool=str(a.pool),dataset=pool['dataset'],basis_root=a.basis_root,cells=panel(a.basis_root))
validate_manifest(manifest,pool,a.pool)
write_json(a.output,manifest)
