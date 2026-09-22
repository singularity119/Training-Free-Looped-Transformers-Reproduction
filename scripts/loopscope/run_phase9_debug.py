#!/usr/bin/env python3
"""Phase 9 debug/diagnostic entry point.

Gate A only validates the manifest or exposes the future tensor smoke path;
model loading, validation-pool acquisition, and GPU execution belong to Gate B
and are deliberately not implicit here.
"""

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase9_accuracy import load_config, validate_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("manifest", "tensor-smoke"), default="manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    counts = validate_panel(manifest.get("cells", []), config)
    if args.mode == "manifest":
        print(json.dumps({"status": "DEBUG_MANIFEST_READY", **counts}, sort_keys=True))
        return
    import torch
    from tflt.loopscope.phase9_runtime import tensor_smoke_checks
    print(json.dumps({"status": "TENSOR_SMOKE_COMPLETE", "checks": tensor_smoke_checks(torch)}, sort_keys=True))


if __name__ == "__main__":
    main()
