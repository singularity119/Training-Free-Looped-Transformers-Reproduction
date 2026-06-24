"""Model registry helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "models.json"


def load_model_registry(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    registry_path = Path(path) if path else DEFAULT_REGISTRY
    with registry_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["models"]


def resolve_model(alias_or_repo: str, registry_path: Optional[str] = None) -> Dict[str, Any]:
    registry = load_model_registry(registry_path)
    if alias_or_repo in registry:
        item = dict(registry[alias_or_repo])
        item["alias"] = alias_or_repo
        return item
    for alias, item in registry.items():
        if item.get("repo_id") == alias_or_repo:
            out = dict(item)
            out["alias"] = alias
            return out
    return {"alias": alias_or_repo, "repo_id": alias_or_repo, "kind": "custom"}
