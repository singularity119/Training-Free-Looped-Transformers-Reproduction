"""Strict revision closure built from authoritative resolved artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RevisionClosure:
    model_repo: str
    model_commit: str
    tokenizer_repo: str
    tokenizer_commit: str
    renderer_hash: str
    config_hash: str
    lm_eval_version: str
    model_artifact_hash: str
    tokenizer_artifact_hash: str
    closure_id: str


def build_revision_closure(
    model_artifact: Path,
    tokenizer_artifact: Path,
    renderer: Path,
    config: Path,
    lm_eval_version: str,
) -> RevisionClosure:
    if not lm_eval_version.strip():
        raise ValueError("lm_eval_version must be non-empty")
    model = _load_resolved_artifact(model_artifact, "model")
    tokenizer = _load_resolved_artifact(tokenizer_artifact, "tokenizer")
    core = {
        "model_repo": model["repo_id"],
        "model_commit": model["commit_hash"],
        "tokenizer_repo": tokenizer["repo_id"],
        "tokenizer_commit": tokenizer["commit_hash"],
        "renderer_hash": _sha256_file(renderer),
        "config_hash": _sha256_file(config),
        "lm_eval_version": lm_eval_version,
        "model_artifact_hash": _sha256_file(model_artifact),
        "tokenizer_artifact_hash": _sha256_file(tokenizer_artifact),
    }
    closure_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RevisionClosure(closure_id=closure_id, **core)


def closure_dict(closure: RevisionClosure) -> Dict[str, Any]:
    return asdict(closure)


def _load_resolved_artifact(path: Path, kind: str) -> Dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid %s revision artifact: %s" % (kind, path)) from exc
    repo_id = payload.get("repo_id")
    commit = payload.get("commit_hash") or payload.get("_commit_hash")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("%s revision artifact is missing repo_id" % kind)
    if not isinstance(commit, str) or not _is_hex_commit(commit):
        raise ValueError("%s revision artifact is missing a resolved commit hash" % kind)
    return {"repo_id": repo_id, "commit_hash": commit.lower()}


def _is_hex_commit(value: str) -> bool:
    return len(value) in {40, 64} and all(ch in "0123456789abcdef" for ch in value.lower())


def _sha256_file(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError("required revision-closure input is unreadable: %s" % path) from exc
    return hashlib.sha256(content).hexdigest()
