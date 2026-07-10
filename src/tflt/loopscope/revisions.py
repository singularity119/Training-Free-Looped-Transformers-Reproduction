"""Fail-closed Hugging Face model/tokenizer revision provenance."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional


class RevisionClosureError(ValueError):
    """Raised when an exact phase-one snapshot cannot be proven."""


def require_exact_commit(value: Any, context: str) -> str:
    commit = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise RevisionClosureError(
            "%s must be an exact lowercase 40-64 hex Hugging Face commit" % context
        )
    return commit


def resolved_model_commit(model: Any) -> Optional[str]:
    config = getattr(model, "config", None)
    for value in (
        getattr(config, "_commit_hash", None),
        getattr(model, "_commit_hash", None),
        getattr(model, "commit_hash", None),
    ):
        if value:
            return str(value)
    return None


def resolved_tokenizer_commit(tokenizer: Any) -> Optional[str]:
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    for value in (
        getattr(tokenizer, "_commit_hash", None),
        getattr(tokenizer, "commit_hash", None),
        init_kwargs.get("_commit_hash"),
    ):
        if value:
            return str(value)
    return None


def strict_revision_closure(
    model: Any, tokenizer: Any, manifest_commit: Any
) -> Dict[str, Any]:
    """Prove that both loaded objects resolved to the frozen manifest commit."""

    expected = require_exact_commit(manifest_commit, "manifest revision")
    model_commit = require_exact_commit(
        resolved_model_commit(model), "resolved model revision"
    )
    tokenizer_commit = require_exact_commit(
        resolved_tokenizer_commit(tokenizer), "resolved tokenizer revision"
    )
    if model_commit != expected:
        raise RevisionClosureError(
            "resolved model revision %s differs from manifest revision %s"
            % (model_commit, expected)
        )
    if tokenizer_commit != expected:
        raise RevisionClosureError(
            "resolved tokenizer revision %s differs from manifest revision %s"
            % (tokenizer_commit, expected)
        )
    return {
        "schema_version": "loopscope.revision-closure.v1",
        "manifest_commit": expected,
        "model_commit": model_commit,
        "tokenizer_commit": tokenizer_commit,
        "match": True,
    }
