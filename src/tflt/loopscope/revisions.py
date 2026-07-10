"""Fail-closed Hugging Face model/tokenizer revision provenance."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


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


TOKENIZER_REVISION_ARTIFACTS: Tuple[str, ...] = (
    "tokenizer_config.json",
    "tokenizer.json",
)

_TOKENIZER_RESOLVER_KWARGS = (
    "cache_dir",
    "force_download",
    "proxies",
    "resume_download",
    "local_files_only",
    "token",
    "subfolder",
)


def resolve_tokenizer_commit(
    repo_id: str,
    load_kwargs: Mapping[str, Any],
    *,
    artifacts: Sequence[str] = TOKENIZER_REVISION_ARTIFACTS,
    resolver: Optional[Callable[..., Optional[str]]] = None,
    extractor: Optional[Callable[[Optional[str], Optional[str]], Optional[str]]] = None,
) -> str:
    """Resolve the tokenizer commit from Hugging Face snapshot artifact paths.

    The requested revision selects artifacts but is never accepted as evidence of
    what the Hub resolver actually returned.
    """

    requested = require_exact_commit(
        load_kwargs.get("revision"), "requested tokenizer revision"
    )
    if resolver is None or extractor is None:
        try:
            from transformers.utils.hub import cached_file, extract_commit_hash
        except Exception as exc:  # pragma: no cover - remote dependency path.
            raise RevisionClosureError(
                "transformers Hub resolver is required for tokenizer revision closure"
            ) from exc
        if resolver is None:
            resolver = cached_file
        if extractor is None:
            extractor = extract_commit_hash

    resolver_kwargs = {
        key: load_kwargs[key]
        for key in _TOKENIZER_RESOLVER_KWARGS
        if key in load_kwargs
    }
    resolver_kwargs["revision"] = requested
    resolver_kwargs["_raise_exceptions_for_missing_entries"] = False

    commits = []
    for filename in artifacts:
        resolved_path = resolver(repo_id, filename, **resolver_kwargs)
        if resolved_path is None:
            continue
        commit = require_exact_commit(
            extractor(resolved_path, None),
            "resolved tokenizer revision from %s" % filename,
        )
        commits.append(commit)

    unique = sorted(set(commits))
    if not unique:
        raise RevisionClosureError(
            "resolved tokenizer revision is missing from tokenizer artifact paths"
        )
    if len(unique) != 1:
        raise RevisionClosureError(
            "conflicting tokenizer artifact revisions: %s" % ", ".join(unique)
        )
    return unique[0]


def load_tokenizer_with_resolved_commit(
    auto_tokenizer: Any,
    repo_id: str,
    load_kwargs: Mapping[str, Any],
    *,
    resolver: Optional[Callable[..., Optional[str]]] = None,
    extractor: Optional[Callable[[Optional[str], Optional[str]], Optional[str]]] = None,
) -> Tuple[Any, Optional[str]]:
    """Load a tokenizer and return authoritative snapshot provenance when pinned."""

    kwargs = dict(load_kwargs)
    if kwargs.get("revision"):
        commit = resolve_tokenizer_commit(
            repo_id,
            kwargs,
            resolver=resolver,
            extractor=extractor,
        )
        tokenizer = auto_tokenizer.from_pretrained(repo_id, **kwargs)
        return tokenizer, commit
    tokenizer = auto_tokenizer.from_pretrained(repo_id, **kwargs)
    return tokenizer, resolved_tokenizer_commit(tokenizer)


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
    model: Any, tokenizer_commit: Any, manifest_commit: Any
) -> Dict[str, Any]:
    """Prove that both loaded objects resolved to the frozen manifest commit."""

    expected = require_exact_commit(manifest_commit, "manifest revision")
    model_commit = require_exact_commit(
        resolved_model_commit(model), "resolved model revision"
    )
    tokenizer_commit = require_exact_commit(tokenizer_commit, "resolved tokenizer revision")
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
