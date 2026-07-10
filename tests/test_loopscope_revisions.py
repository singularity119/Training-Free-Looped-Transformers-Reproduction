import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tflt import audit, eval_runner
from tflt.eval_runner import _write_model_revision
from tflt.loopscope import probe, window_probe
from tflt.loopscope.revisions import (
    RevisionClosureError,
    load_tokenizer_with_resolved_commit,
    resolve_tokenizer_commit,
    strict_revision_closure,
)


COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


class _Config:
    _commit_hash = COMMIT
    architectures = ["FixtureModel"]
    model_type = "fixture"


class _Model:
    config = _Config()


class _ModelWithoutCommit:
    class config:
        pass


class _Tokenizer4513Shape:
    """Real tokenizer shape: the loader consumed, rather than retained, _commit_hash."""

    def __init__(self):
        self.init_kwargs = {"revision": COMMIT}


def _snapshot_path(commit, filename):
    return "/cache/models--org--model/snapshots/%s/%s" % (commit, filename)


class LoopScopeRevisionClosureTest(unittest.TestCase):
    def test_resolved_cache_path_closes_4513_shaped_tokenizer(self):
        auto_tokenizer = Mock()
        tokenizer = _Tokenizer4513Shape()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        resolver_calls = []

        def resolver(repo_id, filename, **kwargs):
            resolver_calls.append((repo_id, filename, kwargs))
            if filename in ("tokenizer_config.json", "tokenizer.json"):
                return _snapshot_path(COMMIT, filename)
            return None

        loaded, tokenizer_commit = load_tokenizer_with_resolved_commit(
            auto_tokenizer,
            "org/model",
            {
                "trust_remote_code": True,
                "revision": COMMIT,
                "cache_dir": "/cache",
                "local_files_only": True,
            },
            resolver=resolver,
        )

        self.assertIs(loaded, tokenizer)
        self.assertEqual(tokenizer_commit, COMMIT)
        self.assertEqual(
            strict_revision_closure(_Model(), tokenizer_commit, COMMIT)["tokenizer_commit"],
            COMMIT,
        )
        self.assertTrue(resolver_calls)
        self.assertTrue(all(call[2]["revision"] == COMMIT for call in resolver_calls))
        self.assertTrue(all(call[2]["cache_dir"] == "/cache" for call in resolver_calls))
        self.assertTrue(all(call[2]["local_files_only"] for call in resolver_calls))
        auto_tokenizer.from_pretrained.assert_called_once_with(
            "org/model",
            trust_remote_code=True,
            revision=COMMIT,
            cache_dir="/cache",
            local_files_only=True,
        )

    def test_non_phase_load_without_revision_remains_compatible(self):
        auto_tokenizer = Mock()
        tokenizer = _Tokenizer4513Shape()
        tokenizer.init_kwargs = {}
        auto_tokenizer.from_pretrained.return_value = tokenizer
        resolver = Mock(side_effect=AssertionError("resolver must not be called"))

        loaded, tokenizer_commit = load_tokenizer_with_resolved_commit(
            auto_tokenizer,
            "org/model",
            {"trust_remote_code": True},
            resolver=resolver,
        )

        self.assertIs(loaded, tokenizer)
        self.assertIsNone(tokenizer_commit)
        resolver.assert_not_called()
        auto_tokenizer.from_pretrained.assert_called_once_with(
            "org/model", trust_remote_code=True
        )

    def test_resolver_requires_exact_requested_revision(self):
        with self.assertRaisesRegex(RevisionClosureError, "requested tokenizer revision"):
            resolve_tokenizer_commit("org/model", {}, resolver=Mock())

    def test_requested_or_init_kwargs_revision_is_not_resolved_evidence(self):
        tokenizer = _Tokenizer4513Shape()
        self.assertEqual(tokenizer.init_kwargs["revision"], COMMIT)
        with self.assertRaisesRegex(RevisionClosureError, "resolved tokenizer revision"):
            strict_revision_closure(_Model(), None, COMMIT)

    def test_resolved_artifact_without_commit_is_rejected(self):
        def resolver(repo_id, filename, **kwargs):
            if filename == "tokenizer_config.json":
                return "/cache/models--org--model/tokenizer_config.json"
            return None

        with self.assertRaisesRegex(RevisionClosureError, "resolved tokenizer revision"):
            resolve_tokenizer_commit(
                "org/model", {"revision": COMMIT}, resolver=resolver
            )

    def test_resolved_artifact_manifest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "tokenizer revision"):
            strict_revision_closure(_Model(), OTHER_COMMIT, COMMIT)

    def test_conflicting_resolved_artifact_commits_are_rejected(self):
        def resolver(repo_id, filename, **kwargs):
            if filename == "tokenizer_config.json":
                return _snapshot_path(COMMIT, filename)
            if filename == "tokenizer.json":
                return _snapshot_path(OTHER_COMMIT, filename)
            return None

        with self.assertRaisesRegex(RevisionClosureError, "conflicting tokenizer artifact"):
            resolve_tokenizer_commit(
                "org/model", {"revision": COMMIT}, resolver=resolver
            )

    def test_missing_resolved_model_revision_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "resolved model revision"):
            strict_revision_closure(_ModelWithoutCommit(), COMMIT, COMMIT)

    def test_model_tokenizer_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "tokenizer revision"):
            strict_revision_closure(_Model(), OTHER_COMMIT, COMMIT)

    def test_manifest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "model revision"):
            strict_revision_closure(_Model(), COMMIT, OTHER_COMMIT)

    def test_all_phase_one_loaders_use_shared_authoritative_helper(self):
        loaders = (
            probe.run_layer_probe,
            window_probe.run_window_probe,
            audit.run_loop_effect_audit,
            eval_runner.run_lm_eval,
        )
        for loader in loaders:
            with self.subTest(loader=loader.__module__):
                source = inspect.getsource(loader)
                self.assertIn("load_tokenizer_with_resolved_commit", source)
                self.assertNotIn("AutoTokenizer.from_pretrained", source)

    def test_eval_artifact_records_three_party_revision_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            _write_model_revision(
                output,
                _Model(),
                _Tokenizer4513Shape(),
                "org/model",
                COMMIT,
                resolved_tokenizer_revision=COMMIT,
            )
            payload = json.loads(
                (output / "model_revision.json").read_text(encoding="utf-8")
            )
        self.assertEqual(payload["model_commit"], COMMIT)
        self.assertEqual(payload["tokenizer_commit"], COMMIT)
        self.assertEqual(payload["manifest_commit"], COMMIT)
        self.assertTrue(payload["match"])


if __name__ == "__main__":
    unittest.main()
