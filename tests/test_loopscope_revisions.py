import tempfile
import unittest
from pathlib import Path

from tflt.eval_runner import _write_model_revision
from tflt.loopscope.revisions import RevisionClosureError, strict_revision_closure


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


class _Tokenizer:
    def __init__(self, commit=COMMIT):
        self.init_kwargs = {"_commit_hash": commit, "revision": "untrusted-request-value"}


class LoopScopeRevisionClosureTest(unittest.TestCase):
    def test_missing_resolved_model_revision_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "resolved model revision"):
            strict_revision_closure(_ModelWithoutCommit(), _Tokenizer(), COMMIT)

    def test_missing_resolved_tokenizer_revision_is_rejected(self):
        tokenizer = _Tokenizer()
        tokenizer.init_kwargs = {"revision": COMMIT}
        with self.assertRaisesRegex(RevisionClosureError, "resolved tokenizer revision"):
            strict_revision_closure(_Model(), tokenizer, COMMIT)

    def test_model_tokenizer_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "tokenizer revision"):
            strict_revision_closure(_Model(), _Tokenizer(OTHER_COMMIT), COMMIT)

    def test_manifest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RevisionClosureError, "model revision"):
            strict_revision_closure(_Model(), _Tokenizer(), OTHER_COMMIT)

    def test_eval_artifact_records_three_party_revision_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            _write_model_revision(output, _Model(), _Tokenizer(), "org/model", COMMIT)
            import json

            payload = json.loads((output / "model_revision.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["model_commit"], COMMIT)
        self.assertEqual(payload["tokenizer_commit"], COMMIT)
        self.assertEqual(payload["manifest_commit"], COMMIT)
        self.assertTrue(payload["match"])


if __name__ == "__main__":
    unittest.main()
