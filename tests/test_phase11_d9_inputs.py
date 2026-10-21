import csv
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import urllib.error
import zipfile

from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_data import GPQA_COLUMNS, GPQA_SEED, prepare_gpqa_inputs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/prepare_phase11_d9_inputs.py"
SPEC = importlib.util.spec_from_file_location("d9_inputs", SCRIPT)
d9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d9)


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, **kwargs):
        assert kwargs in ({"add_generation_prompt": True},
                          {"add_generation_prompt": True, "enable_thinking": True})
        assert [m["role"] for m in messages] == ["user"]
        self.content = messages[0]["content"]
        return [10, 11] if tokenize else "new thinking prompt"

    def encode(self, text, add_special_tokens=False):
        return [10, 11]

    def convert_tokens_to_ids(self, text):
        return {"<think>": 1, "</think>": 2}[text]


def archive_fixture(count=448, member=d9.GPQA_ARCHIVE_MEMBER, columns=GPQA_COLUMNS):
    raw = [dict(zip(GPQA_COLUMNS,
        [f"synthetic question {i}", *[f"synthetic {i} option {j}" for j in range(4)]]))
        for i in range(count)]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[*columns, "Explanation"])
    writer.writeheader()
    for row in raw:
        writer.writerow({**{k: row[k] for k in columns}, "Explanation": "not a model input"})
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(member, stream.getvalue())
    return payload.getvalue(), raw


class D9InputsTests(unittest.TestCase):
    def test_prepublished_source_cli_does_not_download_archive_again(self):
        payload, _ = archive_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "source.csv"
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_path.write_bytes(archive.read(d9.GPQA_ARCHIVE_MEMBER))
            output = root / "prepared"
            argv = [str(SCRIPT), "--model-dir", "/admitted-model", "--source-csv", str(csv_path),
                    "--output-dir", str(output)]
            tokenizer_factory = SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=tokenizer_factory)}), \
                 patch.object(sys, "argv", argv), \
                 patch.object(d9, "fetch_author_rows", side_effect=AssertionError("unexpected archive fetch")), \
                 redirect_stdout(io.StringIO()):
                d9.main()
            receipt = json.loads((output / "alignment.json").read_text())
            self.assertEqual(receipt["acquisition"], "prepublished_private_csv")
            self.assertEqual(receipt["count"], 448)

    @unittest.skipUnless(shutil.which("zip"), "encrypted synthetic ZIP creation needs the system zip utility")
    def test_public_distribution_password_opens_encrypted_synthetic_archive(self):
        payload, expected = archive_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(io.BytesIO(payload)) as plain:
                member = root / d9.GPQA_ARCHIVE_MEMBER
                member.parent.mkdir()
                member.write_bytes(plain.read(d9.GPQA_ARCHIVE_MEMBER))
            encrypted = root / "synthetic.zip"
            subprocess.run([shutil.which("zip"), "-q", "-P", d9.GPQA_ARCHIVE_PASSWORD.decode(),
                            str(encrypted), d9.GPQA_ARCHIVE_MEMBER], cwd=root, check=True)
            with zipfile.ZipFile(encrypted) as archive:
                self.assertTrue(archive.getinfo(d9.GPQA_ARCHIVE_MEMBER).flag_bits & 1)
                with self.assertRaises(RuntimeError):
                    archive.read(d9.GPQA_ARCHIVE_MEMBER)
            self.assertEqual(d9.project_archive(encrypted.read_bytes()), expected)

    def test_original_archive_projection_reuses_producer_and_sealed_gold(self):
        payload, raw = archive_fixture()
        projected = d9.project_archive(payload)
        self.assertEqual(projected, raw)
        tokenizer = Tokenizer()
        with patch.object(d9, "prepare_gpqa_inputs", wraps=prepare_gpqa_inputs) as producer:
            pool, gold, receipt = d9.build_inputs(projected, tokenizer)
            producer.assert_called_once()
        rng = random.Random(GPQA_SEED)
        for i, (row, sealed) in enumerate(zip(pool["rows"], gold)):
            permutation = list(range(4))
            rng.shuffle(permutation)
            self.assertEqual(row["choices"], [raw[i][GPQA_COLUMNS[1:][j]] for j in permutation])
            self.assertEqual(sealed["option_source_indices"], permutation)
            self.assertEqual(row["source_index"], i)
            self.assertEqual(row["identity"],
                f"github:idavidrein/gpqa@{d9.GPQA_GITHUB_COMMIT}:dataset/gpqa_main.csv:{i}")
        self.assertTrue(all(r["input_ids"] == [10, 11] and r["prompt"] == "new thinking prompt" for r in pool["rows"]))
        self.assertIn("Final answer: (X)", tokenizer.content)
        self.assertEqual(receipt["status"], "SOURCE_CONTRACT_REUSED")
        self.assertFalse(receipt["hpc_artifact_readback_performed"])
        self.assertTrue(pool["d9_source_alignment"]["canonical_source_reused"])
        identities = pool["test_identities"]
        expected = [r["choices"].index(raw[i]["Correct Answer"]) for i, r in enumerate(pool["rows"])]
        self.assertEqual(gold_indices(gold, identities, {i: list("ABCD") for i in identities},
                                      {i: "gpqa_main" for i in identities}), expected)
        serialized = json.dumps(pool)
        for key in (*GPQA_COLUMNS[1:], "Explanation", "gold_letter", "option_source_indices"):
            self.assertNotIn(key, serialized)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            d9.write_outputs(root, pool, gold, receipt)
            self.assertEqual(json.loads((root / "alignment.json").read_text()), receipt)
            self.assertEqual(len((root / "sealed/gpqa_main-gold.jsonl").read_text().splitlines()), 448)
            self.assertNotIn("synthetic question", (root / "alignment.json").read_text())

    def test_archive_member_count_and_schema_fail_fast(self):
        payload, _ = archive_fixture(member="dataset/gpqa_diamond.csv")
        with self.assertRaises(KeyError):
            d9.project_archive(payload)
        payload, _ = archive_fixture(count=447)
        with self.assertRaisesRegex(ValueError, "448"):
            d9.project_archive(payload)
        payload, _ = archive_fixture(columns=GPQA_COLUMNS[:-1])
        with self.assertRaisesRegex(ValueError, "five original"):
            d9.project_archive(payload)

    def test_anonymous_fixed_url_and_fetch_failure_propagates(self):
        payload, raw = archive_fixture()
        with patch.object(d9.urllib.request, "build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value.read.return_value = payload
            self.assertEqual(d9.fetch_author_rows(), raw)
            build.return_value.open.assert_called_once_with(d9.GPQA_ARCHIVE_URL, timeout=60)
        error = urllib.error.HTTPError(d9.GPQA_ARCHIVE_URL, 404, "not found", {}, None)
        with patch.object(d9.urllib.request, "build_opener") as build:
            build.return_value.open.side_effect = error
            with self.assertRaises(urllib.error.HTTPError):
                d9.fetch_author_rows()

    def test_synthetic_uses_original_engineering_fixture_without_gold(self):
        with patch.object(d9, "fetch_author_rows", side_effect=AssertionError("dataset read")):
            pool, gold, receipt = d9.build_synthetic_inputs(Tokenizer())
        self.assertIsNone(gold)
        self.assertEqual(pool["scope"], "PREFLIGHT_ONLY")
        self.assertTrue(pool["engineering_synthetic"])
        self.assertEqual(len(pool["rows"]), 1)
        self.assertEqual(pool["rows"][0]["identity"], "phase11-d8-synthetic:gpqa_main:0")
        self.assertEqual(receipt["status"], "ENGINEERING_SYNTHETIC")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            d9.write_outputs(root, pool, gold, receipt)
            self.assertEqual(set(p.name for p in root.iterdir()), {"thinking-pool.json", "alignment.json"})


if __name__ == "__main__":
    unittest.main()
