import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.phase11_accuracy import arc_record, generation_record
from tflt.loopscope.phase11_panel import COUNTS, DATASETS, MODEL, REVISION, load_config, score_manifest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/phase11_canary.py"
SPEC = importlib.util.spec_from_file_location("phase11_canary", SCRIPT)
canary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canary)


def formal_pool(dataset):
    recipe = load_config()["datasets"][dataset]
    rows = []
    for index in range(COUNTS[dataset]):
        length = 3 + index % 17
        row = {"identity": f"fixture:{dataset}:{index}", "task": dataset, "split": recipe["split"],
               "subject": "physics", "choice_labels": ["A", "B"], "choices": ["one", "two"],
               "prompt": "fixture prompt", "fewshot_sample_ids": ["fixture-demo"] * recipe["num_fewshot"]}
        if dataset == "arc_challenge":
            row["tokenization"] = {MODEL: {"candidates": [{"input_length": length}, {"input_length": length + 1}]}}
        else:
            row.update(input_ids=[11] * length, prompt_token_length=length)
        rows.append(row)
    pool = {"schema_version": "loopscope.phase11.inputs.v1", "task": dataset, "model": MODEL,
            "model_revision": REVISION, "tokenizer_revision": REVISION, "target_gold_loaded": False,
            "scope": "FORMAL_TEST", "engineering_synthetic": False, "dataset_revision": recipe["revision"],
            "split": recipe["split"], "rows": rows, "test_identities": [row["identity"] for row in rows]}
    if dataset == "gpqa_main":
        pool.update({key: recipe[key] for key in ("source_kind", "github_repo", "github_commit", "archive_member")})
    return pool


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_attempt(root, panel, manifest, pool, cell, sample):
    root.mkdir()
    index = sample["canonical_index"]
    row = pool["rows"][index]
    metadata = {"cell": cell, "acquisition_cell": cell, "engineering_check": None,
                "engineering_generate_steps": None, "dataset": panel["dataset"], "scope": "FORMAL_TEST",
                "indices": [index], "source_commit": "fixture-commit", "pool": panel["pool"],
                "manifest": str(manifest), "target_gold_loaded": False, "max_new_tokens": 2048,
                "shard": {"index": index, "count": COUNTS[panel["dataset"]], "rule": "canonical_index_modulo_count"}}
    write_json(root / "summary.json", {"status": "RAW_COMPLETE", "target_gold_loaded": False, "metadata": metadata,
               "count": 1, "elapsed_seconds": 2.0, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200})
    write_json(root / "command_args.json", metadata)
    write_json(root / "env.json", {"gpu": "fixture-gpu", "total_memory": 1000})
    item = {"identity": row["identity"], "canonical_index": index, "elapsed_seconds": 1.0}
    if panel["dataset"] == "arc_challenge":
        record = arc_record(row, [-1, -2], [{"continuation_length": 1}] * 2, index)
        item["max_input_tokens"] = sample["input_token_length"]
    else:
        record = generation_record(row, "Private fixture answer Final answer: (A)", [12, 99], [99], 2048, index)
        item.update(prefill_input_tokens=sample["input_token_length"], prefill_count=1, decode_count=1)
    (root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "telemetry.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")


class CanaryTests(unittest.TestCase):
    def test_length_quantiles_keep_original_indices_and_arc_candidate_max(self):
        for dataset in DATASETS:
            pool = formal_pool(dataset)
            samples = canary.select_members(dataset, pool["rows"])
            ordered = sorted(range(len(pool["rows"])), key=lambda i: (canary.input_length(dataset, pool["rows"][i]), i))
            expected = [ordered[(len(ordered) - 1) * numerator // 4] for numerator in (1, 2, 3, 4)]
            self.assertEqual([sample["canonical_index"] for sample in samples], expected)
            self.assertEqual(len({sample["identity"] for sample in samples}), 4)
            for sample in samples:
                index = sample["canonical_index"]
                self.assertEqual(sample["input_token_length"], 3 + index % 17 + int(dataset == "arc_challenge"))

    def test_membership_write_once_and_all_216_formal_singletons_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifests, panels, pools = {}, {}, {}
            for dataset in DATASETS:
                pool = formal_pool(dataset)
                pool_path, manifest = base / f"{dataset}-pool.json", base / f"{dataset}-manifest.json"
                write_json(pool_path, pool)
                panel = score_manifest(dataset, pool_path, "FORMAL_TEST")
                write_json(manifest, panel)
                manifests[dataset], panels[dataset], pools[dataset] = str(manifest), panel, pool
            membership = canary.build_membership(manifests)
            output = base / "membership.json"
            args = canary.parser().parse_args(["membership", *sum((["--manifest", f"{d}={p}"] for d, p in manifests.items()), []),
                                               "--out", str(output)])
            self.assertEqual(canary.main(args), 0)
            with self.assertRaises(FileExistsError):
                canary.main(args)
            self.assertEqual(canary.read_json(output), membership)
            jobs = []
            for dataset in DATASETS:
                for cell in panels[dataset]["cells"]:
                    for sample in membership["tasks"][dataset]["samples"]:
                        root = base / f"attempt-{len(jobs)}"
                        write_attempt(root, panels[dataset], manifests[dataset], pools[dataset], cell, sample)
                        jobs.append({"dataset": dataset, "cell_id": cell["cell_id"],
                                     "canonical_index": sample["canonical_index"], "run_root": str(root), "batch": 0})
            result = canary.verify_canary(membership, {"jobs": jobs})
            self.assertEqual(result["status"], "CANARY_CLOSED_GOLD_UNREAD")
            self.assertEqual(result["record_count"], 216)
            self.assertFalse(result["full_panel_closed"])
            self.assertFalse(result["target_gold_loaded"])
            serialized = json.dumps(result)
            for forbidden in ("Private fixture", "generated_text", "scores", "prediction", "extraction", "accuracy"):
                self.assertNotIn(forbidden, serialized)
            with self.assertRaisesRegex(ValueError, "216"):
                canary.verify_canary(membership, {"jobs": jobs[:-1]})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                canary.verify_canary(membership, {"jobs": [jobs[0]] + jobs[:-1]})
            wrong = copy.deepcopy(jobs)
            wrong[0]["canonical_index"] = (wrong[0]["canonical_index"] + 1) % COUNTS[wrong[0]["dataset"]]
            with self.assertRaisesRegex(ValueError, "unknown"):
                canary.verify_canary(membership, {"jobs": wrong})
            root = Path(jobs[0]["run_root"])
            summary = canary.read_json(root / "summary.json")
            summary["metadata"]["shard"]["count"] = 4
            write_json(root / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "index/N"):
                canary.verify_canary(membership, {"jobs": jobs})


if __name__ == "__main__":
    unittest.main()
