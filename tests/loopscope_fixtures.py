"""Shared strict MMLU five-shot provenance fixtures for LoopScope tests."""

import hashlib

from tflt.loopscope.schema import canonical_json_bytes, manifest_sha256


FAKE_DATASET_REVISION = "a" * 40


class FakeRendererBackend:
    """Deterministic dependency-injection backend for renderer contract tests."""

    def __init__(self, record_count=8):
        self.record_count = int(record_count)

    def distribution_version(self):
        return "0.4.11"

    def resolve_task_names(self, task_group):
        if task_group != "mmlu":
            raise ValueError("fixture only supports mmlu")
        return ["mmlu_math"]

    def collect(
        self,
        task_names,
        dataset_revision,
        target_split,
        fewshot_split,
        seed,
        max_targets_per_task,
    ):
        del seed
        if list(task_names) != ["mmlu_math"]:
            raise ValueError("unexpected fixture task list")
        count = self.record_count
        if max_targets_per_task is not None:
            count = min(count, int(max_targets_per_task))
        source = "cais/mmlu@%s" % dataset_revision
        demos = [
            {
                "id": "math:dev:%d" % index,
                "doc_index": index,
                "source": "cais/mmlu",
                "split": fewshot_split,
                "subject": "math",
                "doc_sha256": ("%x" % (index + 1)) * 64,
                "rendered_sha256": ("%x" % (index + 6)) * 64,
                "gold_sha256": hashlib.sha256(
                    (" %s" % chr(ord("A") + index % 4)).encode("utf-8")
                ).hexdigest(),
            }
            for index in range(5)
        ]
        records = []
        for index in range(count):
            text = "".join(
                "Demo %d\nA. a\nB. b\nC. c\nD. d\nAnswer: %s\n\n"
                % (value, chr(ord("A") + value % 4))
                for value in range(5)
            ) + "Target %d\nA. a\nB. b\nC. c\nD. d\nAnswer:" % index
            record_id = "mmlu_math:%s:%d" % (target_split, index)
            records.append(
                {
                    "id": record_id,
                    "target_doc_id": record_id,
                    "target_doc_index": index,
                    "target_doc_sha256": hashlib.sha256(
                        ("target-doc-%d" % index).encode("utf-8")
                    ).hexdigest(),
                    "dataset_fingerprint": hashlib.sha256(
                        b"fixture-processed-target"
                    ).hexdigest(),
                    "task_group": "mmlu",
                    "task_name": "mmlu_math",
                    "num_fewshot": 5,
                    "source": source,
                    "split": target_split,
                    "subject": "math",
                    "text": text,
                    "render_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "uses_target_gold_labels": False,
                    "fewshot_answers_present": True,
                    "fewshot_sample_ids": [demo["id"] for demo in demos],
                    "demonstrations": [dict(demo) for demo in demos],
                }
            )
        return {
            "renderer_entrypoint": "lm_eval.api.task.ConfigurableTask.fewshot_context",
            "source": source,
            "records": records,
            "source_files": [
                {"path": "api/task.py", "sha256": hashlib.sha256(b"task.py").hexdigest()},
                {
                    "path": "tasks/mmlu/math.yaml",
                    "sha256": hashlib.sha256(b"math.yaml").hexdigest(),
                },
            ],
            "task_configs": [
                {
                    "task_name": "mmlu_math",
                    "dataset_path": "cais/mmlu",
                    "dataset_name": "math",
                    "fewshot_split": fewshot_split,
                    "doc_to_text": "fixture-template",
                    "doc_to_target": "fixture-target",
                    "fewshot_delimiter": "\\n\\n",
                    "target_delimiter": " ",
                    "config_sha256": hashlib.sha256(b"config").hexdigest(),
                    "config_file": "tasks/mmlu/math.yaml",
                    "config_file_sha256": hashlib.sha256(b"math.yaml").hexdigest(),
                }
            ],
            "dataset_tasks": [
                {
                    "task_name": "mmlu_math",
                    "dataset_path": "cais/mmlu",
                    "dataset_name": "math",
                    "revision": dataset_revision,
                    "raw_split_fingerprints": {
                        fewshot_split: "fixture-dev-fingerprint",
                        target_split: "fixture-target-fingerprint",
                    },
                    "processed_target_fingerprint": "fixture-processed-target",
                    "processed_fewshot_fingerprint": "fixture-processed-dev",
                }
            ],
        }


def contract_record(record_id, text, source="fixture", split="dev", subject="math"):
    prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    demos = [
        {
            "id": "%s-demo-%d" % (subject, index),
            "doc_index": index,
            "source": source,
            "split": "dev",
            "subject": subject,
            "doc_sha256": ("%x" % (index + 1)) * 64,
            "rendered_sha256": ("%x" % (index + 6)) * 64,
            "gold_sha256": hashlib.sha256(
                ("fixture-gold-%d" % index).encode("utf-8")
            ).hexdigest(),
        }
        for index in range(5)
    ]
    renderer = {
        "renderer_entrypoint": "lm_eval.tasks.mmlu",
        "lm_eval_version": "0.4.11",
        "renderer_source_sha256": "1" * 64,
        "source_files_sha256": "1" * 64,
        "template_sha256": "2" * 64,
        "task_configs_sha256": "2" * 64,
        "render_contract_sha256": "3" * 64,
        "dataset_revision": FAKE_DATASET_REVISION,
        "dataset_fingerprint_sha256": "8" * 64,
        "source_projection_sha256": "9" * 64,
        "render_sha256": prompt_hash,
        "renderer_manifest_sha256": "4" * 64,
    }
    return {
        "id": record_id,
        "text": text,
        "source": source,
        "split": split,
        "subject": subject,
        "prompt_sha256": prompt_hash,
        "task_group": "mmlu",
        "task_name": "mmlu_%s" % subject,
        "target_doc_id": "%s-target-%s" % (subject, record_id),
        "target_doc_index": 0,
        "target_doc_sha256": hashlib.sha256(
            ("fixture-target-%s" % record_id).encode("utf-8")
        ).hexdigest(),
        "dataset_fingerprint": hashlib.sha256(b"fixture-dataset").hexdigest(),
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": renderer,
        "fewshot_sample_ids": [demo["id"] for demo in demos],
        "demonstrations": demos,
    }


def rendering_entry(record):
    return {
        "id": record["id"],
        "target": {
            "source": record["source"],
            "split": record["split"],
            "subject": record["subject"],
        },
        "task_group": "mmlu",
        "task_name": record["task_name"],
        "target_doc_id": record["target_doc_id"],
        "target_doc_index": record["target_doc_index"],
        "target_doc_sha256": record["target_doc_sha256"],
        "dataset_fingerprint": record["dataset_fingerprint"],
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": dict(record["renderer"]),
        "fewshot_sample_ids": list(record["fewshot_sample_ids"]),
        "demonstrations": [dict(value) for value in record["demonstrations"]],
        "prompt_sha256": record["prompt_sha256"],
    }


def finalize_probe_pool(pool):
    records = []
    for item in pool["records"]:
        record = contract_record(
            str(item["id"]),
            "fixture prompt %s\nAnswer:" % item["id"],
            source=pool["source"],
            split=pool["split"],
        )
        record["prompt_sha256"] = str(item["prompt_sha256"])
        record["renderer"]["render_sha256"] = str(item["prompt_sha256"])
        records.append(record)
    rendering = [rendering_entry(record) for record in records]
    render_subset = hashlib.sha256(canonical_json_bytes(rendering)).hexdigest()
    pool.update(
        {
            "task_group": "mmlu",
            "num_fewshot": 5,
            "uses_target_gold_labels": False,
            "fewshot_answers_present": True,
            "renderer": dict(records[0]["renderer"]),
            "render_contract_sha256": "3" * 64,
            "rendering_records": rendering,
            "source_render_contract_subset_sha256": render_subset,
            "selected_render_contract_subset_sha256": render_subset,
        }
    )
    selected = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": pool["source"],
        "split": pool["split"],
        "count": pool["count"],
        "seed": pool["seed"],
        "sample_ids": pool["sample_ids"],
        "records": pool["records"],
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": pool["renderer"],
        "render_contract_sha256": pool["render_contract_sha256"],
        "rendering_records": rendering,
        "render_contract_subset_sha256": render_subset,
        "source_manifest_sha256": pool["source_manifest_sha256"],
    }
    pool["manifest_sha256"] = manifest_sha256(selected)
    pool["selected_subset_sha256"] = pool["manifest_sha256"]
    return pool


def source_pool_manifest(records, seed=7):
    rendering = [rendering_entry(record) for record in records]
    payload = {
        "schema_version": "loopscope.probe-pool-manifest.v1",
        "source": records[0]["source"],
        "split": records[0]["split"],
        "count": len(records),
        "seed": seed,
        "sample_ids": [record["id"] for record in records],
        "records": [
            {"id": record["id"], "prompt_sha256": record["prompt_sha256"]}
            for record in records
        ],
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": {
            "renderer_entrypoint": "lm_eval.tasks.mmlu",
            "lm_eval_version": "0.4.11",
            "renderer_source_sha256": "1" * 64,
            "source_files_sha256": "1" * 64,
            "template_sha256": "2" * 64,
            "task_configs_sha256": "2" * 64,
            "render_contract_sha256": "3" * 64,
            "dataset_revision": FAKE_DATASET_REVISION,
            "dataset_fingerprint_sha256": "8" * 64,
            "source_projection_sha256": "9" * 64,
        },
        "rendering_records": rendering,
        "render_contract_subset_sha256": hashlib.sha256(
            canonical_json_bytes(rendering)
        ).hexdigest(),
    }
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload
