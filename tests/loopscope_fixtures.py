"""Shared strict MMLU five-shot provenance fixtures for LoopScope tests."""

import hashlib

from tflt.loopscope.schema import canonical_json_bytes, manifest_sha256


def contract_record(record_id, text, source="fixture", split="dev", subject="math"):
    prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    demos = [
        {
            "id": "%s-demo-%d" % (subject, index),
            "source": source,
            "split": "dev",
            "subject": subject,
            "doc_sha256": ("%x" % (index + 1)) * 64,
            "rendered_sha256": ("%x" % (index + 6)) * 64,
        }
        for index in range(5)
    ]
    renderer = {
        "renderer_entrypoint": "lm_eval.tasks.mmlu",
        "lm_eval_version": "0.4.11",
        "renderer_source_sha256": "1" * 64,
        "template_sha256": "2" * 64,
        "render_contract_sha256": "3" * 64,
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
            "template_sha256": "2" * 64,
            "render_contract_sha256": "3" * 64,
        },
        "rendering_records": rendering,
        "render_contract_subset_sha256": hashlib.sha256(
            canonical_json_bytes(rendering)
        ).hexdigest(),
    }
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload
