"""Gold-free Phase11 task closure, followed by frozen exploratory analysis."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from .phase11_data import COUNTS
from .phase8_accuracy_stats import holm_adjust, paired_contrast

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261008
STRENGTHS = (0.1, 0.5, 0.9)
POLICIES = ("fixed_t0", "lag1", "current_t")
ONLINE_FAMILY = "ONLINE_VS_NATIVE_AND_LOOP_EXPLORATORY"
STRATEGY_FAMILY = "STRATEGY_EXPLORATORY"
BASE_FAMILY = "LOOP_VS_NATIVE_BASE"
FAMILY_COUNTS = {ONLINE_FAMILY: 30, STRATEGY_FAMILY: 12, BASE_FAMILY: 2}


def frozen_contrasts(panel):
    """Generate 44 contrasts from 18 independent cells, excluding aliases."""
    cells = panel["cells"]
    if len(cells) != 18 or len({c["cell_id"] for c in cells}) != 18:
        raise ValueError("task panel requires exactly 18 independent cells")
    lookup = {}
    for cell in cells:
        key = (cell["arm"], cell.get("k"), cell.get("direction_policy"), cell.get("strength"))
        if key in lookup:
            raise ValueError("duplicate scientific cell")
        lookup[key] = cell["cell_id"]
    native = lookup[("Native", None, None, 0.0)]
    loops = {k: lookup[("Loop", k, None, 0.0)] for k in (2, 3)}
    result = []
    for cell in cells:
        if cell["arm"] == "Online":
            for ref, comparison in ((native, "Online-Native"), (loops[cell["k"]], "Online-Loop")):
                result.append({"reference": ref, "treatment": cell["cell_id"], "family": ONLINE_FAMILY,
                               "comparison": comparison})
    for strength in STRENGTHS:
        fixed = lookup[("Online", 2, "fixed_t0", strength)]
        current = lookup[("Online", 2, "current_t", strength)]
        result.append({"reference": fixed, "treatment": current, "family": STRATEGY_FAMILY,
                       "comparison": "K2-current_t-fixed_t0", "strength": strength})
        for reference, treatment in combinations(POLICIES, 2):
            result.append({"reference": lookup[("Online", 3, reference, strength)],
                           "treatment": lookup[("Online", 3, treatment, strength)],
                           "family": STRATEGY_FAMILY, "comparison": f"K3-{treatment}-{reference}",
                           "strength": strength})
    for k in (2, 3):
        result.append({"reference": native, "treatment": loops[k], "family": BASE_FAMILY,
                       "comparison": f"LoopK{k}-Native"})
    if Counter(r["family"] for r in result) != Counter(FAMILY_COUNTS):
        raise ValueError("frozen contrast families differ from 30/12/2")
    return result


def align_records(dataset, records, identities):
    """Small pure helper for exact identity and canonical-index closure."""
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("canonical identities must be nonempty and unique")
    expected = {identity: index for index, identity in enumerate(identities)}
    result = {}
    for record in records:
        identity = record["identity"]
        if identity not in expected or identity in result:
            raise ValueError("raw records contain unknown or duplicate identity")
        if record["canonical_index"] != expected[identity] or record["status"] != "OK":
            raise ValueError("raw record has wrong canonical index or runtime failure")
        result[identity] = record
    if len(result) != len(identities):
        raise ValueError("raw records do not cover the complete canonical population")
    return [result[identity] for identity in identities]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _attempt_records(panel, pool, directory, scope, checked_manifests):
    from .phase11_accuracy import engineering_cell, validate_record
    root = Path(directory)
    summary = _read_json(root / "summary.json")
    command_args = _read_json(root / "command_args.json")
    metadata = summary["metadata"]
    dataset, rows = panel["dataset"], pool["rows"]
    cell = next((c for c in panel["cells"] if c == metadata["cell"]), None)
    if summary["status"] != "RAW_COMPLETE" or summary["target_gold_loaded"] is not False:
        raise ValueError("cell root is incomplete or already gold-bearing")
    if metadata["dataset"] != dataset or cell is None or metadata["scope"] != scope:
        raise ValueError("root dataset/scientific cell/scope differs from frozen manifest")
    if metadata["acquisition_cell"] != engineering_cell(cell, metadata["engineering_check"], scope):
        raise ValueError("attempt acquisition cell differs from its declared engineering check")
    if metadata["pool"] != panel["pool"]:
        raise ValueError("raw roots do not share the exact frozen input path")
    manifest_path = metadata["manifest"]
    if manifest_path not in checked_manifests:
        if _read_json(manifest_path) != panel:
            raise ValueError("raw root manifest differs from frozen task panel")
        checked_manifests.add(manifest_path)
    indices = metadata["indices"]
    if not indices or indices != sorted(set(indices)) or any(type(i) is not int or i < 0 or i >= len(rows) for i in indices):
        raise ValueError("attempt indices are invalid")
    records = _read_records(root / "records.jsonl")
    if sorted(r["canonical_index"] for r in records) != indices:
        raise ValueError("attempt records differ from declared exact indices")
    for record in records:
        validate_record(record, dataset)
        row = rows[record["canonical_index"]]
        if (record["identity"] != row["identity"] or record["candidate_labels"] != row["choice_labels"]
                or record["category"] != row["subject"]):
            raise ValueError("raw identity/candidate order/category differs from frozen input")
        if dataset == "arc_challenge" and record["choice_text_lengths"] != [len(choice) for choice in row["choices"]]:
            raise ValueError("ARC normalization lengths differ from complete original choices")
        if scope == "FORMAL_TEST" and dataset != "arc_challenge" and record["max_new_tokens"] != 2048:
            raise ValueError("formal generation length bound differs from frozen 2048")
    if "cell" in command_args and command_args["cell"] != cell:
        raise ValueError("command_args cell differs from raw summary")
    commit = metadata["source_commit"]
    if not isinstance(commit, str) or not commit:
        raise ValueError("raw root lacks source commit")
    facts = {"run_root": str(root.resolve()), "indices": indices, "records_count": len(records),
             "source_commit": commit, "cell_id": cell["cell_id"], "engineering_check": metadata["engineering_check"]}
    return facts, records


def _source_commit_metadata(commits, approved_source_commits):
    """Accept mixed revisions only under an explicit, externally audited set."""
    observed = sorted(commits)
    if approved_source_commits is None:
        if len(observed) != 1:
            raise ValueError("panel raw roots do not share one source commit")
        return {"source_commit": observed[0]}
    if isinstance(approved_source_commits, str):
        raise ValueError("approved source commits must be a nonempty collection")
    approved = set(approved_source_commits)
    if not approved or any(not isinstance(commit, str) or not commit for commit in approved):
        raise ValueError("approved source commits must be nonempty strings")
    if not set(observed).issubset(approved):
        raise ValueError("raw root source commit is outside the approved source set")
    return {"source_commit": observed[0] if len(observed) == 1 else None,
            "source_commits": observed, "approved_source_commits": sorted(approved)}


def verify_attempt(panel, pool, root, scope="PREFLIGHT_ONLY", *, approved_source_commits=None):
    """Verify one synthetic producer attempt without declaring task closure."""
    from .phase11_accuracy import load_pool
    from .phase11_panel import validate_score_manifest
    if scope != "PREFLIGHT_ONLY":
        raise ValueError("single-attempt verification is restricted to synthetic preflight")
    validate_score_manifest(panel, scope=scope)
    if load_pool(Path(panel["pool"]), panel["dataset"], scope) != pool:
        raise ValueError("attempt pool differs from the manifest's exact input file")
    facts, records = _attempt_records(panel, pool, root, scope, set())
    sources = _source_commit_metadata({facts["source_commit"]}, approved_source_commits)
    return {"schema_version": "loopscope.phase11.attempt_verification.v1", "status": "ATTEMPT_RAW_VERIFIED",
            "scope": scope, "dataset": panel["dataset"], "target_gold_loaded": False, **facts, **sources,
            "identities": [r["identity"] for r in sorted(records, key=lambda r: r["canonical_index"])]}


def close_panel(panel, pool, cell_roots, *, approved_source_commits=None):
    """Re-read all 18 raw cells/shards before any gold file is opened."""
    from .phase11_accuracy import load_pool, validate_cell_records
    from .phase11_panel import validate_score_manifest
    dataset = panel["dataset"]
    if dataset not in COUNTS:
        raise ValueError("unknown Phase11 dataset")
    frozen_contrasts(panel)
    validate_score_manifest(panel, scope="FORMAL_TEST")
    pool_rows = pool["rows"]
    identities = [row["identity"] for row in pool_rows]
    if len(identities) != COUNTS[dataset] or len(set(identities)) != COUNTS[dataset]:
        raise ValueError("formal pool differs from the full frozen identity population")
    if pool["task"] != dataset or set(cell_roots) != {cell["cell_id"] for cell in panel["cells"]}:
        raise ValueError("closure roots differ from task's 18 independent cells")
    aligned, paths, commits, checked_manifests = {}, {}, set(), set()
    pool_path = panel["pool"]
    if not pool_path or load_pool(Path(pool_path), dataset, "FORMAL_TEST") != pool:
        raise ValueError("formal pool differs from the manifest's exact input file")
    for cell in panel["cells"]:
        cell_id = cell["cell_id"]
        roots = cell_roots[cell_id]
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if not roots:
            raise ValueError("cell has no raw run roots")
        records, seen_indices, root_facts = [], set(), []
        for directory in roots:
            facts, shard_records = _attempt_records(panel, pool, directory, "FORMAL_TEST", checked_manifests)
            if facts["cell_id"] != cell_id:
                raise ValueError("raw attempt belongs to a different panel cell")
            indices = facts["indices"]
            if seen_indices.intersection(indices):
                raise ValueError("cell shards overlap")
            commits.add(facts["source_commit"])
            seen_indices.update(indices)
            records.extend(shard_records)
            root_facts.append(facts)
        ordered = align_records(dataset, records, identities)
        aligned[cell_id] = validate_cell_records(dataset, ordered, identities)
        if any(record["candidate_labels"] != row["choice_labels"] or record["category"] != row["subject"]
               for record, row in zip(aligned[cell_id], pool_rows)):
            raise ValueError("raw candidate order/category differs from the frozen input pool")
        paths[cell_id] = root_facts
    sources = _source_commit_metadata(commits, approved_source_commits)
    closure = {"schema_version": "loopscope.phase11.panel_closure.v1", "status": "PANEL_CLOSED_GOLD_UNREAD",
               "dataset": dataset, "target_gold_loaded": False, **sources,
               "independent_cell_count": 18, "sample_count": len(identities), "identities": identities,
               "cells": panel["cells"], "raw_roots": paths,
               "record_counts": {cell: len(rows) for cell, rows in aligned.items()}}
    return closure, aligned


def gold_indices(gold_rows, identities, labels_by_identity, categories_by_identity):
    """Exact identity join; standard label_index or sealed GPQA gold_letter."""
    if isinstance(gold_rows, dict):
        gold_rows = gold_rows["rows"]
    expected = set(identities)
    gold = {}
    for row in gold_rows:
        identity = row["identity"]
        if identity not in expected or identity in gold:
            raise ValueError("gold contains unknown or duplicate identity")
        labels = labels_by_identity[identity]
        if "label_index" in row:
            value = row["label_index"]
        else:
            value = labels.index(row["gold_letter"])
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(labels):
            raise ValueError("gold label index is outside actual candidate count")
        if "category" in row and row["category"] != categories_by_identity[identity]:
            raise ValueError("gold category differs from canonical model input")
        gold[identity] = value
    if len(gold) != len(identities):
        raise ValueError("gold identity join is incomplete")
    return [gold[identity] for identity in identities]


def generation_prediction(record):
    """Independently enforce exactly one legal standalone answer line."""
    labels = record["candidate_labels"]
    matches = []
    for line in record["generated_text"].splitlines():
        match = re.fullmatch(r"\s*Final answer: \(([A-Z])\)\s*", line)
        if match and match[1] in labels:
            matches.append(labels.index(match[1]))
    value = matches[0] if len(matches) == 1 else None
    if record["prediction_index"] != value:
        raise ValueError("stored prediction differs from strict raw-text extraction")
    return value


def score_cell(dataset, records, gold):
    if not len(records) == len(gold) or not records:
        raise ValueError("scoring requires complete aligned records and gold")
    correct, raw_correct, failures, truncated, lengths = [], [], 0, 0, []
    for record, label in zip(records, gold):
        if record["status"] != "OK":
            raise ValueError("runtime failure cannot be scored as a model answer")
        if dataset == "arc_challenge":
            scores = [float(s) for s in record["scores"]]
            text_lengths = record["choice_text_lengths"]
            if len(scores) != len(text_lengths) or not scores or any(n <= 0 for n in text_lengths) or not all(math.isfinite(s) for s in scores):
                raise ValueError("ARC scoring needs all finite full-text candidate scores")
            raw_prediction = max(range(len(scores)), key=scores.__getitem__)
            prediction = max(range(len(scores)), key=lambda i: scores[i] / text_lengths[i])
            raw_correct.append(int(raw_prediction == label))
        else:
            prediction = generation_prediction(record)
            failures += prediction is None
            truncated += bool(record["truncated"])
            lengths.append(record["generated_token_count"])
        correct.append(int(prediction is not None and prediction == label))
    n = len(records)
    facts = {"n": n, "correct": sum(correct), "accuracy": sum(correct) / n,
             "metric": "acc_norm" if dataset == "arc_challenge" else "accuracy"}
    if dataset == "arc_challenge":
        facts.update(acc_norm=facts["accuracy"], acc=sum(raw_correct) / n)
    else:
        facts.update(extraction_failure_count=failures, extraction_failure_rate=failures / n,
                     truncated_count=truncated, truncated_rate=truncated / n,
                     generation_length={"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / n})
    return facts, correct


def analyze_records(panel, aligned_records, gold_rows):
    """Pure analysis helper; the production entry point closes before calling."""
    specs = frozen_contrasts(panel)
    dataset = panel["dataset"]
    first = aligned_records[panel["cells"][0]["cell_id"]]
    identities = [record["identity"] for record in first]
    labels = {r["identity"]: r["candidate_labels"] for r in first}
    categories = {r["identity"]: r["category"] for r in first}
    gold = gold_indices(gold_rows, identities, labels, categories)
    subjects = [categories[i] for i in identities] if dataset == "mmlu_pro" else ["all_items"] * len(identities)
    cell_results, correctness, stats_by_cell = [], {}, {}
    for cell in panel["cells"]:
        records = align_records(dataset, aligned_records[cell["cell_id"]], identities)
        if any(r["candidate_labels"] != labels[r["identity"]] or r["category"] != categories[r["identity"]] for r in records):
            raise ValueError("cell candidate order/category differs from common input")
        stats, correctness[cell["cell_id"]] = score_cell(dataset, records, gold)
        stats_by_cell[cell["cell_id"]] = stats
        cell_results.append({**cell, **stats})
    contrasts = [{**spec, **paired_contrast(correctness[spec["reference"]], correctness[spec["treatment"]], subjects,
                    bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)} for spec in specs]
    for family, count in FAMILY_COUNTS.items():
        selected = [row for row in contrasts if row["family"] == family]
        adjusted = holm_adjust([row["mcnemar_exact_p"] for row in selected])
        for row, pvalue in zip(selected, adjusted):
            row.update(holm_adjusted_p=pvalue, holm_family_size=count, holm_reject_alpha_0_05=pvalue <= 0.05)
    return {"schema_version": "loopscope.phase11.analysis.v1", "status": "TASK_ANALYZED", "dataset": dataset,
            "sample_count": len(identities), "dataset_recipe": dict(panel["dataset_recipe"]) |
                ({"hf_content_equivalence_verified": False} if dataset == "gpqa_main" else {}),
            "cells": cell_results, "contrasts": contrasts,
            "display_rows": [{**row, **stats_by_cell[row["result_cell_id"]]} for row in panel["display_rows"]],
            "independent_cell_count": 18, "alias_count": 3,
            "families": FAMILY_COUNTS, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "interpretation": {"panel": "exploratory full grid including negative results",
                "ci": "nominal 95 percent; Holm determines family significance",
                "limits": "No matched-norm control; no direction-specific attribution or transfer/lambda-selection claim",
                "current_t": "changes both intervention onset and direction timing"}}


def analyze_closed_panel(panel, pool, cell_roots, gold_path, *, approved_source_commits=None):
    closure, records = close_panel(panel, pool, cell_roots, approved_source_commits=approved_source_commits)
    # This is the first gold read, after a fresh reread/closure of all raw files.
    path = Path(gold_path)
    gold = _read_records(path) if path.suffix == ".jsonl" else _read_json(path)
    report = analyze_records(panel, records, gold)
    report["closure"] = closure
    report["target_gold_loaded"] = True
    return report
