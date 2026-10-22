"""Portable D13 final-choice review workflow. No model, network or GT on prepare/seal.

Semantic decisions are supplied by blinded reviewers, never invented by this module.
The terminal matcher is the original D13 candidate grammar, not a semantic judge.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re

from .phase8_accuracy_stats import exact_mcnemar_p, holm_adjust

RULE = "D13-final-choice-v1.0"
N = 448
SEED = 20261022
REPLICATES = 20000


def read_rows(path):
    # File iteration preserves U+2028 inside model output strings.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def unique(rows, key):
    result = {}
    for row in rows:
        item = key(row)
        if item in result:
            raise ValueError(f"duplicate key: {item}")
        result[item] = row
    return result


def terminal_candidate(text):
    """Original D13 safe surface candidates; reviewer must check contextual conflicts."""
    start = text.find("</think>") + len("</think>") if "</think>" in text else 0
    region = text[start:]
    rx = re.compile(
        r"(?im)^[ \t*#>]*final\s+answer\s*[*:：\s]*\(?\s*([a-d])\s*\)?"
        r"[ \t*.!$]*(?:<\|im_end\|>)?[ \t]*$")
    matches = list(rx.finditer(region))
    if matches and len({m.group(1).upper() for m in matches}) == 1:
        match = matches[-1]
        tail = region[match.end():]
        if not re.sub(r"[\s*#.!$]|<\|im_end\|>|<\|endoftext\|>", "", tail):
            return dict(selected_label=match.group(1).upper(),
                        type="CLEAR_TERMINAL_FINAL", evidence_start=start + match.start(),
                        evidence_end=start + match.end())
    bare = re.fullmatch(r"\s*([a-d])\s*(?:<\|im_end\|>)?\s*", region, re.I)
    if bare:
        return dict(selected_label=bare.group(1).upper(), type="CLEAR_BARE_RESPONSE",
                    evidence_start=start, evidence_end=len(text))
    return None


def validate_source(rows, manifest):
    cells = manifest["cells"]
    if manifest["rule_version"] != RULE or manifest["expected_n"] != N or not cells:
        raise ValueError("manifest must freeze D13 v1 and GPQA Main N=448")
    unique(cells, lambda row: row["cell"])
    allowed_cells = {row["cell"] for row in cells}
    by_cell = defaultdict(list)
    for row in rows:
        if set(row) != {"cell", "identity", "canonical_index", "output", "options"}:
            raise ValueError("normalized source must contain only the five documented fields")
        if row["cell"] not in allowed_cells or not isinstance(row["output"], str):
            raise ValueError("unknown cell or non-text output")
        if (not isinstance(row["identity"], str) or not row["identity"]
                or type(row["canonical_index"]) is not int):
            raise ValueError("identity/index invalid")
        if (set(row["options"]) != set("ABCD")
                or not all(isinstance(v, str) for v in row["options"].values())):
            raise ValueError("options must be original A/B/C/D texts without gold")
        by_cell[row["cell"]].append(row)
    if set(by_cell) != allowed_cells:
        raise ValueError("missing manifest cells")
    groups = defaultdict(list)
    for cell in cells:
        members = by_cell[cell["cell"]]
        unique(members, lambda row: row["identity"])
        if len(members) != N or {r["canonical_index"] for r in members} != set(range(N)):
            raise ValueError("incomplete/duplicate panel: do not silently discard missing rows")
        groups[cell["group"]].append(cell["cell"])
        if not isinstance(cell["configuration"], dict) or not cell["configuration"]:
            raise ValueError("each cell requires its actual configuration")
    for group_cells in groups.values():
        signatures = [{r["identity"]: (r["canonical_index"], r["options"])
                       for r in by_cell[cell]} for cell in group_cells]
        if any(sig != signatures[0] for sig in signatures[1:]):
            raise ValueError("paired cells differ in identity, index or option ordering")
    contrasts = manifest["contrasts"]
    unique(contrasts, lambda row: row["id"])
    config = {row["cell"]: row for row in cells}
    for contrast in contrasts:
        ref, treatment = contrast["reference"], contrast["treatment"]
        if (ref == treatment or ref not in config or treatment not in config
                or config[ref]["group"] != config[treatment]["group"]
                or not isinstance(contrast["family"], str) or not contrast["family"]):
            raise ValueError("contrasts must freeze valid within-group comparisons and families")


def prepare(source, manifest_path, root):
    rows = read_rows(source)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_source(rows, manifest)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("private", "blind", "review", "sealed"):
        (root / name).mkdir()
    write_json(root / "private/manifest.json", manifest)
    random.Random(SEED).shuffle(rows)
    blind, join, candidates, templates = [], [], [], []
    for index, row in enumerate(rows, 1):
        item_id = f"R{index:06d}"
        blind.append(dict(id=item_id, output=row["output"], options=row["options"]))
        join.append(dict(id=item_id, cell=row["cell"], identity=row["identity"],
                         canonical_index=row["canonical_index"]))
        candidate = terminal_candidate(row["output"])
        if candidate:
            candidate.update(id=item_id, rule_version=RULE,
                             evidence=row["output"][candidate["evidence_start"]:
                                                    candidate["evidence_end"]])
            candidates.append(candidate)
        templates.append(dict(id=item_id, rule_version=RULE, selected_label=None,
                              type="REVIEW_PENDING", evidence_start=0, evidence_end=0,
                              evidence="", reason="", method="", reviewer=""))
    write_rows(root / "private/join-map.jsonl", join)
    write_rows(root / "blind/all.jsonl", blind)
    write_rows(root / "review/surface-candidates.jsonl", candidates)
    write_rows(root / "review/decision-template.jsonl", templates)
    write_json(root / "private/preparation.json", dict(created_utc=utcnow(), records=len(rows)))
    return dict(records=len(rows), surface_candidates=len(candidates), semantic_review_complete=False)


def validate_decisions(blind, decisions):
    indexed = unique(decisions, lambda row: row["id"])
    if set(indexed) != set(blind):
        raise ValueError("decision membership differs from blind outputs")
    for item_id, row in indexed.items():
        label, typ = row["selected_label"], row["type"]
        if row["rule_version"] != RULE or typ == "REVIEW_PENDING":
            raise ValueError("wrong rule version or unfinished review")
        if label is not None and label not in list("ABCD"):
            raise ValueError("invalid selected label")
        null_type = typ.startswith("AMBIGUOUS") or typ in {"NO_COMMITMENT", "OPTION_MAPPING_MISSING"}
        if (label is None) != null_type:
            raise ValueError("null decision requires ambiguity/no commitment/missing mapping type")
        output = blind[item_id]["output"]
        start, end = row["evidence_start"], row["evidence_end"]
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(output)
                or (start == end and (output or label is not None))
                or output[start:end] != row["evidence"]):
            raise ValueError(f"evidence offset/text mismatch: {item_id}")
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("reason", "method", "reviewer")):
            raise ValueError("missing reason, review method or reviewer")
        alternatives = row.get("alternative_labels", [])
        if not isinstance(alternatives, list) or any(
                value is not None and value not in list("ABCD") for value in alternatives):
            raise ValueError("invalid pre-gold alternatives")
    return indexed


def seal(root, decisions_path, disclosure_path):
    root = Path(root)
    if (root / "sealed/extraction-seal.json").exists():
        raise ValueError("already sealed; create a new version, never overwrite")
    blind = unique(read_rows(root / "blind/all.jsonl"), lambda row: row["id"])
    indexed = validate_decisions(blind, read_rows(decisions_path))
    join = read_rows(root / "private/join-map.jsonl")
    manifest = json.loads((root / "private/manifest.json").read_text(encoding="utf-8"))
    mapping = unique(join, lambda row: row["id"])
    if set(mapping) != set(blind):
        raise ValueError("join membership differs from blind outputs")
    validate_source([dict(cell=mapping[k]["cell"], identity=mapping[k]["identity"],
                          canonical_index=mapping[k]["canonical_index"],
                          output=r["output"], options=r["options"])
                     for k, r in blind.items()], manifest)
    disclosure = json.loads(Path(disclosure_path).read_text(encoding="utf-8"))
    if disclosure["gold_accessed_in_new_extraction"] is not False:
        raise ValueError("cannot call a gold-exposed extraction blind D13")
    if not disclosure["historical_exposure"] or not disclosure["manual_scope"]:
        raise ValueError("disclose historical exposure and actual manual reading scope")
    rows = [indexed[key] for key in sorted(indexed)]
    write_rows(root / "sealed/blind.jsonl", list(blind.values()))
    write_rows(root / "sealed/join-map.jsonl", join)
    write_json(root / "sealed/manifest.json", manifest)
    write_rows(root / "sealed/final-decisions-v1.jsonl", rows)
    receipt = dict(rule_version=RULE, sealed_utc=utcnow(), records=len(rows), review_pending=0,
                   type_counts=dict(Counter(r["type"] for r in rows)), disclosure=disclosure)
    write_json(root / "sealed/extraction-seal.json", receipt)
    for path in (root / "sealed").iterdir():
        path.chmod(0o444)
    return receipt


def paired_ci(differences):
    """Same stdlib RNG, identity draws and linear percentiles as original D13."""
    rng = random.Random(SEED)
    draws = sorted(sum(differences[rng.randrange(N)] for _ in range(N)) / N * 100
                   for _ in range(REPLICATES))
    def percentile(q):
        position = (len(draws) - 1) * q
        index = int(position)
        return draws[index] + (draws[min(index + 1, len(draws) - 1)] - draws[index]) * (position - index)
    return [percentile(0.025), percentile(0.975)]


def score(root, gold_path, legacy_path=None, output=None):
    root = Path(root)
    receipt = json.loads((root / "sealed/extraction-seal.json").read_text(encoding="utf-8"))
    blind_rows = read_rows(root / "sealed/blind.jsonl")
    blind = unique(blind_rows, lambda row: row["id"])
    decisions = validate_decisions(blind, read_rows(root / "sealed/final-decisions-v1.jsonl"))
    join = unique(read_rows(root / "sealed/join-map.jsonl"), lambda row: row["id"])
    manifest = json.loads((root / "sealed/manifest.json").read_text(encoding="utf-8"))
    if (receipt["rule_version"] != RULE or receipt["review_pending"] != 0
            or receipt["records"] != len(decisions) or set(join) != set(blind)):
        raise ValueError("seal/membership validation failed before GT access")
    source = [dict(cell=join[r["id"]]["cell"], identity=join[r["id"]]["identity"],
                   canonical_index=join[r["id"]]["canonical_index"], output=r["output"],
                   options=r["options"]) for r in blind_rows]
    validate_source(source, manifest)
    scoring = Path(output) if output else root / "scoring"
    scoring.mkdir(parents=True, exist_ok=False)
    # Receipt is written BEFORE the first GT read and is never overwritten on retries.
    access = dict(started_utc=utcnow(), gold_path=str(Path(gold_path).resolve()),
                  extraction_sealed_utc=receipt["sealed_utc"])
    first_access = root / "private/first-gold-access.json"
    if not first_access.exists():
        write_json(first_access, access)
    write_json(scoring / "gold-access-receipt.json", access)
    gold = unique(read_rows(gold_path), lambda row: (row["cell"], row["identity"]))
    expected = {(r["cell"], r["identity"]) for r in join.values()}
    if set(gold) != expected or any(r["gold_letter"] not in list("ABCD") for r in gold.values()):
        raise ValueError("gold membership/labels differ from panel")
    legacy = None
    if legacy_path:
        legacy = unique(read_rows(legacy_path), lambda row: (row["cell"], row["identity"]))
        if set(legacy) != expected or any(
                r["predicted_label"] is not None and r["predicted_label"] not in list("ABCD")
                for r in legacy.values()):
            raise ValueError("historical S2 membership/labels invalid")
        if any(not r.get("provenance") or not r.get("parser_region") for r in legacy.values()):
            raise ValueError("historical S2 needs producer/region provenance, never guess from S0")
    scored = []
    for item_id, meta in join.items():
        decision = decisions[item_id]
        key = (meta["cell"], meta["identity"])
        correct = int(decision["selected_label"] == gold[key]["gold_letter"])
        row = dict(**meta, selected_label=decision["selected_label"], type=decision["type"],
                   gold_letter=gold[key]["gold_letter"], correct=correct,
                   alternative_labels=decision.get("alternative_labels", []))
        if legacy is not None:
            row.update(old_label=legacy[key]["predicted_label"],
                       old_correct=int(legacy[key]["predicted_label"] == gold[key]["gold_letter"]))
        scored.append(row)
    per_cell = {c["cell"]: sorted([r for r in scored if r["cell"] == c["cell"]],
                                 key=lambda r: r["canonical_index"]) for c in manifest["cells"]}
    configurations = []
    for config in manifest["cells"]:
        rows = per_cell[config["cell"]]
        result = dict(**config, n=N, correct=sum(r["correct"] for r in rows),
                      clear=sum(r["selected_label"] is not None for r in rows),
                      type_counts=dict(Counter(r["type"] for r in rows)))
        result["accuracy_pct"] = result["correct"] / N * 100
        if legacy is not None:
            result["old_correct"] = sum(r["old_correct"] for r in rows)
        configurations.append(result)
    contrasts = []
    for spec in manifest["contrasts"]:
        ref, treatment = per_cell[spec["reference"]], per_cell[spec["treatment"]]
        differences = [t["correct"] - r["correct"] for r, t in zip(ref, treatment)]
        gains, losses = differences.count(1), differences.count(-1)
        result = dict(**spec, n=N, delta_pp=sum(differences) / N * 100,
                      wrong_to_right=gains, right_to_wrong=losses,
                      ci95_pp=paired_ci(differences), mcnemar_exact_p=exact_mcnemar_p(gains, losses))
        if legacy is not None:
            result["old_delta_pp"] = sum(t["old_correct"] - r["old_correct"]
                                         for r, t in zip(ref, treatment)) / N * 100
            result["gain_change_pp"] = result["delta_pp"] - result["old_delta_pp"]
        contrasts.append(result)
    for family in {c["family"] for c in contrasts}:
        members = [c for c in contrasts if c["family"] == family]
        for row, adjusted in zip(members, holm_adjust([c["mcnemar_exact_p"] for c in members])):
            row.update(holm_p=adjusted, family_size=len(members))
    result = dict(rule_version=RULE, configurations=configurations, contrasts=contrasts,
                  bootstrap=dict(replicates=REPLICATES, seed=SEED, kind="paired_identity_percentile"),
                  interpretation="same-question post-hoc exploration; not independent generalization")
    write_rows(scoring / "scored-rows.jsonl", scored)
    write_rows(scoring / "ambiguities-and-no-commitment.jsonl",
               [r for r in scored if r["selected_label"] is None])
    if legacy is not None:
        write_rows(scoring / "new-old-disagreements.jsonl",
                   [r for r in scored if r["selected_label"] != r["old_label"]])
    write_json(scoring / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--root", required=True)
    lock = sub.add_parser("seal")
    lock.add_argument("--root", required=True)
    lock.add_argument("--decisions", required=True)
    lock.add_argument("--disclosure", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--root", required=True)
    scoring.add_argument("--gold", required=True)
    scoring.add_argument("--legacy")
    scoring.add_argument("--output", help="fresh scoring attempt directory; preserves first GT receipt")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare(args["source"], args["manifest"], args["root"])
    elif command == "seal":
        result = seal(args["root"], args["decisions"], args["disclosure"])
    else:
        result = score(args["root"], args["gold"], args["legacy"], args["output"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
