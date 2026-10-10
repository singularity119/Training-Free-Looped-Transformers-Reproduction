"""Gate D-2's frozen S1 extraction; original Phase11 S0 is unchanged."""
from __future__ import annotations

import re

S1_PATTERN = r"final[ \t]+answer[ \t]*[:：][ \t]*(?:\([ \t]*([A-D])[ \t]*\)|([A-D]))[ \t]*[.!;,．！。]?"


def extract_s1(text):
    matches = []
    for line in text.splitlines():
        match = re.fullmatch(S1_PATTERN, line.strip(), re.IGNORECASE)
        if match:
            matches.append((match[1] or match[2]).upper())
    valid = len(matches) == 1
    return {"extraction_status": "unique_valid" if valid else
            ("no_valid_final_line" if not matches else "multiple_valid_final_lines"),
            "valid_final_line_count": len(matches),
            "predicted_label": matches[0] if valid else None,
            "prediction_index": "ABCD".index(matches[0]) if valid else None}


def score_s1(rows, gold):
    if not rows or len(rows) != len(gold):
        raise ValueError("S1 scoring requires all aligned samples")
    correct = [int(r["prediction_index"] is not None and r["prediction_index"] == g)
               for r, g in zip(rows, gold)]
    n = len(rows)
    valid = sum(r["extraction_status"] == "unique_valid" for r in rows)
    groups = {}
    for reason in ("eos", "length_limit"):
        indices = [i for i, r in enumerate(rows) if r["end_reason"] == reason]
        groups[reason] = {"n": len(indices),
                          "extraction_success_count": sum(rows[i]["prediction_index"] is not None for i in indices),
                          "correct": sum(correct[i] for i in indices)}
    if sum(g["n"] for g in groups.values()) != n:
        raise ValueError("unknown termination reason")
    return {"n": n, "correct": sum(correct), "accuracy": sum(correct) / n,
            "extraction_success_count": valid, "extraction_success_rate": valid / n,
            "extraction_failure_count": n - valid, "extraction_failure_rate": (n - valid) / n,
            "truncated_count": groups["length_limit"]["n"],
            "truncated_rate": groups["length_limit"]["n"] / n, "groups": groups}, correct
