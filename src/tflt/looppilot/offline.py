"""Strict doc-level pairing and counterfactual policy primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from tflt.looppilot.controller import Action
from tflt.looppilot.schema import DocKey


@dataclass(frozen=True)
class EvalSample:
    key: DocKey
    revision_closure_id: str
    prediction: int
    gold: int
    choice_loglikelihoods: Tuple[float, ...]
    subject: str
    prompt_length: int

    def __post_init__(self) -> None:
        if not self.revision_closure_id.strip():
            raise ValueError("revision_closure_id must be non-empty")
        if len(self.choice_loglikelihoods) != 4:
            raise ValueError("MMLU samples must contain exactly four choice loglikelihoods")
        if self.prediction not in range(4) or self.gold not in range(4):
            raise ValueError("prediction and gold must be choice indices 0..3")
        if not self.subject.strip() or self.prompt_length < 1:
            raise ValueError("subject and positive prompt_length are required")
        if any(not math.isfinite(float(value)) for value in self.choice_loglikelihoods):
            raise ValueError("choice loglikelihoods must be finite")


@dataclass(frozen=True)
class PairedSample:
    baseline: EvalSample
    loop: EvalSample

    @property
    def baseline_correct(self) -> bool:
        return self.baseline.prediction == self.baseline.gold

    @property
    def loop_correct(self) -> bool:
        return self.loop.prediction == self.loop.gold

    @property
    def outcome(self) -> str:
        if self.baseline_correct and self.loop_correct:
            return "both_correct"
        if not self.baseline_correct and not self.loop_correct:
            return "both_wrong"
        if not self.baseline_correct and self.loop_correct:
            return "loop_helps"
        return "loop_hurts"


def strict_join_samples(
    baseline_rows: Iterable[EvalSample], loop_rows: Iterable[EvalSample]
) -> List[PairedSample]:
    baseline = _unique_index(baseline_rows, "baseline")
    loop = _unique_index(loop_rows, "loop")
    if set(baseline) != set(loop):
        missing_loop = sorted(set(baseline) - set(loop))
        missing_baseline = sorted(set(loop) - set(baseline))
        raise ValueError(
            "paired join is not closed; missing_loop=%s missing_baseline=%s"
            % (missing_loop, missing_baseline)
        )
    pairs = []
    for canonical in sorted(baseline):
        left, right = baseline[canonical], loop[canonical]
        if left.revision_closure_id != right.revision_closure_id:
            raise ValueError("revision closure mismatch for %s" % canonical)
        if left.gold != right.gold or left.subject != right.subject:
            raise ValueError("paired sample metadata mismatch for %s" % canonical)
        pairs.append(PairedSample(left, right))
    return pairs


def validate_doc_actions(rows: Sequence[Tuple[DocKey, int, Action]]) -> None:
    grouped: Dict[str, Dict[int, Action]] = {}
    for doc_key, choice_index, action in rows:
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        choices = grouped.setdefault(doc_key.canonical(), {})
        if choice_index in choices:
            raise ValueError("duplicate choice request for %s index %d" % (doc_key.canonical(), choice_index))
        choices[choice_index] = action
    for canonical, choices in grouped.items():
        if set(choices) != {0, 1, 2, 3}:
            raise ValueError("doc %s must contain choice requests 0,1,2,3" % canonical)
        if len(set(choices.values())) != 1:
            raise ValueError("doc %s has inconsistent LoopPilot actions" % canonical)


def counterfactual_correct(pair: PairedSample, action: Action) -> bool:
    if action == Action.BASELINE:
        return pair.baseline_correct
    if action == Action.LOOP_K2:
        return pair.loop_correct
    raise ValueError("unsupported action: %s" % action)


def _unique_index(rows: Iterable[EvalSample], label: str) -> Dict[str, EvalSample]:
    indexed: Dict[str, EvalSample] = {}
    for row in rows:
        canonical = row.key.canonical()
        if canonical in indexed:
            raise ValueError("duplicate %s sample key: %s" % (label, canonical))
        indexed[canonical] = row
    return indexed
