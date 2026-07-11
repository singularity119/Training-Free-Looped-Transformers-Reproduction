"""Fail-fast, label-free LoopPilot controller interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol


class Action(str, Enum):
    BASELINE = "BASELINE"
    LOOP_K2 = "LOOP_K2"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("controller decision reason must be non-empty")


class Controller(Protocol):
    def decide(self, probe: Any = None) -> Decision:
        ...


class AlwaysLoop:
    """Engineering controller used to prove equivalence with legacy K=2."""

    def decide(self, probe: Any = None) -> Decision:
        return Decision(Action.LOOP_K2, "always_loop_engineering_control")


class NeverLoop:
    """Engineering controller used to prove equivalence with baseline."""

    def decide(self, probe: Any = None) -> Decision:
        return Decision(Action.BASELINE, "never_loop_engineering_control")


@dataclass(frozen=True)
class FrozenRule:
    """A versioned single-signal rule; its threshold must be supplied explicitly."""

    signal: str
    operator: str
    threshold: float
    rule_id: str

    def __post_init__(self) -> None:
        if self.signal not in {"r0_median", "c0", "n0_median"}:
            raise ValueError("online rule signal must be one of r0_median, c0, n0_median")
        if self.operator not in {"lt", "le", "gt", "ge"}:
            raise ValueError("operator must be one of lt, le, gt, ge")
        if not self.rule_id.strip():
            raise ValueError("rule_id must be non-empty")
        if not isinstance(self.threshold, (int, float)):
            raise TypeError("threshold must be numeric")

    def decide(self, probe: Any = None) -> Decision:
        if not isinstance(probe, Mapping):
            raise TypeError("FrozenRule requires a mapping probe")
        if self.signal not in probe:
            raise KeyError("probe is missing required signal %s" % self.signal)
        value = probe[self.signal]
        if not isinstance(value, (int, float)):
            raise TypeError("probe signal %s must be numeric" % self.signal)
        comparisons = {
            "lt": value < self.threshold,
            "le": value <= self.threshold,
            "gt": value > self.threshold,
            "ge": value >= self.threshold,
        }
        action = Action.LOOP_K2 if comparisons[self.operator] else Action.BASELINE
        return Decision(action, "%s:%s" % (self.rule_id, self.operator))


def parse_controller_spec(spec: str) -> Controller:
    """Parse engineering-only controller names and reject implicit fallbacks."""

    normalized = spec.strip().lower().replace("-", "_")
    if normalized == "always_loop":
        return AlwaysLoop()
    if normalized == "never_loop":
        return NeverLoop()
    raise ValueError("unsupported controller spec: %s" % spec)
