"""Configuration objects for looped inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


IterationMode = str
LoopStrategy = str
CacheStrategy = str
DecodeMode = str


_ITERATION_MODES = {"block", "layer"}
_STRATEGIES = {
    "naive",
    "uniform_loop",
    "damped_euler",
    "euler",
    "rk",
    "heun",
    "midpoint",
    "rk4",
    "heavy_ball",
    "anderson",
}
_CACHE_STRATEGIES = {"first", "last", "none"}
_DECODE_MODES = {"bypass", "full", "first_n"}


@dataclass(frozen=True)
class LoopConfig:
    """Public loop wrapper configuration.

    `window` is inclusive: `(a, b)` means loop layers `a, ..., b`.
    """

    model_alias: str
    window: Tuple[int, int]
    k: int = 2
    iteration_mode: IterationMode = "block"
    strategy: LoopStrategy = "damped_euler"
    alpha: float = 1.0
    beta: float = 0.0
    cache_strategy: CacheStrategy = "last"
    decode_mode: DecodeMode = "bypass"
    first_n: Optional[int] = None
    anderson_m: int = 5
    anderson_lambda: float = 1e-4
    audit_collector: Any = None

    def __post_init__(self) -> None:
        a, b = self.window
        if a < 0 or b < 0 or a > b:
            raise ValueError("window must be an inclusive non-negative range (a, b)")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.iteration_mode not in _ITERATION_MODES:
            raise ValueError("iteration_mode must be one of %s" % sorted(_ITERATION_MODES))
        if self.strategy not in _STRATEGIES:
            raise ValueError("strategy must be one of %s" % sorted(_STRATEGIES))
        if self.cache_strategy not in _CACHE_STRATEGIES:
            raise ValueError("cache_strategy must be one of %s" % sorted(_CACHE_STRATEGIES))
        if self.decode_mode not in _DECODE_MODES:
            raise ValueError("decode_mode must be one of %s" % sorted(_DECODE_MODES))
        if self.decode_mode == "first_n" and (self.first_n is None or self.first_n < 1):
            raise ValueError("first_n must be set to a positive integer for decode_mode='first_n'")
        if self.anderson_m < 2:
            raise ValueError("anderson_m must be >= 2")
        if self.anderson_lambda < 0:
            raise ValueError("anderson_lambda must be >= 0")

    @property
    def start(self) -> int:
        return self.window[0]

    @property
    def end(self) -> int:
        return self.window[1]

    @property
    def window_width(self) -> int:
        return self.end - self.start + 1

    @classmethod
    def from_window_string(cls, model_alias: str, window: str, **kwargs: object) -> "LoopConfig":
        """Build a config from `a:b`, `[a,b]`, or `a-b` window text."""

        text = window.strip().replace("[", "").replace("]", "").replace(" ", "")
        for sep in (":", ",", "-"):
            if sep in text:
                left, right = text.split(sep, 1)
                return cls(model_alias=model_alias, window=(int(left), int(right)), **kwargs)
        layer = int(text)
        return cls(model_alias=model_alias, window=(layer, layer), **kwargs)


def normalize_tasks(tasks: Sequence[str]) -> Tuple[str, ...]:
    return tuple(task.strip() for task in tasks if task.strip())
