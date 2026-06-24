"""Loop-iteration strategy kernels.

The functions operate on tensor-like objects supporting `+`, `-`, and scalar
multiplication. Torch tensors work on the remote environment; floats are used in
local tests.
"""

from __future__ import annotations

from typing import Callable, List

from tflt.config import LoopConfig


TensorLike = object
Operator = Callable[[TensorLike], TensorLike]


def blend(x: TensorLike, y: TensorLike, weight: float) -> TensorLike:
    return x + weight * (y - x)  # type: ignore[operator]


def residual(operator: Operator, x: TensorLike) -> TensorLike:
    y = operator(x)
    return y - x  # type: ignore[operator]


def run_loop(operator: Operator, x0: TensorLike, config: LoopConfig) -> TensorLike:
    """Apply the configured loop strategy to `operator` starting at `x0`."""

    strategy = "damped_euler" if config.strategy == "euler" else config.strategy
    if config.k == 1 and strategy in {"naive", "uniform_loop", "damped_euler"}:
        return operator(x0)
    if strategy in {"naive", "uniform_loop"}:
        return _naive(operator, x0, config.k)
    if strategy == "damped_euler":
        return _damped_euler(operator, x0, config.k, config.alpha)
    if strategy == "heavy_ball":
        return _heavy_ball(operator, x0, config.k, config.alpha, config.beta)
    if strategy == "heun":
        return _heun(operator, x0, config.alpha)
    if strategy == "midpoint":
        return _midpoint(operator, x0, config.alpha)
    if strategy == "rk4":
        return _rk4(operator, x0, config.alpha)
    if strategy == "rk":
        return _paper_rk(operator, x0, config)
    if strategy == "anderson":
        return _anderson(operator, x0, config)
    raise ValueError("unknown loop strategy: %s" % config.strategy)


def _naive(operator: Operator, x: TensorLike, k: int) -> TensorLike:
    for _ in range(k):
        x = operator(x)
    return x


def _damped_euler(operator: Operator, x: TensorLike, k: int, alpha: float) -> TensorLike:
    step = alpha / float(k)
    for _ in range(k):
        x = x + step * residual(operator, x)  # type: ignore[operator]
    return x


def _heavy_ball(operator: Operator, x: TensorLike, k: int, alpha: float, beta: float) -> TensorLike:
    step = alpha / float(k)
    prev = x
    for i in range(k):
        r = residual(operator, x)
        momentum = 0.0 if i == 0 else beta
        nxt = x + step * r + momentum * (x - prev)  # type: ignore[operator]
        prev, x = x, nxt
    return x


def _heun(operator: Operator, x: TensorLike, alpha: float) -> TensorLike:
    k1 = residual(operator, x)
    probe = x + alpha * k1  # type: ignore[operator]
    k2 = residual(operator, probe)
    return x + 0.5 * alpha * (k1 + k2)  # type: ignore[operator]


def _midpoint(operator: Operator, x: TensorLike, alpha: float) -> TensorLike:
    k1 = residual(operator, x)
    probe = x + 0.5 * alpha * k1  # type: ignore[operator]
    k2 = residual(operator, probe)
    return x + alpha * k2  # type: ignore[operator]


def _rk4(operator: Operator, x: TensorLike, alpha: float) -> TensorLike:
    k1 = residual(operator, x)
    k2 = residual(operator, x + 0.5 * alpha * k1)  # type: ignore[operator]
    k3 = residual(operator, x + 0.5 * alpha * k2)  # type: ignore[operator]
    k4 = residual(operator, x + alpha * k3)  # type: ignore[operator]
    return x + (alpha / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)  # type: ignore[operator]


def _rk_dispatch(operator: Operator, x: TensorLike, config: LoopConfig) -> TensorLike:
    if config.k <= 1:
        return x + config.alpha * residual(operator, x)  # type: ignore[operator]
    if config.k == 2:
        return _heun(operator, x, config.alpha)
    if config.k == 3:
        # Kutta's third-order method.
        k1 = residual(operator, x)
        k2 = residual(operator, x + 0.5 * config.alpha * k1)  # type: ignore[operator]
        k3 = residual(
            operator,
            x + config.alpha * (-1.0 * k1 + 2.0 * k2),  # type: ignore[operator]
        )
        return x + (config.alpha / 6.0) * (k1 + 4.0 * k2 + k3)  # type: ignore[operator]
    if config.k == 4:
        return _rk4(operator, x, config.alpha)
    return _damped_euler(operator, x, config.k, config.alpha)


def _paper_rk(operator: Operator, x: TensorLike, config: LoopConfig) -> TensorLike:
    """Paper-style K-stage RK wrapper.

    The practical paper variant blends the one-step window output `g(x0)` with
    the K damped-Euler substeps. `beta=0` recovers the default damped Euler path.
    """

    refined = _damped_euler(operator, x, config.k, config.alpha)
    if config.beta == 0:
        return refined
    one_step = operator(x)
    return config.beta * one_step + (1.0 - config.beta) * refined  # type: ignore[operator]


def _anderson(operator: Operator, x: TensorLike, config: LoopConfig) -> TensorLike:
    """Regularized Anderson acceleration for torch tensors, with safe fallback."""

    try:
        import torch
    except Exception:
        return _damped_euler(operator, x, config.k, config.alpha)

    if not hasattr(x, "reshape"):
        return _damped_euler(operator, x, config.k, config.alpha)

    xs: List[TensorLike] = []
    fs: List[TensorLike] = []
    cur = x
    for _ in range(config.k):
        nxt = operator(cur)
        xs.append(cur)
        fs.append(nxt)
        recent_x = xs[-config.anderson_m :]
        recent_f = fs[-config.anderson_m :]
        if len(recent_x) < 2:
            cur = blend(cur, nxt, config.alpha)
            continue
        try:
            residuals = [(_flatten(f) - _flatten(xi)) for xi, f in zip(recent_x, recent_f)]
            mat = torch.stack(residuals, dim=0)
            gram = mat @ mat.transpose(0, 1)
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            gram = gram + config.anderson_lambda * eye
            ones = torch.ones((gram.shape[0], 1), dtype=gram.dtype, device=gram.device)
            top = torch.cat([gram, ones], dim=1)
            bottom = torch.cat([ones.transpose(0, 1), torch.zeros_like(ones[:1])], dim=1)
            system = torch.cat([top, bottom], dim=0)
            rhs = torch.zeros((system.shape[0], 1), dtype=gram.dtype, device=gram.device)
            rhs[-1, 0] = 1.0
            coeff = torch.linalg.solve(system, rhs)[:-1, 0]
            mixed = None
            for c, f in zip(coeff, recent_f):
                mixed = c * f if mixed is None else mixed + c * f
            cur = blend(cur, mixed, config.alpha)
        except Exception:
            cur = blend(cur, nxt, config.alpha)
    return cur


def _flatten(x: TensorLike) -> TensorLike:
    return x.reshape(-1)  # type: ignore[attr-defined]
