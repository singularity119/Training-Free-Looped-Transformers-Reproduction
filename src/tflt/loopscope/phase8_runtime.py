"""Phase 8 answer-position residual collection and fixed rank-one damping.

Torch is imported only at tensor execution time. The caller supplies the exact
HFLM context position for each batch-one request; cache stash never calls this
Euler callback. No full sequence tensor is retained.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path


def identity_key(identity):
    """Readable, stable key for a JSON-compatible sample identity."""
    return json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class AnswerResidualCollector:
    """Bounded native-residual rows, deduplicated across choice requests."""

    def __init__(self, identities, k):
        self.identities = list(identities)
        self.k = k
        self.keys = [identity_key(x) for x in self.identities]
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("collector identities must be nonempty and unique")
        if k not in (2, 4):
            raise ValueError("Phase 8 collector K must be 2 or 4")
        self.rows = {}
        self.duplicate_calls = 0
        self.positions = {}

    def collect(self, identity, t, vector, position):
        key = identity_key(identity)
        if key not in self.keys or not 0 <= t < self.k:
            raise ValueError("residual identity or timestep outside collector contract")
        row_key = (key, t)
        if row_key in self.rows:
            self.duplicate_calls += 1
            return
        import torch
        row = vector.detach().to(device="cpu", dtype=torch.float32).clone()
        if row.ndim != 1 or not bool(torch.isfinite(row).all()):
            raise ValueError("answer residual must be a finite vector")
        self.rows[row_key] = row
        self.positions[row_key] = position

    def matrix(self, t=1):
        import torch
        missing = [key for key in self.keys if (key, t) not in self.rows]
        if missing:
            raise ValueError("missing calibration residuals: %s" % missing)
        return torch.stack([self.rows[(key, t)] for key in self.keys])

    def summary(self):
        return {"identity_count": len(self.keys), "k": self.k,
                "row_count": len(self.rows), "duplicate_calls": self.duplicate_calls,
                "rows_by_t": {str(t): sum(key[1] == t for key in self.rows)
                              for t in range(self.k)}}


class Phase8Runtime:
    """Callable ``(native_delta, t)`` attached to LoopConfig.residual_transform."""

    def __init__(self, direction=None, strength=0.5, collector=None):
        if strength not in (0.0, 0.5):
            raise ValueError("Phase 8 permits strength 0 (equivalence) or 0.5")
        self.direction = direction
        self.strength = strength
        self.collector = collector
        self._active = None
        self._device_direction = None
        self.calls = []

    @contextmanager
    def context(self, identity, position):
        if self._active is not None:
            raise RuntimeError("nested Phase 8 request context")
        if not isinstance(position, int) or position < 0:
            raise ValueError("pre-answer position must be a nonnegative integer")
        self._active = (identity, position)
        try:
            yield self
        finally:
            self._active = None

    def __call__(self, delta, t):
        if self._active is None:
            raise RuntimeError("Phase 8 residual callback requires evaluator context")
        identity, position = self._active
        if delta.ndim != 3 or delta.shape[0] != 1 or position >= delta.shape[1]:
            raise ValueError("Phase 8 expects batch-one residual and an in-range context position")
        raw = delta[0, position]
        if self.collector is not None:
            self.collector.collect(identity, t, raw, position)
        applied = t >= 1 and self.direction is not None and self.strength != 0
        self.calls.append({"identity": identity, "t": t, "position": position,
                           "applied": applied})
        # Keep exact native tensor object on t0, collection-only and zero paths.
        if not applied:
            return delta
        import torch
        if self._device_direction is None or self._device_direction.device != delta.device:
            v = torch.as_tensor(self.direction, dtype=torch.float32, device=delta.device)
            if v.ndim != 1 or v.numel() != delta.shape[-1] or not bool(torch.isfinite(v).all()):
                raise ValueError("basis must be a finite vector matching hidden width")
            if not math.isclose(float(torch.linalg.vector_norm(v)), 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("runtime basis must have unit length")
            self._device_direction = v
        v = self._device_direction
        raw32 = raw.float()
        damped = raw32 - self.strength * v * torch.dot(v, raw32)
        if not bool(torch.isfinite(damped).all()):
            raise ValueError("nonfinite spectral residual")
        result = delta.clone()
        result[0, position] = damped.to(dtype=delta.dtype)
        return result


def fit_basis(collector, metadata):
    """Uncentered, unnormalized CPU float64 reduced SVD of native t1 rows."""
    import torch
    matrix = collector.matrix(t=1).to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("nonfinite calibration residual")
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    energy = torch.sum(singular_values.square())
    if not bool(torch.isfinite(energy)) or float(energy) <= 0:
        raise ValueError("calibration residuals have no finite nonzero energy")
    direction = vh[0].clone()
    pivot = int(torch.argmax(torch.abs(direction)))
    if float(direction[pivot]) < 0:
        direction = -direction
    direction = direction.float()
    direction = direction / torch.linalg.vector_norm(direction)
    second = float(singular_values[1]) if singular_values.numel() > 1 else 0.0
    return {"schema": "loopscope-phase8-basis-v1", "metadata": dict(metadata),
            "k": collector.k, "fit_t": 1, "rank": 1, "lambda": 0.5,
            "algorithm": "cpu_float64_reduced_svd_uncentered_unnormalized",
            "source_identities": collector.identities, "row_count": len(collector.keys),
            "direction": direction.tolist(),
            "diagnostics": {"top1_energy_fraction": float(singular_values[0].square() / energy),
                            "singular_value_1": float(singular_values[0]),
                            "singular_value_2": second,
                            "singular_value_gap": float(singular_values[0]) - second}}


def save_basis(path, basis):
    """Write once, preserving earlier calibration attempts."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(basis, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_basis(path, expected_metadata=None):
    with Path(path).open(encoding="utf-8") as handle:
        basis = json.load(handle)
    if basis.get("schema") != "loopscope-phase8-basis-v1" or basis.get("k") not in (2, 4):
        raise ValueError("unexpected Phase 8 basis schema or K")
    if expected_metadata is not None and basis["metadata"] != expected_metadata:
        raise ValueError("basis metadata differs from requested model/window/K provenance")
    direction = basis["direction"]
    if not direction or not all(math.isfinite(v) for v in direction):
        raise ValueError("basis direction must be finite")
    if not math.isclose(math.hypot(*direction), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("basis direction must have unit length")
    return basis


def tensor_smoke_checks(torch):
    """Small real-tensor semantic checks, callable inside authorized debug jobs."""
    for dtype in (torch.float16, torch.bfloat16):
        delta = torch.tensor([[[2., 3.], [4., 6.], [5., 7.]]], dtype=dtype)
        runtime = Phase8Runtime([1., 0.])
        with runtime.context("synthetic", 1):
            assert runtime(delta, 0) is delta
            result = runtime(delta, 1)
        expected = delta.clone()
        expected[0, 1, 0] = 2
        assert torch.equal(result, expected)
        zero = Phase8Runtime([1., 0.], strength=0)
        with zero.context("synthetic", 1):
            assert zero(delta, 1) is delta
        assert torch.equal(delta[0, 1], torch.tensor([4., 6.], dtype=dtype))
    return {"parallel_damped": True, "orthogonal_unchanged": True,
            "t0_unchanged": True, "other_tokens_unchanged": True,
            "zero_same_object": True, "native_input_unmodified": True}
