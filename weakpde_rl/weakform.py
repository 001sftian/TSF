from __future__ import annotations

from dataclasses import dataclass

from .data import FieldData, idx
from .library import CandidateTerm, derivative_cache
from .linalg import column_norms


@dataclass(frozen=True)
class WeakSystem:
    theta: list[list[float]]
    y: list[float]
    names: list[str]
    complexities: list[float]
    raw_theta: list[list[float]]


class WeakFormProjector:
    """Compact-window weak-form projection for noisy regular-grid data."""

    def __init__(self, radius: int = 1, stride: int = 2):
        if radius < 1:
            raise ValueError("radius must be positive")
        self.radius = radius
        self.stride = stride

    def _window_weights(self) -> list[tuple[int, int, int, int, float]]:
        r = self.radius
        values: list[tuple[int, int, int, int, float]] = []
        total = 0.0
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    for dn in range(-r, r + 1):
                        s2 = (di / r) ** 2 + (dj / r) ** 2 + (dk / r) ** 2 + (dn / r) ** 2
                        w = max(1.0 - s2 / 4.0, 0.0) ** 3
                        values.append((di, dj, dk, dn, w))
                        total += w
        return [(di, dj, dk, dn, w / total) for di, dj, dk, dn, w in values]

    def project(self, data: FieldData, terms: list[CandidateTerm]) -> WeakSystem:
        cache = derivative_cache(data.u, data.spacing, data.shape)
        fields = [term.evaluator(cache) for term in terms]
        target = cache["u_t"]
        weights = self._window_weights()
        r = self.radius
        nx, ny, nz, nt = data.shape
        rows: list[list[float]] = []
        rhs: list[float] = []
        for i in range(r, nx - r, self.stride):
            for j in range(r, ny - r, self.stride):
                for k in range(r, nz - r, self.stride):
                    for n in range(r, nt - r, self.stride):
                        row = []
                        for field in fields:
                            row.append(sum(field[idx(i + di, j + dj, k + dk, n + dn, data.shape)] * w for di, dj, dk, dn, w in weights))
                        rows.append(row)
                        rhs.append(sum(target[idx(i + di, j + dj, k + dk, n + dn, data.shape)] * w for di, dj, dk, dn, w in weights))
        scales = column_norms(rows)
        scales = [s if s else 1.0 for s in scales]
        theta = [[value / scales[col] for col, value in enumerate(row)] for row in rows]
        return WeakSystem(theta, rhs, [t.name for t in terms], [t.complexity for t in terms], rows)
