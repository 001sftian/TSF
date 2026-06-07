from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .data import Field4D, idx

Cache = dict[str, Field4D]


@dataclass(frozen=True)
class CandidateTerm:
    """A candidate right-hand-side PDE term or operator block."""

    name: str
    evaluator: Callable[[Cache], Field4D]
    derivative_order: int = 0
    complexity: float = 1.0
    block: str = "term"


def _zeros(shape: tuple[int, int, int, int]) -> Field4D:
    nx, ny, nz, nt = shape
    return [0.0] * (nx * ny * nz * nt)


def periodic_first(u: Field4D, spacing: float, axis: int, shape: tuple[int, int, int, int]) -> Field4D:
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    p = [i, j, k, n]
                    m = [i, j, k, n]
                    p[axis] += 1
                    m[axis] -= 1
                    out[idx(i, j, k, n, shape)] = (u[idx(*p, shape)] - u[idx(*m, shape)]) / (2.0 * spacing)
    return out


def periodic_second(u: Field4D, spacing: float, axis: int, shape: tuple[int, int, int, int]) -> Field4D:
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    p = [i, j, k, n]
                    m = [i, j, k, n]
                    p[axis] += 1
                    m[axis] -= 1
                    c = idx(i, j, k, n, shape)
                    out[c] = (u[idx(*p, shape)] - 2.0 * u[c] + u[idx(*m, shape)]) / (spacing * spacing)
    return out


def time_first(u: Field4D, dt: float, shape: tuple[int, int, int, int]) -> Field4D:
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    c = idx(i, j, k, n, shape)
                    if n == 0:
                        out[c] = (-3.0 * u[c] + 4.0 * u[idx(i, j, k, 1, shape)] - u[idx(i, j, k, 2, shape)]) / (2.0 * dt)
                    elif n == nt - 1:
                        out[c] = (3.0 * u[c] - 4.0 * u[idx(i, j, k, nt - 2, shape)] + u[idx(i, j, k, nt - 3, shape)]) / (2.0 * dt)
                    else:
                        out[c] = (u[idx(i, j, k, n + 1, shape)] - u[idx(i, j, k, n - 1, shape)]) / (2.0 * dt)
    return out


def derivative_cache(u: Field4D, spacing: tuple[float, float, float, float], shape: tuple[int, int, int, int]) -> Cache:
    dx, dy, dz, dt = spacing
    cache: Cache = {"u": u, "1": [1.0] * len(u)}
    cache["u_t"] = time_first(u, dt, shape)
    cache["u_x"] = periodic_first(u, dx, 0, shape)
    cache["u_y"] = periodic_first(u, dy, 1, shape)
    cache["u_z"] = periodic_first(u, dz, 2, shape)
    cache["u_xx"] = periodic_second(u, dx, 0, shape)
    cache["u_yy"] = periodic_second(u, dy, 1, shape)
    cache["u_zz"] = periodic_second(u, dz, 2, shape)
    cache["laplacian"] = [a + b + c for a, b, c in zip(cache["u_xx"], cache["u_yy"], cache["u_zz"])]
    return cache


def build_scalar_library(include_decoys: bool = True) -> list[CandidateTerm]:
    terms = [
        CandidateTerm("u", lambda c: c["u"], 0, 1.0, "atom"),
        CandidateTerm("u_x", lambda c: c["u_x"], 1, 1.2, "gradient"),
        CandidateTerm("u_y", lambda c: c["u_y"], 1, 1.2, "gradient"),
        CandidateTerm("u_z", lambda c: c["u_z"], 1, 1.2, "gradient"),
        CandidateTerm("laplacian", lambda c: c["laplacian"], 2, 1.5, "operator_block"),
    ]
    if include_decoys:
        terms.extend([
            CandidateTerm("1", lambda c: c["1"], 0, 1.0, "atom"),
            CandidateTerm("u^2", lambda c: [v * v for v in c["u"]], 0, 1.4, "nonlinear"),
            CandidateTerm("u^3", lambda c: [v * v * v for v in c["u"]], 0, 1.6, "nonlinear"),
            CandidateTerm("u*u_x", lambda c: [a * b for a, b in zip(c["u"], c["u_x"])], 1, 1.8, "nonlinear_gradient"),
            CandidateTerm("u*u_y", lambda c: [a * b for a, b in zip(c["u"], c["u_y"])], 1, 1.8, "nonlinear_gradient"),
            CandidateTerm("u*u_z", lambda c: [a * b for a, b in zip(c["u"], c["u_z"])], 1, 1.8, "nonlinear_gradient"),
            CandidateTerm("u_xx", lambda c: c["u_xx"], 2, 1.7, "second_derivative"),
            CandidateTerm("u_yy", lambda c: c["u_yy"], 2, 1.7, "second_derivative"),
            CandidateTerm("u_zz", lambda c: c["u_zz"], 2, 1.7, "second_derivative"),
        ])
    return terms
