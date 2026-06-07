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
    """Fourth-order centered periodic first derivative."""
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    p1 = [i, j, k, n]; p2 = [i, j, k, n]
                    m1 = [i, j, k, n]; m2 = [i, j, k, n]
                    p1[axis] += 1; p2[axis] += 2
                    m1[axis] -= 1; m2[axis] -= 2
                    out[idx(i, j, k, n, shape)] = (
                        u[idx(*m2, shape)] - 8.0 * u[idx(*m1, shape)] + 8.0 * u[idx(*p1, shape)] - u[idx(*p2, shape)]
                    ) / (12.0 * spacing)
    return out


def periodic_second(u: Field4D, spacing: float, axis: int, shape: tuple[int, int, int, int]) -> Field4D:
    """Fourth-order centered periodic second derivative."""
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    p1 = [i, j, k, n]; p2 = [i, j, k, n]
                    m1 = [i, j, k, n]; m2 = [i, j, k, n]
                    p1[axis] += 1; p2[axis] += 2
                    m1[axis] -= 1; m2[axis] -= 2
                    c = idx(i, j, k, n, shape)
                    out[c] = (
                        -u[idx(*p2, shape)] + 16.0 * u[idx(*p1, shape)] - 30.0 * u[c]
                        + 16.0 * u[idx(*m1, shape)] - u[idx(*m2, shape)]
                    ) / (12.0 * spacing * spacing)
    return out


def periodic_third(u: Field4D, spacing: float, axis: int, shape: tuple[int, int, int, int]) -> Field4D:
    """Fourth-order-stencil centered periodic third derivative."""
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    p1 = [i, j, k, n]; p2 = [i, j, k, n]
                    m1 = [i, j, k, n]; m2 = [i, j, k, n]
                    p1[axis] += 1; p2[axis] += 2
                    m1[axis] -= 1; m2[axis] -= 2
                    out[idx(i, j, k, n, shape)] = (
                        -u[idx(*m2, shape)] + 2.0 * u[idx(*m1, shape)] - 2.0 * u[idx(*p1, shape)] + u[idx(*p2, shape)]
                    ) / (2.0 * spacing ** 3)
    return out


def periodic_fourth(u: Field4D, spacing: float, axis: int, shape: tuple[int, int, int, int]) -> Field4D:
    """Second-order centered periodic fourth derivative on a five-point stencil."""
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    p1 = [i, j, k, n]; p2 = [i, j, k, n]
                    m1 = [i, j, k, n]; m2 = [i, j, k, n]
                    p1[axis] += 1; p2[axis] += 2
                    m1[axis] -= 1; m2[axis] -= 2
                    c = idx(i, j, k, n, shape)
                    out[c] = (
                        u[idx(*m2, shape)] - 4.0 * u[idx(*m1, shape)] + 6.0 * u[c]
                        - 4.0 * u[idx(*p1, shape)] + u[idx(*p2, shape)]
                    ) / (spacing ** 4)
    return out


def time_first(u: Field4D, dt: float, shape: tuple[int, int, int, int]) -> Field4D:
    """Fourth-order centered time derivative with second-order edge fallback."""
    nx, ny, nz, nt = shape
    out = _zeros(shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for n in range(nt):
                    c = idx(i, j, k, n, shape)
                    if 2 <= n <= nt - 3:
                        out[c] = (
                            u[idx(i, j, k, n - 2, shape)] - 8.0 * u[idx(i, j, k, n - 1, shape)]
                            + 8.0 * u[idx(i, j, k, n + 1, shape)] - u[idx(i, j, k, n + 2, shape)]
                        ) / (12.0 * dt)
                    elif n == 0:
                        out[c] = (-3.0 * u[c] + 4.0 * u[idx(i, j, k, 1, shape)] - u[idx(i, j, k, 2, shape)]) / (2.0 * dt)
                    elif n == nt - 1:
                        out[c] = (3.0 * u[c] - 4.0 * u[idx(i, j, k, nt - 2, shape)] + u[idx(i, j, k, nt - 3, shape)]) / (2.0 * dt)
                    else:
                        out[c] = (u[idx(i, j, k, min(nt - 1, n + 1), shape)] - u[idx(i, j, k, max(0, n - 1), shape)]) / (2.0 * dt)
    return out


def _add_derivatives(cache: Cache, field: str, spacing: tuple[float, float, float, float], shape: tuple[int, int, int, int]) -> None:
    dx, dy, dz, dt = spacing
    f = cache[field]
    cache[f"{field}_t"] = time_first(f, dt, shape)
    cache[f"{field}_x"] = periodic_first(f, dx, 0, shape)
    cache[f"{field}_y"] = periodic_first(f, dy, 1, shape)
    cache[f"{field}_z"] = periodic_first(f, dz, 2, shape)
    cache[f"{field}_xx"] = periodic_second(f, dx, 0, shape)
    cache[f"{field}_yy"] = periodic_second(f, dy, 1, shape)
    cache[f"{field}_zz"] = periodic_second(f, dz, 2, shape)
    cache[f"{field}_xxx"] = periodic_third(f, dx, 0, shape)
    cache[f"{field}_xxxx"] = periodic_fourth(f, dx, 0, shape)
    cache[f"{field}_yyyy"] = periodic_fourth(f, dy, 1, shape)
    cache[f"{field}_zzzz"] = periodic_fourth(f, dz, 2, shape)


def derivative_cache(u: Field4D, spacing: tuple[float, float, float, float], shape: tuple[int, int, int, int], extra_fields: dict[str, Field4D] | None = None) -> Cache:
    cache: Cache = {"u": u, "1": [1.0] * len(u)}
    if extra_fields:
        cache.update(extra_fields)
    for field in list(cache):
        if field != "1":
            _add_derivatives(cache, field, spacing, shape)
    cache["u_t"] = cache["u_t"]
    return cache


def build_scalar_library(include_decoys: bool = True) -> list[CandidateTerm]:
    """Expanded scalar-field term library with no compact operator blocks."""
    terms = [
        CandidateTerm("u", lambda c: c["u"], 0, 1.0, "atom"),
        CandidateTerm("u_x", lambda c: c["u_x"], 1, 1.2, "first_derivative"),
        CandidateTerm("u_y", lambda c: c["u_y"], 1, 1.2, "first_derivative"),
        CandidateTerm("u_z", lambda c: c["u_z"], 1, 1.2, "first_derivative"),
        CandidateTerm("u_xx", lambda c: c["u_xx"], 2, 1.5, "second_derivative"),
        CandidateTerm("u_yy", lambda c: c["u_yy"], 2, 1.5, "second_derivative"),
        CandidateTerm("u_zz", lambda c: c["u_zz"], 2, 1.5, "second_derivative"),
        CandidateTerm("u_u_x", lambda c: [a * b for a, b in zip(c["u"], c["u_x"])], 1, 1.8, "product"),
        CandidateTerm("u_u_y", lambda c: [a * b for a, b in zip(c["u"], c["u_y"])], 1, 1.8, "product"),
        CandidateTerm("u_u_z", lambda c: [a * b for a, b in zip(c["u"], c["u_z"])], 1, 1.8, "product"),
        CandidateTerm("u_cubed", lambda c: [v * v * v for v in c["u"]], 0, 1.6, "power"),
        CandidateTerm("u_xxx", lambda c: c["u_xxx"], 3, 2.2, "third_derivative"),
        CandidateTerm("u_xxxx", lambda c: c["u_xxxx"], 4, 2.6, "fourth_derivative"),
        CandidateTerm("u_yyyy", lambda c: c["u_yyyy"], 4, 2.6, "fourth_derivative"),
        CandidateTerm("u_zzzz", lambda c: c["u_zzzz"], 4, 2.6, "fourth_derivative"),
    ]
    if include_decoys:
        terms.extend([
            CandidateTerm("one", lambda c: c["1"], 0, 1.0, "atom"),
            CandidateTerm("u_squared", lambda c: [v * v for v in c["u"]], 0, 1.4, "power"),
        ])
    return terms


def build_nls_library(target: str) -> list[CandidateTerm]:
    other = "b" if target == "a" else "a"
    sign = -1.0 if target == "a" else 1.0
    return [
        CandidateTerm(f"{other}_xx", lambda c, other=other, s=sign: [s * v for v in c[f"{other}_xx"]], 2, 1.5, "second_derivative"),
        CandidateTerm(f"{other}_yy", lambda c, other=other, s=sign: [s * v for v in c[f"{other}_yy"]], 2, 1.5, "second_derivative"),
        CandidateTerm(f"{other}_zz", lambda c, other=other, s=sign: [s * v for v in c[f"{other}_zz"]], 2, 1.5, "second_derivative"),
        CandidateTerm(f"abspsi_squared_{other}", lambda c, other=other, s=sign: [s * (a * a + b * b) * o for a, b, o in zip(c["a"], c["b"], c[other])], 0, 2.0, "product"),
        CandidateTerm(target, lambda c, target=target: c[target], 0, 1.0, "atom"),
    ]


def build_ns_library(component: str) -> list[CandidateTerm]:
    pressure_axis = "xyz"["uvw".index(component)]
    return [
        CandidateTerm(f"{component}_xx", lambda c, component=component: c[f"{component}_xx"], 2, 1.5, "second_derivative"),
        CandidateTerm(f"{component}_yy", lambda c, component=component: c[f"{component}_yy"], 2, 1.5, "second_derivative"),
        CandidateTerm(f"{component}_zz", lambda c, component=component: c[f"{component}_zz"], 2, 1.5, "second_derivative"),
        CandidateTerm(f"u_{component}_x", lambda c, component=component: [a * b for a, b in zip(c["u"], c[f"{component}_x"])], 1, 2.0, "product"),
        CandidateTerm(f"v_{component}_y", lambda c, component=component: [a * b for a, b in zip(c["v"], c[f"{component}_y"])], 1, 2.0, "product"),
        CandidateTerm(f"w_{component}_z", lambda c, component=component: [a * b for a, b in zip(c["w"], c[f"{component}_z"])], 1, 2.0, "product"),
        CandidateTerm(f"p_{pressure_axis}", lambda c, pressure_axis=pressure_axis: c[f"p_{pressure_axis}"], 1, 1.4, "pressure_derivative"),
    ]
