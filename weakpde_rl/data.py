from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

Field4D = list[float]


@dataclass(frozen=True)
class FieldData:
    """Regular-grid spatiotemporal field data stored as flat 4-D arrays."""

    u: Field4D
    shape: tuple[int, int, int, int]
    x: list[float]
    y: list[float]
    z: list[float]
    t: list[float]
    true_coefficients: dict[str, float]
    fields: dict[str, Field4D] = field(default_factory=dict)
    target: str = "u"

    @property
    def spacing(self) -> tuple[float, float, float, float]:
        return (self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0], self.t[1] - self.t[0])


def idx(i: int, j: int, k: int, n: int, shape: tuple[int, int, int, int]) -> int:
    nx, ny, nz, nt = shape
    return (((i % nx) * ny + (j % ny)) * nz + (k % nz)) * nt + n


def _grid(nx: int, ny: int, nz: int, nt: int, t_end: float = 0.5):
    return (
        [2.0 * math.pi * i / nx for i in range(nx)],
        [2.0 * math.pi * i / ny for i in range(ny)],
        [2.0 * math.pi * i / nz for i in range(nz)],
        [t_end * i / (nt - 1) for i in range(nt)],
        (nx, ny, nz, nt),
    )


def _field_from_formula(nx: int, ny: int, nz: int, nt: int, formula, coeffs: dict[str, float], noise: float = 0.0, seed: int = 7, t_end: float = 0.5) -> FieldData:
    x, y, z, t, shape = _grid(nx, ny, nz, nt, t_end)
    values = [0.0] * (nx * ny * nz * nt)
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            for k, zz in enumerate(z):
                for n, tt in enumerate(t):
                    values[idx(i, j, k, n, shape)] = formula(xx, yy, zz, tt)
    if noise:
        rng = random.Random(seed)
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        values = [v + noise * std * rng.gauss(0.0, 1.0) for v in values]
    return FieldData(values, shape, x, y, z, t, coeffs)


def make_3d_reaction_advection_diffusion(nx: int = 18, ny: int = 16, nz: int = 14, nt: int = 11, noise: float = 0.0, seed: int = 7, coeffs: dict[str, float] | None = None) -> FieldData:
    params = {"u": -0.15, "u_x": -0.80, "u_y": 0.45, "u_z": -0.30, "laplacian": 0.06}
    if coeffs:
        params.update(coeffs)
    cx, cy, cz = -params["u_x"], -params["u_y"], -params["u_z"]
    alpha, nu = params["u"], params["laplacian"]
    modes = [(1.0, 1, 1, 1, 0.2), (0.35, 2, 1, 1, -0.7), (0.25, 1, 2, 1, 0.6)]

    def formula(xx, yy, zz, tt):
        value = 0.0
        for amp, kx, ky, kz, phase in modes:
            k2 = kx * kx + ky * ky + kz * kz
            theta = kx * (xx - cx * tt) + ky * (yy - cy * tt) + kz * (zz - cz * tt) + phase
            value += amp * math.exp((alpha - nu * k2) * tt) * math.sin(theta)
        return value

    return _field_from_formula(nx, ny, nz, nt, formula, params, noise, seed, t_end=0.8)


def make_3d_heat(nx: int = 14, ny: int = 12, nz: int = 10, nt: int = 9, nu: float = 0.08) -> FieldData:
    modes = [(1.0, 1, 1, 1, 0.1), (0.4, 2, 1, 1, -0.3)]
    def formula(x, y, z, t):
        return sum(a * math.exp(-nu * (kx*kx+ky*ky+kz*kz) * t) * math.sin(kx*x + ky*y + kz*z + ph) for a, kx, ky, kz, ph in modes)
    return _field_from_formula(nx, ny, nz, nt, formula, {"laplacian": nu}, t_end=0.5)


def make_3d_burgers(nx: int = 14, ny: int = 12, nz: int = 10, nt: int = 9) -> FieldData:
    # Manufactured 3-D scalar Burgers-like data: u_t = -u*u_x + nu*laplacian.
    return _manufactured_scalar(nx, ny, nz, nt, {"u*u_x": -1.0, "laplacian": 0.04}, amplitude=0.28)


def make_3d_allen_cahn(nx: int = 14, ny: int = 12, nz: int = 10, nt: int = 9) -> FieldData:
    return _manufactured_scalar(nx, ny, nz, nt, {"u": 1.0, "u^3": -1.0, "laplacian": 0.03}, amplitude=0.35)


def make_3d_cahn_hilliard(nx: int = 14, ny: int = 12, nz: int = 10, nt: int = 9) -> FieldData:
    return _manufactured_scalar(nx, ny, nz, nt, {"laplacian": -0.15, "biharmonic": -0.01}, amplitude=0.4)


def make_3d_kdv(nx: int = 14, ny: int = 12, nz: int = 10, nt: int = 9) -> FieldData:
    return _manufactured_scalar(nx, ny, nz, nt, {"u*u_x": -6.0, "u_xxx": -0.08}, amplitude=0.12)


def make_3d_reaction_diffusion(nx: int = 14, ny: int = 12, nz: int = 10, nt: int = 9) -> FieldData:
    return _manufactured_scalar(nx, ny, nz, nt, {"u": 0.7, "u^3": -0.9, "laplacian": 0.05}, amplitude=0.32)


def _manufactured_scalar(nx: int, ny: int, nz: int, nt: int, coeffs: dict[str, float], amplitude: float) -> FieldData:
    # Explicit Euler pseudo-spectral-free dataset using the same fourth-order operators used by discovery.
    from .library import derivative_cache
    x, y, z, t, shape = _grid(nx, ny, nz, nt, 0.16)
    dt = t[1] - t[0]
    frames: list[Field4D] = []
    u = [0.0] * (nx * ny * nz)
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            for k, zz in enumerate(z):
                u[(i * ny + j) * nz + k] = amplitude * (math.sin(xx + yy + zz) + 0.35 * math.cos(2 * xx - yy + zz))
    def pack(frames):
        values = [0.0] * (nx * ny * nz * nt)
        for n, frame in enumerate(frames):
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        values[idx(i, j, k, n, shape)] = frame[(i * ny + j) * nz + k]
        return values
    for n in range(nt):
        frames.append(u[:])
        if n == nt - 1:
            break
        tmp = pack([u for _ in range(nt)])
        cache = derivative_cache(tmp, (x[1]-x[0], y[1]-y[0], z[1]-z[0], dt), shape)
        rhs = [0.0] * len(u)
        for name, coef in coeffs.items():
            if name == "u*u_x":
                arr = [a * b for a, b in zip(cache["u"], cache["u_x"])]
            elif name == "u*u_y":
                arr = [a * b for a, b in zip(cache["u"], cache["u_y"])]
            elif name == "u*u_z":
                arr = [a * b for a, b in zip(cache["u"], cache["u_z"])]
            elif name == "u^3":
                arr = [v * v * v for v in cache["u"]]
            else:
                arr = cache[name]
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        rhs[(i * ny + j) * nz + k] += coef * arr[idx(i, j, k, 0, shape)]
        u = [val + dt * rv for val, rv in zip(u, rhs)]
    return FieldData(pack(frames), shape, x, y, z, t, coeffs)


def make_3d_nls(nx: int = 12, ny: int = 10, nz: int = 8, nt: int = 9) -> FieldData:
    x, y, z, t, shape = _grid(nx, ny, nz, nt, 0.5)
    a = [0.0] * (nx * ny * nz * nt); b = [0.0] * len(a)
    amp, kx, ky, kz, g = 0.45, 1, 1, 1, 2.0
    omega = kx*kx + ky*ky + kz*kz - g * amp * amp
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            for k, zz in enumerate(z):
                for n, tt in enumerate(t):
                    phase = kx*xx + ky*yy + kz*zz - omega*tt
                    a[idx(i, j, k, n, shape)] = amp * math.cos(phase)
                    b[idx(i, j, k, n, shape)] = amp * math.sin(phase)
    return FieldData(a, shape, x, y, z, t, {"laplacian_b": -1.0, "(|psi|^2)b": -g}, {"a": a, "b": b}, "a")


def make_3d_ns(nx: int = 12, ny: int = 10, nz: int = 8, nt: int = 9, nu: float = 0.04) -> FieldData:
    x, y, z, t, shape = _grid(nx, ny, nz, nt, 0.5)
    u = [0.0] * (nx * ny * nz * nt); v = [0.0] * len(u); w = [0.0] * len(u); p = [0.0] * len(u)
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            for k, zz in enumerate(z):
                for n, tt in enumerate(t):
                    e = math.exp(-nu * tt)
                    uu = e * (math.sin(zz) + math.cos(yy))
                    vv = e * (math.sin(xx) + math.cos(zz))
                    ww = e * (math.sin(yy) + math.cos(xx))
                    c = idx(i, j, k, n, shape)
                    u[c], v[c], w[c] = uu, vv, ww
                    p[c] = -0.5 * (uu*uu + vv*vv + ww*ww)
    return FieldData(u, shape, x, y, z, t, {"laplacian_u": nu}, {"u": u, "v": v, "w": w, "p": p}, "u")
