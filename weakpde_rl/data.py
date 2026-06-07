from __future__ import annotations

from dataclasses import dataclass
import math
import random

Field4D = list[float]


@dataclass(frozen=True)
class FieldData:
    """Regular-grid scalar spatiotemporal field data stored as a flat 4-D array."""

    u: Field4D
    shape: tuple[int, int, int, int]
    x: list[float]
    y: list[float]
    z: list[float]
    t: list[float]
    true_coefficients: dict[str, float]

    @property
    def spacing(self) -> tuple[float, float, float, float]:
        return (self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0], self.t[1] - self.t[0])


def idx(i: int, j: int, k: int, n: int, shape: tuple[int, int, int, int]) -> int:
    nx, ny, nz, nt = shape
    return (((i % nx) * ny + (j % ny)) * nz + (k % nz)) * nt + n


def make_3d_reaction_advection_diffusion(
    nx: int = 18,
    ny: int = 16,
    nz: int = 14,
    nt: int = 11,
    noise: float = 0.0,
    seed: int = 7,
    coeffs: dict[str, float] | None = None,
) -> FieldData:
    """Create an analytic 3-D field for

    u_t = alpha*u - cx*u_x - cy*u_y - cz*u_z + nu*(u_xx + u_yy + u_zz).
    """

    params = {"u": -0.15, "u_x": -0.80, "u_y": 0.45, "u_z": -0.30, "laplacian": 0.06}
    if coeffs:
        params.update(coeffs)
    x = [2.0 * math.pi * i / nx for i in range(nx)]
    y = [2.0 * math.pi * i / ny for i in range(ny)]
    z = [2.0 * math.pi * i / nz for i in range(nz)]
    t = [0.8 * i / (nt - 1) for i in range(nt)]
    cx, cy, cz = -params["u_x"], -params["u_y"], -params["u_z"]
    alpha, nu = params["u"], params["laplacian"]
    modes = [(1.0, 1, 1, 1, 0.2), (0.35, 2, 1, 1, -0.7), (0.25, 1, 2, 1, 0.6)]
    shape = (nx, ny, nz, nt)
    values = [0.0] * (nx * ny * nz * nt)
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            for k, zz in enumerate(z):
                for n, tt in enumerate(t):
                    value = 0.0
                    for amp, kx, ky, kz, phase in modes:
                        k2 = kx * kx + ky * ky + kz * kz
                        theta = kx * (xx - cx * tt) + ky * (yy - cy * tt) + kz * (zz - cz * tt) + phase
                        value += amp * math.exp((alpha - nu * k2) * tt) * math.sin(theta)
                    values[idx(i, j, k, n, shape)] = value
    if noise:
        rng = random.Random(seed)
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        values = [v + noise * std * rng.gauss(0.0, 1.0) for v in values]
    return FieldData(values, shape, x, y, z, t, params)
