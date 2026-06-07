"""Synthetic sparse-observation data for rogue-wave localization.

The generator intentionally keeps the numerical model lightweight. It creates
Peregrine-like rational wave packets whose peak width shrinks and peak height
increases with the rogue-wave order. For research use, replace this module with
solutions produced by the focusing nonlinear Schrödinger equation, experiments,
or a validated CFD/wave-tank simulator while preserving the same batch schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class RogueWaveBatch:
    """Batch returned by the synthetic generator.

    Attributes:
        context_coords: Observed sparse coordinates with shape ``[B, P, 2]``.
        context_values: Observed wave amplitudes with shape ``[B, P, 1]``.
        grid_coords: Dense target coordinates with shape ``[B, H*W, 2]``.
        heatmap: Dense normalized localization target with shape ``[B, H*W, 1]``.
        order_labels: Integer labels in ``[0, max_order - 1]`` with shape ``[B]``.
        centers: True rogue-wave centers in normalized coordinates, shape ``[B, 2]``.
    """

    context_coords: np.ndarray
    context_values: np.ndarray
    grid_coords: np.ndarray
    heatmap: np.ndarray
    order_labels: np.ndarray
    centers: np.ndarray


def _make_grid(height: int, width: int, extent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-extent, extent, width, dtype=np.float32)
    t = np.linspace(-extent, extent, height, dtype=np.float32)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    coords = np.stack([xx, tt], axis=-1).reshape(-1, 2)
    return xx, tt, coords


def _rogue_profile(
    xx: np.ndarray,
    tt: np.ndarray,
    *,
    order: int,
    center: np.ndarray,
    phase: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth surrogate for an order-N rogue-wave amplitude field."""
    width = 0.75 / np.sqrt(order)
    dx = (xx - center[0]) / width
    dt = (tt - center[1]) / width
    radius2 = dx**2 + dt**2

    # Peregrine-like peak: first order peaks near 3x background; higher orders
    # peak near (2N+1)x background with narrower support.
    background = 1.0 + 0.05 * rng.normal(size=xx.shape).astype(np.float32)
    rational_peak = 1.0 + (2.0 * order) / (1.0 + radius2)

    # Add weak satellite lobes for N >= 2 so the model learns richer
    # high-order geometry rather than only the maximum value.
    satellites = np.zeros_like(xx, dtype=np.float32)
    if order >= 2:
        for k in range(order):
            angle = phase + 2.0 * np.pi * k / order
            sx = center[0] + width * 1.35 * np.cos(angle)
            st = center[1] + width * 1.35 * np.sin(angle)
            satellites += 0.28 * order * np.exp(-((xx - sx) ** 2 + (tt - st) ** 2) / (2.0 * width**2))

    field = background * rational_peak + satellites
    field = field.astype(np.float32)
    heatmap_sigma = width * 0.45
    heatmap = np.exp(-((xx - center[0]) ** 2 + (tt - center[1]) ** 2) / (2.0 * heatmap_sigma**2))
    return field, heatmap.astype(np.float32)


def generate_batch(
    batch_size: int,
    *,
    grid_size: tuple[int, int] = (64, 64),
    context_points: int = 256,
    context_extent: float = 1.0,
    full_extent: float = 4.0,
    min_order: int = 1,
    max_order: int = 5,
    seed: int | None = None,
) -> RogueWaveBatch:
    """Generate sparse-context observations and dense localization targets.

    The context points are sampled only from a small window around the origin,
    simulating the user's setting: sparse first-order-like local measurements are
    used to infer the order and position of rogue events over a much larger area.
    """
    if min_order < 1 or max_order < min_order:
        raise ValueError("orders must satisfy 1 <= min_order <= max_order")
    if context_extent >= full_extent:
        raise ValueError("context_extent should be smaller than full_extent")

    rng = np.random.default_rng(seed)
    height, width = grid_size
    xx, tt, dense_coords = _make_grid(height, width, full_extent)

    context_coords = np.empty((batch_size, context_points, 2), dtype=np.float32)
    context_values = np.empty((batch_size, context_points, 1), dtype=np.float32)
    grid_coords = np.broadcast_to(dense_coords, (batch_size, dense_coords.shape[0], 2)).copy()
    heatmaps = np.empty((batch_size, dense_coords.shape[0], 1), dtype=np.float32)
    order_labels = np.empty((batch_size,), dtype=np.int64)
    centers = np.empty((batch_size, 2), dtype=np.float32)

    for item in range(batch_size):
        order = int(rng.integers(min_order, max_order + 1))
        center = rng.uniform(-full_extent * 0.72, full_extent * 0.72, size=2).astype(np.float32)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        field, target_heatmap = _rogue_profile(xx, tt, order=order, center=center, phase=phase, rng=rng)

        sampled = rng.uniform(-context_extent, context_extent, size=(context_points, 2)).astype(np.float32)
        ix = np.clip(np.searchsorted(np.linspace(-full_extent, full_extent, width), sampled[:, 0]), 0, width - 1)
        it = np.clip(np.searchsorted(np.linspace(-full_extent, full_extent, height), sampled[:, 1]), 0, height - 1)
        values = field[it, ix] + rng.normal(0.0, 0.03, size=context_points).astype(np.float32)

        context_coords[item] = sampled
        context_values[item, :, 0] = values
        heatmaps[item, :, 0] = target_heatmap.reshape(-1)
        order_labels[item] = order - 1
        centers[item] = center

    return RogueWaveBatch(context_coords, context_values, grid_coords, heatmaps, order_labels, centers)


class SyntheticRogueWaveDataset:
    """Infinite iterable dataset for online training."""

    def __init__(self, *, batch_size: int = 8, seed: int | None = None, **batch_kwargs: object) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.batch_kwargs = batch_kwargs
        self._step = 0

    def __iter__(self) -> Iterator[RogueWaveBatch]:
        while True:
            seed = None if self.seed is None else self.seed + self._step
            self._step += 1
            yield generate_batch(self.batch_size, seed=seed, **self.batch_kwargs)
