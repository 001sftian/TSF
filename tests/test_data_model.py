from __future__ import annotations

import importlib.util

import pytest

np = pytest.importorskip("numpy")

from rogue_locator.data import generate_batch


def test_generate_batch_shapes_and_order_labels() -> None:
    batch = generate_batch(3, grid_size=(16, 20), context_points=11, seed=1)

    assert batch.context_coords.shape == (3, 11, 2)
    assert batch.context_values.shape == (3, 11, 1)
    assert batch.grid_coords.shape == (3, 16 * 20, 2)
    assert batch.heatmap.shape == (3, 16 * 20, 1)
    assert batch.order_labels.shape == (3,)
    assert np.all((0 <= batch.order_labels) & (batch.order_labels < 5))
    assert np.all(batch.heatmap >= 0.0)
    assert np.all(batch.heatmap <= 1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_transformer_forward_shapes() -> None:
    import torch

    from rogue_locator.model import SparseRogueWaveTransformer, estimate_center

    batch = generate_batch(2, grid_size=(8, 8), context_points=9, seed=2)
    model = SparseRogueWaveTransformer(d_model=32, num_heads=4, encoder_layers=1, decoder_layers=1)
    outputs = model(
        torch.as_tensor(batch.context_coords),
        torch.as_tensor(batch.context_values),
        torch.as_tensor(batch.grid_coords),
    )

    assert outputs["heatmap_logits"].shape == (2, 64, 1)
    assert outputs["order_logits"].shape == (2, 5)
    assert estimate_center(torch.as_tensor(batch.grid_coords), outputs["heatmap_logits"]).shape == (2, 2)
