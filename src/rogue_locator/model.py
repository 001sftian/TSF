"""Transformer model for sparse-context rogue-wave detection and localization."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Deterministic sinusoidal coordinate embedding."""

    def __init__(self, input_dim: int = 2, num_bands: int = 8) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_bands, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_dim = input_dim * num_bands * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        angles = coords[..., None, :] * self.frequencies[:, None]
        angles = angles.flatten(start_dim=-2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SparseRogueWaveTransformer(nn.Module):
    """Infer high-order rogue-wave class and dense location heatmap.

    The model encodes sparse small-window observations as tokens, then decodes a
    heatmap on arbitrary large-domain query coordinates. This makes inference
    resolution independent: pass a denser or wider query grid to localize over a
    larger area without changing the learned parameters.
    """

    def __init__(
        self,
        *,
        max_order: int = 5,
        d_model: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        fourier_bands: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_order = max_order
        self.coord_features = FourierFeatures(2, fourier_bands)
        token_dim = self.coord_features.output_dim + 1
        self.context_projection = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(self.coord_features.output_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.heatmap_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.order_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_order),
        )

    def forward(
        self,
        context_coords: torch.Tensor,
        context_values: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the detector.

        Args:
            context_coords: Sparse coordinates, shape ``[B, P, 2]``.
            context_values: Sparse amplitudes, shape ``[B, P, 1]``.
            query_coords: Dense localization coordinates, shape ``[B, Q, 2]``.

        Returns:
            ``heatmap_logits`` with shape ``[B, Q, 1]`` and ``order_logits``
            with shape ``[B, max_order]``.
        """
        context_features = torch.cat([self.coord_features(context_coords), context_values], dim=-1)
        memory = self.encoder(self.context_projection(context_features))
        queries = self.query_projection(self.coord_features(query_coords))
        decoded = self.decoder(queries, memory)
        pooled = memory.mean(dim=1)
        return {
            "heatmap_logits": self.heatmap_head(decoded),
            "order_logits": self.order_head(pooled),
        }


def estimate_center(query_coords: torch.Tensor, heatmap_logits: torch.Tensor) -> torch.Tensor:
    """Return soft-argmax centers from dense heatmap logits."""
    weights = torch.softmax(heatmap_logits.squeeze(-1), dim=-1)
    return torch.sum(weights[..., None] * query_coords, dim=1)
