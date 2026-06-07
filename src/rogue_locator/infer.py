"""Inference CLI for a trained rogue-wave locator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import generate_batch
from .model import SparseRogueWaveTransformer, estimate_center


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SparseRogueWaveTransformer(max_order=args.max_order).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Demo inference uses a synthetic sparse observation. Replace this block with
    # measured sparse coordinates and amplitudes from sensors or experiments.
    batch = generate_batch(
        1,
        grid_size=(args.grid_size, args.grid_size),
        context_points=args.context_points,
        max_order=args.max_order,
        seed=args.seed,
    )
    with torch.no_grad():
        context_coords = torch.as_tensor(batch.context_coords, device=device)
        context_values = torch.as_tensor(batch.context_values, device=device)
        grid_coords = torch.as_tensor(batch.grid_coords, device=device)
        outputs = model(context_coords, context_values, grid_coords)
        center = estimate_center(grid_coords, outputs["heatmap_logits"])[0].cpu().numpy()
        order = int(outputs["order_logits"].argmax(dim=-1).item()) + 1
        probability = torch.softmax(outputs["order_logits"], dim=-1).max().item()

    print(f"predicted_order={order} order_confidence={probability:.3f}")
    print(f"predicted_center_x={center[0]:.3f} predicted_center_t={center[1]:.3f}")
    print(f"true_order={int(batch.order_labels[0]) + 1} true_center={np.round(batch.centers[0], 3).tolist()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--context-points", type=int, default=256)
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    infer(build_parser().parse_args())


if __name__ == "__main__":
    main()
