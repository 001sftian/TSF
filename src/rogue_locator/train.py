"""Training CLI for the sparse rogue-wave Transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import SyntheticRogueWaveDataset
from .model import SparseRogueWaveTransformer, estimate_center


def _tensor(array, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = SyntheticRogueWaveDataset(
        batch_size=args.batch_size,
        seed=args.seed,
        grid_size=(args.grid_size, args.grid_size),
        context_points=args.context_points,
    )
    model = SparseRogueWaveTransformer(max_order=args.max_order).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step, batch in zip(range(1, args.steps + 1), dataset):
        context_coords = _tensor(batch.context_coords, device)
        context_values = _tensor(batch.context_values, device)
        grid_coords = _tensor(batch.grid_coords, device)
        heatmap = _tensor(batch.heatmap, device)
        order_labels = _tensor(batch.order_labels, device)
        true_centers = _tensor(batch.centers, device)

        outputs = model(context_coords, context_values, grid_coords)
        heatmap_loss = F.binary_cross_entropy_with_logits(outputs["heatmap_logits"], heatmap)
        order_loss = F.cross_entropy(outputs["order_logits"], order_labels)
        predicted_centers = estimate_center(grid_coords, outputs["heatmap_logits"])
        center_loss = F.smooth_l1_loss(predicted_centers, true_centers)
        loss = args.heatmap_weight * heatmap_loss + args.order_weight * order_loss + args.center_weight * center_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            predicted_order = outputs["order_logits"].argmax(dim=-1)
            acc = (predicted_order == order_labels).float().mean().item()
            print(
                f"step={step:05d} loss={loss.item():.4f} heatmap={heatmap_loss.item():.4f} "
                f"order={order_loss.item():.4f} center={center_loss.item():.4f} order_acc={acc:.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.output)
    print(f"saved checkpoint to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--context-points", type=int, default=256)
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--order-weight", type=float, default=0.4)
    parser.add_argument("--center-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/rogue_transformer.pt"))
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
