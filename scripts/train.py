from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vehicle_damage import TARGET_CLASSES
from vehicle_damage.dataset import CocoDamageDataset, collate_fn
from vehicle_damage.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fasterrcnn.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = CocoDamageDataset(args.data_dir / "train", augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_model(device=device, pretrained=True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=0.0005
    )

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            losses = model(images, targets)
            loss = sum(loss_value for loss_value in losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        mean_loss = running_loss / max(1, len(loader))
        print(f"epoch={epoch + 1}/{args.epochs} mean_loss={mean_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(TARGET_CLASSES),
            "epochs": args.epochs,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
