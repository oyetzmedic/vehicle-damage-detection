from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from vehicle_damage import LABEL_TO_NAME
from vehicle_damage.dataset import CocoDamageDataset, collate_fn
from vehicle_damage.model import build_model
from vehicle_damage.routing import threshold_sweep


def scalar(value):
    if hasattr(value, "numel") and value.numel() == 1:
        return float(value.detach().cpu())
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fasterrcnn.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--candidate-floor", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoDamageDataset(args.data_dir / "test", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = build_model(device=device, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )
    candidate_scores = []

    with torch.inference_mode():
        for images, targets in loader:
            images_device = [image.to(device) for image in images]
            predictions = model(images_device)

            predictions_cpu = [
                {key: value.detach().cpu() for key, value in prediction.items()}
                for prediction in predictions
            ]
            targets_cpu = [
                {key: value.detach().cpu() for key, value in target.items()}
                for target in targets
            ]
            metric.update(predictions_cpu, targets_cpu)

            for prediction in predictions_cpu:
                scores = prediction["scores"]
                candidate_scores.extend(
                    float(score)
                    for score in scores[scores >= args.candidate_floor].tolist()
                )

    raw = metric.compute()
    result = {
        "map": scalar(raw["map"]),
        "map_50": scalar(raw["map_50"]),
        "map_75": scalar(raw["map_75"]),
    }

    classes = raw.get("classes")
    per_class = raw.get("map_per_class")
    if classes is not None and per_class is not None:
        result["per_class_ap"] = {
            LABEL_TO_NAME.get(int(class_id), str(int(class_id))): float(ap)
            for class_id, ap in zip(classes.tolist(), per_class.tolist())
        }

    result["candidate_floor"] = args.candidate_floor
    result["threshold_sweep"] = threshold_sweep(
        candidate_scores,
        thresholds=(0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
