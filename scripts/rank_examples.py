from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torchvision.ops import box_iou

from vehicle_damage import LABEL_TO_NAME, TARGET_CLASSES
from vehicle_damage.dataset import CocoDamageDataset
from vehicle_damage.model import build_model


def score_image(prediction, target, score_threshold: float, iou_threshold: float):
    scores = prediction["scores"].detach().cpu()
    keep = scores >= score_threshold

    pred_boxes = prediction["boxes"].detach().cpu()[keep]
    pred_labels = prediction["labels"].detach().cpu()[keep]
    pred_scores = scores[keep]

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    used_gt: set[int] = set()
    matched_scores: list[float] = []
    matched_classes: list[str] = []

    for pred_index in range(len(pred_boxes)):
        same_class = torch.where(gt_labels == pred_labels[pred_index])[0]
        available = [int(index) for index in same_class if int(index) not in used_gt]
        if not available:
            continue

        candidate_boxes = gt_boxes[available]
        ious = box_iou(pred_boxes[pred_index].unsqueeze(0), candidate_boxes)[0]
        best_local = int(torch.argmax(ious))
        best_iou = float(ious[best_local])

        if best_iou >= iou_threshold:
            gt_index = available[best_local]
            used_gt.add(gt_index)
            matched_scores.append(float(pred_scores[pred_index]))
            matched_classes.append(LABEL_TO_NAME[int(pred_labels[pred_index])])

    tp = len(used_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "max_tp_score": max(matched_scores, default=0.0),
        "matched_classes": ";".join(sorted(set(matched_classes))),
        "_matched_scores": matched_scores,
        "_matched_classes": matched_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/fasterrcnn_epoch5.pt"),
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=Path("outputs/test_epoch5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/example_ranking.csv"),
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoDamageDataset(args.data_dir, augment=False)

    model = build_model(device=device, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    best_by_class = {name: None for name in TARGET_CLASSES}

    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            image_id = int(target["image_id"])
            filename = dataset.coco.loadImgs([image_id])[0]["file_name"]

            prediction = model([image.to(device)])[0]
            scored = score_image(
                prediction,
                target,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )

            row = {
                "filename": filename,
                "annotated_path": str(args.annotated_dir / filename),
                "gt_count": scored["gt_count"],
                "pred_count": scored["pred_count"],
                "tp": scored["tp"],
                "fp": scored["fp"],
                "fn": scored["fn"],
                "precision": scored["precision"],
                "recall": scored["recall"],
                "f1": scored["f1"],
                "max_tp_score": scored["max_tp_score"],
                "matched_classes": scored["matched_classes"],
            }
            rows.append(row)

            for class_name, matched_score in zip(
                scored["_matched_classes"], scored["_matched_scores"]
            ):
                current = best_by_class[class_name]
                if current is None or matched_score > current["score"]:
                    best_by_class[class_name] = {
                        "filename": filename,
                        "score": matched_score,
                        "f1": scored["f1"],
                        "tp": scored["tp"],
                    }

    rows.sort(
        key=lambda row: (
            row["f1"],
            row["tp"],
            row["max_tp_score"],
            -row["fp"],
        ),
        reverse=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename",
        "annotated_path",
        "gt_count",
        "pred_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "max_tp_score",
        "matched_classes",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"score_threshold={args.score_threshold} "
        f"iou_threshold={args.iou_threshold} "
        f"images={len(rows)}"
    )
    print("\nTop overall examples:")
    for row in rows[: args.top]:
        print(
            f"{row['filename']} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']} "
            f"f1={row['f1']:.3f} max_tp_score={row['max_tp_score']:.3f} "
            f"classes={row['matched_classes'] or '-'}"
        )

    print("\nBest matched example per class:")
    for class_name in TARGET_CLASSES:
        item = best_by_class[class_name]
        if item is None:
            print(f"{class_name}: no matched detection at these thresholds")
        else:
            print(
                f"{class_name}: {item['filename']} "
                f"score={item['score']:.3f} f1={item['f1']:.3f} tp={item['tp']}"
            )

    print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
