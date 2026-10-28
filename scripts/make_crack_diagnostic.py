from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from vehicle_damage import LABEL_TO_NAME
from vehicle_damage.dataset import CocoDamageDataset
from vehicle_damage.model import build_model


GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 165, 255)


def draw_box(
    image: np.ndarray,
    box: torch.Tensor,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        default="damage_4715_jpg.rf.604b87b7edd1126e5c62010101b9b66d.jpg",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/test"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/fasterrcnn_epoch5.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/portfolio_examples/crack_diagnostic_clean.jpg"),
    )
    parser.add_argument("--prediction-floor", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoDamageDataset(args.data_dir, augment=False)

    target_index = None
    for index, image_id in enumerate(dataset.image_ids):
        info = dataset.coco.loadImgs([image_id])[0]
        if info["file_name"] == args.filename:
            target_index = index
            break

    if target_index is None:
        raise FileNotFoundError(f"Image not found in dataset: {args.filename}")

    image_tensor, target = dataset[target_index]
    image_id = int(target["image_id"])

    crack_label = next(
        label for label, name in LABEL_TO_NAME.items() if name == "Crack"
    )
    crack_gt_indices = torch.where(target["labels"] == crack_label)[0]
    if len(crack_gt_indices) == 0:
        raise RuntimeError("Selected image has no Crack ground-truth object.")

    model = build_model(device=device, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.inference_mode():
        prediction = model([image_tensor.to(device)])[0]

    scores = prediction["scores"].detach().cpu()
    keep = scores >= args.prediction_floor
    pred_boxes = prediction["boxes"].detach().cpu()[keep]
    pred_labels = prediction["labels"].detach().cpu()[keep]
    pred_scores = scores[keep]

    crack_pred_indices = torch.where(pred_labels == crack_label)[0]
    if len(crack_pred_indices) == 0:
        raise RuntimeError(
            f"No Crack prediction found above score floor {args.prediction_floor}."
        )

    best_gt = None
    best_pred = None
    best_score = None
    best_iou = -1.0

    for gt_index in crack_gt_indices.tolist():
        gt_box = target["boxes"][gt_index].unsqueeze(0)
        crack_boxes = pred_boxes[crack_pred_indices]
        ious = box_iou(gt_box, crack_boxes)[0]
        local_index = int(torch.argmax(ious))
        iou = float(ious[local_index])
        pred_index = int(crack_pred_indices[local_index])

        if iou > best_iou:
            best_iou = iou
            best_gt = target["boxes"][gt_index]
            best_pred = pred_boxes[pred_index]
            best_score = float(pred_scores[pred_index])

    info = dataset.coco.loadImgs([image_id])[0]
    image_path = args.data_dir / info["file_name"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    draw_box(image, best_gt, GT_COLOR, thickness=3)
    draw_box(image, best_pred, PRED_COLOR, thickness=2)

    width = image.shape[1]
    header = np.zeros((104, width, 3), dtype=np.uint8)
    footer = np.zeros((42, width, 3), dtype=np.uint8)

    cv2.putText(
        header,
        "Crack failure analysis",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(header, (10, 39), (30, 55), GT_COLOR, -1)
    cv2.putText(
        header,
        "Ground truth: Crack",
        (40, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(header, (10, 67), (30, 83), PRED_COLOR, -1)
    cv2.putText(
        header,
        f"Prediction: Crack | confidence={best_score:.3f} | IoU={best_iou:.3f}",
        (40, 81),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        footer,
        "Localization is strong, but class confidence is low.",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    canvas = np.vstack([header, image, footer])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Failed to write {args.output}")

    print(f"filename={args.filename}")
    print(f"prediction_score={best_score:.3f}")
    print(f"iou={best_iou:.3f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
