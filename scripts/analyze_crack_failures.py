from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.ops import box_iou

from vehicle_damage import LABEL_TO_NAME
from vehicle_damage.dataset import CocoDamageDataset
from vehicle_damage.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/fasterrcnn_epoch5.pt"),
    )
    parser.add_argument("--prediction-floor", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoDamageDataset(args.data_dir, augment=False)

    model = build_model(device=device, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    crack_label = next(
        label for label, name in LABEL_TO_NAME.items() if name == "Crack"
    )

    total_cracks = 0
    same_class_iou50 = 0
    any_class_iou50 = 0

    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            crack_indices = torch.where(target["labels"] == crack_label)[0]
            if len(crack_indices) == 0:
                continue

            image_id = int(target["image_id"])
            filename = dataset.coco.loadImgs([image_id])[0]["file_name"]

            prediction = model([image.to(device)])[0]
            scores = prediction["scores"].detach().cpu()
            keep = scores >= args.prediction_floor
            pred_boxes = prediction["boxes"].detach().cpu()[keep]
            pred_labels = prediction["labels"].detach().cpu()[keep]
            pred_scores = scores[keep]

            for crack_index in crack_indices.tolist():
                total_cracks += 1
                gt_box = target["boxes"][crack_index].unsqueeze(0)

                if len(pred_boxes) == 0:
                    print(
                        f"{filename} crack={total_cracks} "
                        "no predictions above floor"
                    )
                    continue

                ious = box_iou(gt_box, pred_boxes)[0]
                best_any = int(torch.argmax(ious))
                best_any_iou = float(ious[best_any])
                best_any_label = int(pred_labels[best_any])
                best_any_score = float(pred_scores[best_any])

                crack_pred_indices = torch.where(pred_labels == crack_label)[0]
                if len(crack_pred_indices):
                    crack_ious = ious[crack_pred_indices]
                    best_crack_local = int(torch.argmax(crack_ious))
                    best_crack_index = int(crack_pred_indices[best_crack_local])
                    best_crack_iou = float(ious[best_crack_index])
                    best_crack_score = float(pred_scores[best_crack_index])
                else:
                    best_crack_iou = 0.0
                    best_crack_score = 0.0

                if best_any_iou >= 0.5:
                    any_class_iou50 += 1
                if best_crack_iou >= 0.5:
                    same_class_iou50 += 1

                print(
                    f"{filename} "
                    f"best_any={LABEL_TO_NAME.get(best_any_label, str(best_any_label))} "
                    f"score={best_any_score:.3f} iou={best_any_iou:.3f} "
                    f"best_crack_score={best_crack_score:.3f} "
                    f"best_crack_iou={best_crack_iou:.3f}"
                )

    print()
    print(f"crack_ground_truth_objects={total_cracks}")
    print(f"any_class_iou>=0.5={any_class_iou50}")
    print(f"crack_class_iou>=0.5={same_class_iou50}")


if __name__ == "__main__":
    main()
