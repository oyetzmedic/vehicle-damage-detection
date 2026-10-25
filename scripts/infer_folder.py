from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from vehicle_damage import LABEL_TO_NAME
from vehicle_damage.model import build_model
from vehicle_damage.visualize import write_annotated_image


def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
    return tensor.float() / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fasterrcnn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device=device, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    image_paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    with torch.inference_mode():
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"skip unreadable: {image_path}")
                continue

            prediction = model([to_tensor(image).to(device)])[0]
            output_path = args.output_dir / image_path.name
            write_annotated_image(
                output_path,
                image,
                prediction["boxes"].detach().cpu().numpy(),
                prediction["labels"].detach().cpu().numpy(),
                prediction["scores"].detach().cpu().numpy(),
                LABEL_TO_NAME,
                threshold=args.threshold,
            )
            print(output_path)


if __name__ == "__main__":
    main()
