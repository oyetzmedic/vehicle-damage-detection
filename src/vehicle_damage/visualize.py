from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    boxes,
    labels,
    scores,
    class_names: dict[int, str],
    threshold: float = 0.5,
) -> np.ndarray:
    """Draw prediction boxes, class names and confidence scores."""
    canvas = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        score = float(score)
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        name = class_names.get(int(label), f"class_{int(label)}")
        text = f"{name} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            canvas,
            text,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_annotated_image(
    path: str | Path,
    image: np.ndarray,
    boxes,
    labels,
    scores,
    class_names: dict[int, str],
    threshold: float = 0.5,
) -> Path:
    """Draw detections and write an annotated image."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(
        image, boxes, labels, scores, class_names, threshold=threshold
    )
    if not cv2.imwrite(str(output), annotated):
        raise OSError(f"Could not write image to {output}")
    return output
