from __future__ import annotations

import cv2
import numpy as np


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip xyxy boxes to image bounds."""
    clipped = np.asarray(boxes, dtype=np.float32).copy()
    if clipped.size == 0:
        return clipped.reshape(-1, 4)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    return clipped


def horizontal_flip(
    image: np.ndarray, boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flip an image horizontally and transform xyxy boxes."""
    height, width = image.shape[:2]
    flipped = cv2.flip(image, 1)
    out = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    if out.size:
        x1 = out[:, 0].copy()
        x2 = out[:, 2].copy()
        out[:, 0] = width - x2
        out[:, 2] = width - x1
        out = clip_boxes(out, width, height)
    return flipped, out


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale image brightness while preserving uint8 output."""
    if factor < 0:
        raise ValueError("factor must be non-negative")
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)


def resize_with_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize to (height, width) and scale xyxy boxes."""
    new_height, new_width = size
    if new_height <= 0 or new_width <= 0:
        raise ValueError("size values must be positive")

    old_height, old_width = image.shape[:2]
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    out = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    if out.size:
        out[:, [0, 2]] *= new_width / old_width
        out[:, [1, 3]] *= new_height / old_height
        out = clip_boxes(out, new_width, new_height)
    return resized, out
