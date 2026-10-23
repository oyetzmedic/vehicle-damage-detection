from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_damage.routing import review_rate, route_scores, threshold_sweep
from vehicle_damage.transforms import (
    adjust_brightness,
    horizontal_flip,
    resize_with_boxes,
)
from vehicle_damage.visualize import draw_detections, write_annotated_image


def test_horizontal_flip_transforms_xyxy_boxes():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 5, 30, 25]], dtype=np.float32)

    _, flipped = horizontal_flip(image, boxes)

    np.testing.assert_allclose(flipped, [[70, 5, 90, 25]])


def test_horizontal_flip_keeps_boxes_inside_image():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 60, 40], [5, 7, 20, 30]], dtype=np.float32)

    _, flipped = horizontal_flip(image, boxes)

    assert np.all(flipped[:, 0] >= 0)
    assert np.all(flipped[:, 1] >= 0)
    assert np.all(flipped[:, 2] <= 60)
    assert np.all(flipped[:, 3] <= 40)


def test_brightness_preserves_shape_and_dtype():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)

    brighter = adjust_brightness(image, 1.5)

    assert brighter.shape == image.shape
    assert brighter.dtype == np.uint8
    assert int(brighter[0, 0, 0]) == 150


def test_resize_scales_boxes():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array([[20, 10, 100, 60]], dtype=np.float32)

    resized, scaled = resize_with_boxes(image, boxes, (50, 100))

    assert resized.shape == (50, 100, 3)
    np.testing.assert_allclose(scaled, [[10, 5, 50, 30]])


def test_threshold_router_and_review_rate():
    scores = np.array([0.2, 0.5, 0.9])

    routes = route_scores(scores, threshold=0.5)

    assert routes.tolist() == ["needs_review", "auto_accept", "auto_accept"]
    assert review_rate(scores, threshold=0.5) == pytest.approx(1 / 3)


def test_threshold_sweep_counts_review_volume():
    rows = threshold_sweep([0.2, 0.6, 0.9], thresholds=[0.5, 0.8])

    assert rows[0]["needs_review"] == 1
    assert rows[0]["auto_accept"] == 2
    assert rows[1]["needs_review"] == 2
    assert rows[1]["auto_accept"] == 1


def test_drawing_and_write_create_annotated_image(tmp_path: Path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 70, 50]], dtype=np.float32)
    labels = np.array([1])
    scores = np.array([0.95])

    drawn = draw_detections(
        image, boxes, labels, scores, {1: "Dent"}, threshold=0.5
    )
    assert np.any(drawn != image)

    output = tmp_path / "annotated.jpg"
    result = write_annotated_image(
        output, image, boxes, labels, scores, {1: "Dent"}, threshold=0.5
    )

    assert result == output
    assert output.exists()
    reloaded = cv2.imread(str(output))
    assert reloaded is not None
