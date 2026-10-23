from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from . import TARGET_CLASSES
from .transforms import adjust_brightness, horizontal_flip


class CocoDamageDataset(Dataset):
    """COCO-format dataset filtered to the four vehicle-damage classes."""

    def __init__(
        self,
        split_dir: str | Path,
        augment: bool = False,
        max_images: int | None = None,
        seed: int = 42,
    ):
        self.split_dir = Path(split_dir)
        self.annotation_file = self.split_dir / "_annotations.coco.json"
        if not self.annotation_file.exists():
            raise FileNotFoundError(self.annotation_file)

        self.coco = COCO(str(self.annotation_file))
        categories = self.coco.loadCats(self.coco.getCatIds())
        name_to_category_id = {cat["name"]: cat["id"] for cat in categories}

        missing = [name for name in TARGET_CLASSES if name not in name_to_category_id]
        if missing:
            raise ValueError(f"Missing target categories: {missing}")

        self.category_to_label = {
            name_to_category_id[name]: index + 1
            for index, name in enumerate(TARGET_CLASSES)
        }

        image_ids = []
        for image_id in self.coco.getImgIds():
            annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
            annotations = self.coco.loadAnns(annotation_ids)
            if any(ann["category_id"] in self.category_to_label for ann in annotations):
                image_ids.append(image_id)

        if max_images is not None and len(image_ids) > max_images:
            rng = np.random.default_rng(seed)
            image_ids = sorted(rng.choice(image_ids, size=max_images, replace=False).tolist())

        self.image_ids = image_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        path = self.split_dir / info["file_name"]

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        boxes = []
        labels = []
        areas = []
        crowds = []

        for ann in annotations:
            if ann["category_id"] not in self.category_to_label:
                continue
            x, y, width, height = ann["bbox"]
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_to_label[ann["category_id"]])
            areas.append(float(ann.get("area", width * height)))
            crowds.append(int(ann.get("iscrowd", 0)))

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

        if self.augment:
            if np.random.random() < 0.5:
                image, boxes_np = horizontal_flip(image, boxes_np)
            image = adjust_brightness(image, float(np.random.uniform(0.8, 1.2)))

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0

        target = {
            "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
