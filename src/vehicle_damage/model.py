from __future__ import annotations

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import TARGET_CLASSES


def build_model(
    num_damage_classes: int = len(TARGET_CLASSES),
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 768,
):
    """Build a COCO-pretrained Faster R-CNN with a new classifier head.

    The default 512-768 px transform is a low-memory profile chosen for
    4 GB GPUs such as the GTX 1050 Ti used for this project.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_damage_classes + 1
    )
    return model
