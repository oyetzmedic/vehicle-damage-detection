from __future__ import annotations

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.ops.misc import FrozenBatchNorm2d

from . import TARGET_CLASSES


def _set_trainable_backbone_layers(model: FasterRCNN, trainable_layers: int) -> None:
    """Freeze ResNet body layers to match torchvision's transfer-learning policy."""
    if not 0 <= trainable_layers <= 5:
        raise ValueError("trainable_backbone_layers must be between 0 and 5")

    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(any(name.startswith(layer) for layer in layers_to_train))


def build_model(
    num_damage_classes: int = len(TARGET_CLASSES),
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 768,
    device: torch.device | str = "cpu",
    pretrained: bool = True,
):
    """Build Faster R-CNN using a low-memory weight-loading path.

    The pretrained path moves the model to the target device first and maps
    the official COCO checkpoint directly to that device. Evaluation and
    inference can set pretrained=False and load only the project checkpoint.
    """
    device = torch.device(device)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=5,
    )

    initial_classes = (
        len(weights.meta["categories"]) if pretrained else num_damage_classes + 1
    )
    model = FasterRCNN(
        backbone,
        num_classes=initial_classes,
        min_size=min_size,
        max_size=max_size,
    ).to(device)

    if pretrained:
        state_dict = weights.get_state_dict(
            progress=False,
            check_hash=True,
            map_location=device,
        )
        model.load_state_dict(state_dict)
        del state_dict

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_damage_classes + 1
        ).to(device)

    _set_trainable_backbone_layers(model, trainable_backbone_layers)
    return model
