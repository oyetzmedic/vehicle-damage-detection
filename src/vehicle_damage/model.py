from __future__ import annotations

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
):
    """Build a COCO-pretrained Faster R-CNN with a new classifier head.

    The model is assembled explicitly before loading the official torchvision
    COCO state dict. This avoids a Windows access violation observed on the
    reference machine in the convenience pretrained-weight constructor while
    preserving the pretrained detector weights and FrozenBatchNorm backbone.

    The default 512-768 px transform is a low-memory profile for 4 GB GPUs.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=5,
    )
    model = FasterRCNN(
        backbone,
        num_classes=len(weights.meta["categories"]),
        min_size=min_size,
        max_size=max_size,
    )
    model.load_state_dict(
        weights.get_state_dict(progress=False, check_hash=True)
    )
    _set_trainable_backbone_layers(model, trainable_backbone_layers)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_damage_classes + 1
    )
    return model
