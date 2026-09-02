"""Backbone construction for binary loosening classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

# Each entry maps a CLI name to (torchvision constructor, weights enum,
# attribute holding the classifier head).
SUPPORTED = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "classifier"),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "classifier",
    ),
    "efficientnet_b3": (
        models.efficientnet_b3,
        models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "classifier",
    ),
    "convnext_tiny": (
        models.convnext_tiny,
        models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "classifier",
    ),
}


def build_model(arch: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Return a backbone whose head emits a single logit (loose vs. control)."""
    if arch not in SUPPORTED:
        raise SystemExit(f"Unknown --arch {arch!r}. Choose from: {', '.join(SUPPORTED)}")
    constructor, weights, head_name = SUPPORTED[arch]
    model = constructor(weights=weights if pretrained else None)
    head = getattr(model, head_name)
    if isinstance(head, nn.Sequential):
        # densenet/efficientnet/convnext keep pooling or flattening layers in the
        # head, so only the trailing Linear is replaced.
        linear_index = max(i for i, layer in enumerate(head) if isinstance(layer, nn.Linear))
        head[linear_index] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head[linear_index].in_features, 1)
        )
    else:
        setattr(model, head_name, nn.Sequential(nn.Dropout(dropout), nn.Linear(head.in_features, 1)))
    model.arch = arch
    return model


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    _, _, head_name = SUPPORTED[model.arch]
    return list(getattr(model, head_name).parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    head_ids = {id(p) for p in head_parameters(model)}
    for parameter in model.parameters():
        if id(parameter) not in head_ids:
            parameter.requires_grad_(trainable)


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Deepest convolution, used as the Grad-CAM target layer."""
    conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv = module
    if conv is None:
        raise SystemExit("No Conv2d layer found for Grad-CAM")
    return conv
