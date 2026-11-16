"""Classifier construction. Plain torch/torchvision only (no Lightning)."""
from __future__ import annotations

import torch
from torch import nn

from robust_gcf.core.schemas import PreprocessSpec


def _densenet121(pretrained: bool) -> nn.Module:
    from torchvision import models

    try:  # torchvision >= 0.13
        from torchvision.models import DenseNet121_Weights

        return models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    except ImportError:  # older torchvision (some CF-method envs)
        return models.densenet121(pretrained=pretrained)


def _densenet121_backbone(in_channels: int, pretrained: bool) -> nn.Module:
    net = _densenet121(pretrained)
    if in_channels == 1:
        # Grayscale: sum the pretrained RGB filters into one input channel.
        old = net.features.conv0
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        net.features.conv0 = new
    elif in_channels != 3:
        raise ValueError(f"unsupported in_channels={in_channels}")
    net.classifier = nn.Linear(net.classifier.in_features, 1)  # single positive-class logit
    return net


ARCHS = {"densenet121": _densenet121_backbone}


def build_backbone(arch: str = "densenet121", in_channels: int = 3, pretrained: bool = True) -> nn.Module:
    """Backbone mapping a *normalized* (B, C, S, S) tensor to (B, 1) logits."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; available: {sorted(ARCHS)}")
    return ARCHS[arch](in_channels, pretrained)


def build_classifier(arch: str = "densenet121", preprocess: PreprocessSpec = None,
                     pretrained: bool = True, threshold: float = 0.5):
    from robust_gcf.models.wrapper import ClassifierWrapper

    preprocess = preprocess or PreprocessSpec.default_for(3)
    backbone = build_backbone(arch, preprocess.channels, pretrained)
    return ClassifierWrapper(backbone, preprocess, threshold)
