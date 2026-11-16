"""The one classifier interface every CF method sees.

    ClassifierWrapper(x) -> logit of the positive class, shape (B,)
    x: float tensor in [0, 1], (B, C, H, W), any square resolution (resized to input_size).

Resize + normalization happen inside the module, so they are differentiable and identical
for white-box methods (gradients w.r.t. x), black-box methods, and evaluation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from robust_gcf.core.schemas import PreprocessSpec


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, preprocess: PreprocessSpec, threshold: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.preprocess = preprocess
        self.threshold = float(threshold)
        c = preprocess.channels
        self.register_buffer("mean", torch.tensor(preprocess.mean, dtype=torch.float32).view(1, c, 1, 1))
        self.register_buffer("std", torch.tensor(preprocess.std, dtype=torch.float32).view(1, c, 1, 1))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        s = self.preprocess.input_size
        if x.shape[-2:] == (s, s):
            return x  # no-op path: cached data is stored at 128, so old torch is fine here
        try:
            return F.interpolate(x, size=(s, s), mode=self.preprocess.interpolation,
                                 align_corners=False, antialias=self.preprocess.antialias)
        except TypeError as e:  # torch < 1.11 has no `antialias`
            raise RuntimeError("resizing inside the classifier needs torch >= 1.11; "
                               "feed images at {0}x{0} or upgrade torch".format(s)) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.preprocess.channels:
            raise ValueError(f"expected (B, {self.preprocess.channels}, H, W) in [0,1], got {tuple(x.shape)}")
        x = (self._resize(x) - self.mean) / self.std
        return self.backbone(x).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(x) >= self.threshold).long()

    def freeze(self) -> "ClassifierWrapper":
        """Eval mode (BatchNorm!) and no parameter grads; input gradients still flow."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


# ----------------------------------------------------------------------------- adapters
# Small views for methods whose upstream code expects a different convention.


class FromMinusOneOne(nn.Module):
    """Accept images in [-1, 1] (DiME / guided-diffusion, GCD) instead of [0, 1]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x + 1.0) * 0.5)


class TwoClassLogits(nn.Module):
    """(B,) logit z -> (B, 2) logits [0, z]: softmax(...)[:, 1] == sigmoid(z) exactly.

    Column 1 is the positive (CF source) class, column 0 the negative (CF target) class.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model(x)
        return torch.stack([torch.zeros_like(z), z], dim=1)
