"""Geometric preprocessing (applied once, when building the cache) and train augmentation.

Classifier-side resize/normalize is NOT here: it lives inside `ClassifierWrapper`, so every
CF method feeds the classifier the same [0,1] images and gets identical preprocessing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

CROP_POLICIES = ("center", "none")


def crop_and_resize(img: Image.Image, size: int, policy: str = "center") -> Image.Image:
    """Produce a `size` x `size` image.

    "center": resize the shorter side to `size` (bilinear, PIL antialiasing), then center-crop.
              Identical to torchvision `Resize(size)` + `CenterCrop(size)` on PIL images.
              [check] confirm this matches DiME's CelebA preprocessing (setup ToDo 1).
    "none":   input must already be square; resize only if needed.
    """
    w, h = img.size
    if policy == "center":
        if w <= h:
            nw, nh = size, int(size * h / w)
        else:
            nw, nh = int(size * w / h), size
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.BILINEAR)
        left = int(round((nw - size) / 2.0))
        top = int(round((nh - size) / 2.0))
        return img.crop((left, top, left + size, top + size))
    if policy == "none":
        if w != h:
            raise ValueError(f"crop policy 'none' requires square images, got {w}x{h}")
        return img if w == size else img.resize((size, size), Image.BILINEAR)
    raise ValueError(f"unknown crop policy {policy!r}; expected one of {CROP_POLICIES}")


def to_uint8_hwc(img: Image.Image) -> np.ndarray:
    a = np.asarray(img, dtype=np.uint8)
    return a[..., None] if a.ndim == 2 else a


def to_float_chw(a: np.ndarray):
    """uint8 HWC (possibly a read-only memmap view) -> float32 CHW tensor in [0, 1]."""
    import torch

    x = np.array(a.transpose(2, 0, 1), dtype=np.float32)  # always a writable copy
    x *= 1.0 / 255.0
    return torch.from_numpy(x)


class RandomHorizontalFlip:
    """Picklable (Windows DataLoader workers use spawn)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x):
        import torch

        return x.flip(-1) if torch.rand(()) < self.p else x


def train_transform(dataset: str, hflip=None):
    """Setup default: horizontal flip for CelebA only, nothing for ChestMNIST."""
    if hflip is None:
        hflip = dataset == "celeba"
    return RandomHorizontalFlip() if hflip else None
