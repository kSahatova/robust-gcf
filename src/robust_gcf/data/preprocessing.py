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
              Identical to torchvision `Resize(size)` + `CenterCrop(size)` on PIL images, i.e. the
              deterministic part of DiME's `CelebADataset` transform (its eval-time transform
              with random_flip=False, random_crop=False, before Normalize). Tested.
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


def to_minus_one_one(x):
    """[0, 1] -> [-1, 1]: DiME's `Normalize([0.5]*3, [0.5]*3)`, for feeding DiME/GCD code."""
    return x * 2.0 - 1.0


class RandomHorizontalFlip:
    """Picklable (Windows DataLoader workers use spawn)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x):
        import torch

        return x.flip(-1) if torch.rand(()) < self.p else x


class RandomResizedCrop:
    """torchvision RandomResizedCrop on a CHW tensor (bilinear, antialiased), size preserved."""

    def __init__(self, scale=(0.95, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)

    def __call__(self, x):
        from torchvision.transforms import RandomResizedCrop as _RRC
        from torchvision.transforms import functional as TF

        i, j, h, w = _RRC.get_params(x, list(self.scale), list(self.ratio))
        size = list(x.shape[-2:])
        try:
            return TF.resized_crop(x, i, j, h, w, size, antialias=True)
        except TypeError:  # torchvision < 0.15
            return TF.resized_crop(x, i, j, h, w, size)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def dime_celeba_augmentation(random_flip: bool = True, random_crop: bool = True,
                             crop_scale=(0.95, 1.0)):
    """Random part of DiME's CelebADataset transform, applied to the cached (center-cropped) image.

    DiME: Resize -> RandomHorizontalFlip -> CenterCrop -> RandomResizedCrop(scale=(0.95, 1)).
    Flipping before the center crop equals flipping after it for CelebA: 178x218 resizes to
    128x156, so the crop is purely vertical. Resize + CenterCrop is baked into the cache.
    Differences: the crop here works on a uint8-derived tensor, not on a PIL image, so the
    resampling can differ by up to about one gray level.
    """
    ts = []
    if random_flip:
        ts.append(RandomHorizontalFlip())
    if random_crop:
        ts.append(RandomResizedCrop(scale=crop_scale))
    return Compose(ts) if ts else None


def train_transform(dataset: str, augment=None):
    """Training augmentation. None/True -> dataset default, False -> none.

    CelebA: DiME's augmentation (flip + RandomResizedCrop(scale=(0.95, 1))).
    ChestMNIST: none.
    """
    if augment is False:
        return None
    if dataset == "celeba":
        return dime_celeba_augmentation()
    return None
