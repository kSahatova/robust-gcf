"""Plain torch Datasets over the cache. NO Lightning import: usable from any CF-method env.

Item format (dict, so method adapters can add keys, e.g. COIN's dummy 'masks'):
    {"image": float32 (C, R, R) in [0, 1], "label": float32 scalar (1 = positive), "index": int}
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from robust_gcf.core.schemas import TargetSpec
from robust_gcf.data.cache import ImageCache
from robust_gcf.data.preprocessing import to_float_chw
from robust_gcf.data.targets import binary_labels


def _as_cache(cache) -> ImageCache:
    return cache if isinstance(cache, ImageCache) else ImageCache(cache)


class CachedImageDataset(Dataset):
    def __init__(self, cache: Union[ImageCache, str], cache_indices: Sequence[int],
                 labels: Optional[Sequence[int]] = None, transform: Optional[Callable] = None):
        self.cache = _as_cache(cache)
        self.cache_indices = np.asarray(cache_indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.labels) != len(self.cache_indices):
            raise ValueError("labels and cache_indices differ in length")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.cache_indices)

    def __getitem__(self, i: int) -> dict:
        ci = int(self.cache_indices[i])
        x = to_float_chw(self.cache[ci])
        if self.transform is not None:
            x = self.transform(x)
        item = {"image": x, "index": ci}
        if self.labels is not None:
            item["label"] = torch.tensor(float(self.labels[i]))
        return item


def target_frame(cache, target: TargetSpec) -> pd.DataFrame:
    cache = _as_cache(cache)
    if cache.manifest.dataset != target.dataset:
        raise ValueError(f"target {target.name} is for {target.dataset}, cache is {cache.manifest.dataset}")
    return binary_labels(cache.index, target, cache.manifest.label_columns)


def stratified_subsample(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Positions of a class-stratified random subsample (seeded; for E2 retraining)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        k = max(1, int(round(fraction * len(pos))))
        keep.append(rng.choice(pos, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def make_split_dataset(cache, target: TargetSpec, split: str, transform=None,
                       subsample: Optional[float] = None, seed: int = 0) -> CachedImageDataset:
    cache = _as_cache(cache)
    f = target_frame(cache, target)
    f = f[f["split"] == split]
    idx, lab = f["cache_index"].to_numpy(), f["label"].to_numpy()
    if subsample is not None and subsample < 1:
        sel = stratified_subsample(lab, subsample, seed)
        idx, lab = idx[sel], lab[sel]
    return CachedImageDataset(cache, idx, lab, transform)
