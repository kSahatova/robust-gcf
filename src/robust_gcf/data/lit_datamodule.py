"""LightningDataModule for classifier training (training env only).

Thin shell over `robust_gcf.data.datasets`; CF-method envs should use that module directly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from robust_gcf.core.schemas import TargetSpec
from robust_gcf.data.cache import ImageCache
from robust_gcf.data.datasets import make_split_dataset
from robust_gcf.data.preprocessing import train_transform


class BinaryTargetDataModule(pl.LightningDataModule):
    def __init__(self, cache_dir: str, target: TargetSpec, batch_size: int = 64,
                 eval_batch_size: Optional[int] = None, num_workers: int = 4,
                 train_subsample: Optional[float] = None, subsample_seed: int = 0,
                 balance: str = "none", augment: Optional[bool] = None,
                 pin_memory: bool = True, sampler_seed: int = 0):
        super().__init__()
        if balance not in ("none", "sampler"):
            raise ValueError("balance must be 'none' or 'sampler'")
        self.cache_dir = cache_dir
        self.target = target
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.train_subsample = train_subsample
        self.subsample_seed = subsample_seed
        self.balance = balance
        self.augment = augment
        self.pin_memory = pin_memory
        self.sampler_seed = sampler_seed
        self.datasets = {}
        self.save_hyperparameters({"target": target.to_dict(), "cache_dir": str(cache_dir),
                                   "train_subsample": train_subsample,
                                   "subsample_seed": subsample_seed, "balance": balance})

    def setup(self, stage: Optional[str] = None) -> None:
        if self.datasets:
            return
        cache = ImageCache(self.cache_dir)
        tf = train_transform(self.target.dataset, self.augment)
        self.datasets["train"] = make_split_dataset(cache, self.target, "train", tf,
                                                    self.train_subsample, self.subsample_seed)
        for split in ("val", "test"):
            self.datasets[split] = make_split_dataset(cache, self.target, split)

    # ------------------------------------------------------------------ stats
    def train_counts(self) -> tuple:
        self.setup()
        y = self.datasets["train"].labels
        n_pos = int(y.sum())
        return len(y) - n_pos, n_pos

    @property
    def pos_weight(self) -> float:
        n_neg, n_pos = self.train_counts()
        return n_neg / max(n_pos, 1)

    # ------------------------------------------------------------ dataloaders
    def _loader(self, split: str, train: bool = False) -> DataLoader:
        ds = self.datasets[split]
        kw = dict(num_workers=self.num_workers, pin_memory=self.pin_memory,
                  persistent_workers=self.num_workers > 0)
        if not train:
            return DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False, **kw)
        if self.balance == "sampler":
            y = ds.labels
            w = 1.0 / np.bincount(y, minlength=2)[y]
            g = torch.Generator().manual_seed(self.sampler_seed)
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y),
                                            replacement=True, generator=g)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, drop_last=True, **kw)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True, **kw)

    def train_dataloader(self):
        return self._loader("train", train=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
