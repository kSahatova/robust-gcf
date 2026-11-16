"""ChestMNIST+ (NIH ChestX-ray14 via MedMNIST+) from the MedMNIST .npz file.

Uses the official MedMNIST train/val/test splits. Image ids are "<split>_<position>" because
MedMNIST does not ship the original NIH file names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Label order of MedMNIST's `chestmnist` (INFO["chestmnist"]["label"]).
LABELS = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule", "pneumonia",
    "pneumothorax", "consolidation", "edema", "emphysema", "fibrosis", "pleural", "hernia",
]
SPLITS = ("train", "val", "test")


def npz_path(root: Path, size: int) -> Path:
    return Path(root) / ("chestmnist.npz" if size == 28 else f"chestmnist_{size}.npz")


def download(root, size: int = 128) -> Path:
    import medmnist  # optional dependency

    Path(root).mkdir(parents=True, exist_ok=True)
    medmnist.ChestMNIST(split="train", download=True, size=size, root=str(root))
    return npz_path(Path(root), size)


class ChestMNISTSource:
    name = "chestmnist"
    channels = 1
    label_columns = list(LABELS)

    def __init__(self, root, size: int = 128, download_if_missing: bool = False):
        self.size = size
        self.path = npz_path(Path(root), size)
        if not self.path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"{self.path} not found (pass download_if_missing=True)")
            download(root, size)
        self._images: np.ndarray | None = None
        self._meta: pd.DataFrame | None = None

    @property
    def description(self) -> str:
        return f"MedMNIST+ chestmnist size={self.size} ({self.path.name})"

    def _load(self) -> None:
        with np.load(self.path) as z:
            imgs, frames = [], []
            for split in SPLITS:
                x, y = z[f"{split}_images"], z[f"{split}_labels"]
                if y.shape[1] != len(LABELS):
                    raise ValueError(f"expected {len(LABELS)} labels, got {y.shape[1]}")
                imgs.append(x)
                f = pd.DataFrame(y.astype(np.int8), columns=LABELS)
                f.insert(0, "split", split)
                f.insert(0, "image_id", [f"{split}_{i}" for i in range(len(f))])
                frames.append(f)
        self._images = np.concatenate(imgs)
        self._meta = pd.concat(frames, ignore_index=True)

    def metadata(self) -> pd.DataFrame:
        if self._meta is None:
            self._load()
        return self._meta

    def load(self, i: int) -> Image.Image:
        if self._images is None:
            self._load()
        a = self._images[i]
        if a.ndim == 3:  # defensive: (H, W, 1)
            a = a[..., 0]
        return Image.fromarray(np.ascontiguousarray(a, dtype=np.uint8))  # 2-D uint8 -> mode "L"
