"""CelebA (aligned & cropped JPEGs, 178x218) from local files.

Accepted layouts under `root` (or `root/celeba`):
  * official / torchvision: list_attr_celeba.txt, list_eval_partition.txt, img_align_celeba/
  * Kaggle: list_attr_celeba.csv, list_eval_partition.csv, img_align_celeba/[img_align_celeba/]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = {0: "train", 1: "val", 2: "test"}


def _resolve_root(root: Path) -> Path:
    for cand in (root, root / "celeba"):
        if any((cand / f).exists() for f in ("list_attr_celeba.txt", "list_attr_celeba.csv")):
            return cand
    raise FileNotFoundError(f"CelebA attribute file not found under {root}")


def _resolve_image_dir(root: Path) -> Path:
    d = root / "img_align_celeba"
    if (d / "img_align_celeba").is_dir():  # Kaggle zip nesting
        d = d / "img_align_celeba"
    if not d.is_dir():
        raise FileNotFoundError(f"image folder not found: {d}")
    return d


def _read_attributes(root: Path) -> pd.DataFrame:
    txt = root / "list_attr_celeba.txt"
    if txt.exists():
        lines = txt.read_text().splitlines()
        names = lines[1].split()
        rows = [ln.split() for ln in lines[2:] if ln.strip()]
        ids = [r[0] for r in rows]
        vals = np.asarray([r[1:] for r in rows], dtype=np.int8)
        df = pd.DataFrame(vals, columns=names)
        df.insert(0, "image_id", ids)
    else:
        df = pd.read_csv(root / "list_attr_celeba.csv")
        names = [c for c in df.columns if c != "image_id"]
        df[names] = df[names].astype(np.int8)
    df[names] = (df[names] > 0).astype(np.int8)  # {-1, 1} -> {0, 1}
    return df


def _read_partition(root: Path) -> pd.DataFrame:
    txt = root / "list_eval_partition.txt"
    if txt.exists():
        df = pd.read_csv(txt, sep=r"\s+", header=None, names=["image_id", "partition"])
    else:
        df = pd.read_csv(root / "list_eval_partition.csv")
    return df


class CelebASource:
    name = "celeba"
    channels = 3

    def __init__(self, root):
        self.root = _resolve_root(Path(root))
        self.image_dir = _resolve_image_dir(self.root)
        self._meta: pd.DataFrame | None = None

    @property
    def description(self) -> str:
        return f"CelebA aligned ({self.root})"

    def metadata(self) -> pd.DataFrame:
        if self._meta is None:
            attrs = _read_attributes(self.root)
            part = _read_partition(self.root)
            df = attrs.merge(part, on="image_id", how="inner", validate="one_to_one")
            if len(df) != len(attrs):
                raise ValueError("attribute and partition files disagree on image ids")
            df.insert(1, "split", df.pop("partition").map(SPLITS))
            self._meta = df.sort_values("image_id").reset_index(drop=True)
        return self._meta

    @property
    def label_columns(self) -> list[str]:
        return [c for c in self.metadata().columns if c not in ("image_id", "split")]

    def load(self, i: int) -> Image.Image:
        image_id = self.metadata().at[i, "image_id"]
        with Image.open(self.image_dir / image_id) as im:
            return im.convert("RGB")
