"""Dataset access, from raw files to torch Datasets. NO Lightning import: usable from any CF-method env.

1. Raw-dataset readers (`RawDatasetReader` subclasses: `CelebAReader`, `ChestMNISTReader`) read
   the original files and give one uniform view that `build_cache` (below) consumes:
       metadata()  one row per image: [image_id, split, *label_columns]
                   split in {"train", "val", "test"}; label columns are int8 0/1
       load(i)     raw image of row i as a PIL image (mode "RGB" or "L")
   Readers do no resizing or cropping; that happens once, in the cache build.

2. `CachedImageDataset` and helpers serve images from the cache. Item format (dict, so method
   adapters can add keys, e.g. COIN's dummy 'masks'):
       {"image": float32 (C, R, R) in [0, 1], "label": float32 scalar (1 = positive), "index": int}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from robust_gcf.core.schemas import TargetSpec
from robust_gcf.data.preprocessing import to_float_chw
from robust_gcf.data.targets import binary_labels

from robust_gcf.core.schemas import CacheManifest
from robust_gcf.data.preprocessing import crop_and_resize, to_uint8_hwc


# ============================================================================ raw readers


class RawDatasetReader(ABC):
    """Interface of a raw-dataset reader (see module docstring)."""

    #: dataset name, matches `TargetSpec.dataset` and the cache path
    name: str
    #: channels of the images returned by `load` (3 = RGB, 1 = grayscale)
    channels: int

    @property
    @abstractmethod
    def label_columns(self) -> List[str]:
        """Names of the 0/1 label columns in `metadata()`."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable provenance, stored in the cache manifest."""

    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        """One row per image: image_id, split, then `label_columns`."""

    @abstractmethod
    def load(self, i: int) -> Image.Image:
        """Raw image of metadata row `i`."""

    def __len__(self) -> int:
        return len(self.metadata())


# ------------------------------------------------------------------------------ CelebA

CELEBA_SPLITS = {0: "train", 1: "val", 2: "test"}


def _celeba_root(root: Path) -> Path:
    for cand in (root, root / "celeba"):
        if any((cand / f).exists() for f in ("list_attr_celeba.txt", "list_attr_celeba.csv")):
            return cand
    raise FileNotFoundError(f"CelebA attribute file not found under {root}")


def _celeba_attributes(root: Path) -> pd.DataFrame:
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


def _celeba_partition(root: Path) -> pd.DataFrame:
    txt = root / "list_eval_partition.txt"
    if txt.exists():
        return pd.read_csv(txt, sep=r"\s+", header=None, names=["image_id", "partition"])
    return pd.read_csv(root / "list_eval_partition.csv")


class CelebAReader(RawDatasetReader):
    """CelebA (aligned & cropped JPEGs, 178x218) from local files.

    Expected under `root` (or `root/celeba`), e.g. root = data/raw/celeba:
      img_align_celeba/*.jpg   (flat; un-nest Kaggle's img_align_celeba/img_align_celeba/)
      attributes + partition as official .txt (list_attr_celeba.txt, list_eval_partition.txt)
      or Kaggle .csv (list_attr_celeba.csv, list_eval_partition.csv)
    """

    name = "celeba"
    channels = 3

    def __init__(self, root):
        self.root = _celeba_root(Path(root))
        self.image_dir = self.root / "img_align_celeba"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image folder not found: {self.image_dir}")
        self._meta: pd.DataFrame | None = None

    @property
    def description(self) -> str:
        return f"CelebA aligned ({self.root})"

    def metadata(self) -> pd.DataFrame:
        if self._meta is None:
            attrs = _celeba_attributes(self.root)
            part = _celeba_partition(self.root)
            df = attrs.merge(part, on="image_id", how="inner", validate="one_to_one")
            if len(df) != len(attrs):
                raise ValueError("attribute and partition files disagree on image ids")
            df.insert(1, "split", df.pop("partition").map(CELEBA_SPLITS))
            self._meta = df.sort_values("image_id").reset_index(drop=True)
        return self._meta

    @property
    def label_columns(self) -> List[str]:
        return [c for c in self.metadata().columns if c not in ("image_id", "split")]

    def load(self, i: int) -> Image.Image:
        image_id = self.metadata().at[i, "image_id"]
        with Image.open(self.image_dir / image_id) as im:
            return im.convert("RGB")


# -------------------------------------------------------------------------- ChestMNIST

# Label order of MedMNIST's `chestmnist` (INFO["chestmnist"]["label"]).
CHESTMNIST_LABELS = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule", "pneumonia",
    "pneumothorax", "consolidation", "edema", "emphysema", "fibrosis", "pleural", "hernia",
]
CHESTMNIST_SPLITS = ("train", "val", "test")


def chestmnist_npz_path(root, size: int) -> Path:
    return Path(root) / ("chestmnist.npz" if size == 28 else f"chestmnist_{size}.npz")


def download_chestmnist(root, size: int = 128) -> Path:
    import medmnist  # optional dependency

    Path(root).mkdir(parents=True, exist_ok=True)
    medmnist.ChestMNIST(split="train", download=True, size=size, root=str(root))
    return chestmnist_npz_path(root, size)


class ChestMNISTReader(RawDatasetReader):
    """ChestMNIST+ (NIH ChestX-ray14 via MedMNIST+) from the MedMNIST .npz file.

    Uses the official MedMNIST train/val/test splits. Image ids are "<split>_<position>" because
    MedMNIST does not ship the original NIH file names.
    """

    name = "chestmnist"
    channels = 1
    label_columns = list(CHESTMNIST_LABELS)

    def __init__(self, root, size: int = 128, download_if_missing: bool = False):
        self.size = size
        self.path = chestmnist_npz_path(root, size)
        if not self.path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"{self.path} not found (pass download_if_missing=True)")
            download_chestmnist(root, size)
        self._images: np.ndarray | None = None
        self._meta: pd.DataFrame | None = None

    @property
    def description(self) -> str:
        return f"MedMNIST+ chestmnist size={self.size} ({self.path.name})"

    def _load(self) -> None:
        with np.load(self.path) as z:
            imgs, frames = [], []
            for split in CHESTMNIST_SPLITS:
                x, y = z[f"{split}_images"], z[f"{split}_labels"]
                if y.shape[1] != len(CHESTMNIST_LABELS):
                    raise ValueError(f"expected {len(CHESTMNIST_LABELS)} labels, got {y.shape[1]}")
                imgs.append(x)
                f = pd.DataFrame(y.astype(np.int8), columns=CHESTMNIST_LABELS)
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


READERS = {r.name: r for r in (CelebAReader, ChestMNISTReader)}


def get_reader(name: str, root, **kwargs) -> RawDatasetReader:
    """Reader for dataset `name` ("celeba", "chestmnist") over the raw files in `root`."""
    if name not in READERS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(READERS)}")
    return READERS[name](root, **kwargs)


# ======================================================================= cached images

IMAGES, INDEX, MANIFEST = "images.npy", "index.parquet", "manifest.json"


def _as_cache(cache) -> ImageCache:
    return cache if isinstance(cache, ImageCache) else ImageCache(cache)


class ImageCache:
    """uint8 image cache: one .npy memmap (N, H, W, C) + index.parquet + manifest.json.

    Layout: {data_root}/cache/{dataset}/{crop}_{resolution}/
    images.npy      uint8, (N, R, R, C), memory-mapped read-only
    index.parquet   cache_index, image_id, split, <label columns as int8 0/1>
    manifest.json   CacheManifest; written last (a cache without it is incomplete)
    """
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / MANIFEST).exists():
            raise FileNotFoundError(f"no complete cache at {self.path} (missing {MANIFEST})")
        self.manifest = CacheManifest.load(self.path / MANIFEST)
        self._images: np.ndarray | None = None
        self._index: pd.DataFrame | None = None

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.path / IMAGES, mmap_mode="r")
        return self._images

    @property
    def index(self) -> pd.DataFrame:
        if self._index is None:
            self._index = pd.read_parquet(self.path / INDEX)
        return self._index

    def __len__(self) -> int:
        return self.manifest.num_images

    def __getitem__(self, i: int) -> np.ndarray:
        return self.images[i]

    # Keep pickles small and memmap-free; each DataLoader worker reopens lazily.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        state["_index"] = None
        return state


def _write_images(reader, path: Path, n: int, resolution: int, crop: str, num_workers: int) -> None:
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(n, resolution, resolution, reader.channels)
    )

    def work(i: int) -> None:
        a = to_uint8_hwc(crop_and_resize(reader.load(i), resolution, crop))
        if a.shape[-1] != reader.channels:
            raise ValueError(f"image {i}: expected {reader.channels} channels, got {a.shape[-1]}")
        arr[i] = a

    step = max(n // 20, 1000)
    with ThreadPoolExecutor(max_workers=max(num_workers, 1)) as ex:
        for k, _ in enumerate(ex.map(work, range(n)), 1):
            if k % step == 0 or k == n:
                print(f"[cache] {reader.name}: {k}/{n}", flush=True)
    arr.flush()
    del arr  # release the mapping before the directory is renamed (Windows)


def build_cache(reader, out_dir, resolution: int, crop: str = "center",
                num_workers: int = 8, overwrite: bool = False) -> ImageCache:
    """Preprocess every image of `reader` once and store it as a uint8 cache; return it opened.

    Each raw image is loaded, cropped and resized to `resolution` x `resolution` with
    `crop_and_resize(..., crop)`, then written to `images.npy` (N, R, R, C), in the row order
    of `reader.metadata()`. That row position becomes `cache_index`, the permanent image id
    used by datasets, eval sets and CF outputs. `index.parquet` stores the metadata (image_id,
    split, all label columns) with `cache_index` as first column; `manifest.json` records how
    the cache was built.

    Args:
        reader: a `RawDatasetReader` (see `robust_gcf.data.datasets`) providing `name`, `channels`,
            `label_columns`, `description`, `metadata()` and `load(i) -> PIL.Image`.
        out_dir: final cache folder, usually `core.paths.cache_dir(data_root, dataset, res, crop)`.
        resolution: side length of the stored square images (128 for classifiers; 256 for GCD).
        crop: "center" (resize shorter side + center crop; equals DiME's CelebA transform) or
            "none" (square inputs, resize only). See `preprocessing.crop_and_resize`.
        num_workers: threads decoding/resizing images in parallel.
        overwrite: rebuild even if a complete cache already exists.

    Behaviour:
        * Idempotent: if `out_dir/manifest.json` exists and `overwrite` is False, the existing
          cache is returned without touching the raw data.
        * Atomic: everything is written to `<out_dir>.tmp` and renamed to `out_dir` only after
          the manifest (written last) is saved. A crashed build leaves no manifest, so
          `ImageCache` refuses to open it, and the next call starts over.

    Raises:
        ValueError: an image has a different channel count than `reader.channels`, or a
            crop policy is unknown / not applicable (e.g. "none" on non-square images).
    """
    out_dir = Path(out_dir)
    if (out_dir / MANIFEST).exists() and not overwrite:
        return ImageCache(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    meta = reader.metadata().reset_index(drop=True)
    n = len(meta)
    _write_images(reader, tmp / IMAGES, n, resolution, crop, num_workers)

    index = meta.copy()
    index.insert(0, "cache_index", np.arange(n, dtype=np.int64))
    index.to_parquet(tmp / INDEX, index=False)

    CacheManifest(
        dataset=reader.name,
        resolution=resolution,
        channels=reader.channels,
        crop=crop,
        num_images=n,
        splits={str(k): int(v) for k, v in index["split"].value_counts().items()},
        label_columns=list(reader.label_columns),
        source=reader.description,
    ).save(tmp / MANIFEST)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return ImageCache(out_dir)


# ============================================================================ cached datasets

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
