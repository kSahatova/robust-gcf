"""uint8 image cache: one .npy memmap (N, H, W, C) + index.parquet + manifest.json.

Layout: {data_root}/cache/{dataset}/{crop}_{resolution}/
  images.npy      uint8, (N, R, R, C), memory-mapped read-only
  index.parquet   cache_index, image_id, split, <label columns as int8 0/1>
  manifest.json   CacheManifest; written last (a cache without it is incomplete)
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from robust_gcf.core.schemas import CacheManifest
from robust_gcf.data.preprocessing import crop_and_resize, to_uint8_hwc

IMAGES, INDEX, MANIFEST = "images.npy", "index.parquet", "manifest.json"


class ImageCache:
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
