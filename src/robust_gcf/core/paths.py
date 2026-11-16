"""Canonical on-disk layout (stdlib only)."""
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def cache_dir(data_root: PathLike, dataset: str, resolution: int, crop: str) -> Path:
    return Path(data_root) / "cache" / dataset / "{}_{}".format(crop, resolution)


def classifier_dir(outputs_root: PathLike, dataset: str, target: str, variant: str) -> Path:
    return Path(outputs_root) / "classifiers" / dataset / target / variant


def evalset_dir(outputs_root: PathLike, dataset: str, target: str) -> Path:
    return Path(outputs_root) / "evalsets" / dataset / target


def cfs_dir(outputs_root: PathLike, method: str, dataset: str, target: str) -> Path:
    return Path(outputs_root) / "cfs" / method / dataset / target
