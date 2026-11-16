"""Raw-dataset readers. Each source exposes:

    name: str, channels: int
    metadata() -> DataFrame with columns [image_id, split, *label_columns], one row per image
    label_columns: list[str]   (0/1 int8 columns)
    load(i) -> PIL.Image      (i = row position in metadata())
    description: str
"""
from robust_gcf.data.sources.celeba import CelebASource
from robust_gcf.data.sources.chestmnist import ChestMNISTSource


def get_source(name: str, root, **kwargs):
    if name == "celeba":
        return CelebASource(root, **kwargs)
    if name == "chestmnist":
        return ChestMNISTSource(root, **kwargs)
    raise ValueError(f"unknown dataset: {name}")


__all__ = ["CelebASource", "ChestMNISTSource", "get_source"]
