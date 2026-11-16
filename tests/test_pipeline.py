"""End-to-end smoke tests on tiny synthetic CelebA / ChestMNIST files (no downloads)."""
import numpy as np
import pytest
import torch
from PIL import Image

from robust_gcf.core.schemas import ClassifierMeta, PreprocessSpec, TargetSpec
from robust_gcf.data.cache import ImageCache, build_cache
from robust_gcf.data.datasets import make_split_dataset, stratified_subsample, target_frame
from robust_gcf.data.sources import get_source
from robust_gcf.data.sources.chestmnist import LABELS
from robust_gcf.models import (FromMinusOneOne, TwoClassLogits, build_classifier,
                               load_classifier, save_classifier)
from robust_gcf.training.threshold import choose_threshold

SMILING = TargetSpec(name="celeba_smiling", dataset="celeba", column="Smiling",
                     positive_name="Smiling", negative_name="Not smiling")
EFFUSION = TargetSpec(name="chestmnist_effusion", dataset="chestmnist", column="effusion",
                      positive_name="Effusion", negative_name="Healthy",
                      negative_policy="no_finding", imbalance="pos_weight")


@pytest.fixture(scope="module")
def celeba_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("celeba")
    (root / "img_align_celeba").mkdir()
    rng = np.random.default_rng(0)
    n, names = 12, ["Male", "Smiling", "Young"]
    attr_lines, part_lines = [str(n), " ".join(names)], []
    for i in range(n):
        fid = f"{i + 1:06d}.jpg"
        Image.fromarray(rng.integers(0, 256, (218, 178, 3), dtype=np.uint8)).save(root / "img_align_celeba" / fid)
        vals = ["1" if (i + j) % 2 else "-1" for j in range(len(names))]
        attr_lines.append(f"{fid} " + "  ".join(vals))
        part_lines.append(f"{fid} {0 if i < 8 else (1 if i < 10 else 2)}")
    (root / "list_attr_celeba.txt").write_text("\n".join(attr_lines) + "\n")
    (root / "list_eval_partition.txt").write_text("\n".join(part_lines) + "\n")
    return root


@pytest.fixture(scope="module")
def chest_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("chest")
    rng = np.random.default_rng(1)
    arrays = {}
    for split, n in (("train", 20), ("val", 6), ("test", 6)):
        y = np.zeros((n, len(LABELS)), dtype=np.uint8)
        y[0::3, 2] = 1  # effusion
        y[1::3, 0] = 1  # atelectasis only -> excluded under "no_finding"
        arrays[f"{split}_images"] = rng.integers(0, 256, (n, 64, 64), dtype=np.uint8)
        arrays[f"{split}_labels"] = y
    np.savez(root / "chestmnist_64.npz", **arrays)
    return root


def test_celeba_cache_and_dataset(celeba_root, tmp_path):
    cache = build_cache(get_source("celeba", celeba_root), tmp_path / "c", 64, "center", num_workers=2)
    assert cache.images.shape == (12, 64, 64, 3)
    assert cache.manifest.splits == {"train": 8, "val": 2, "test": 2}
    # center crop == torchvision Resize(64) + CenterCrop(64)
    from torchvision import transforms as T

    ref = T.CenterCrop(64)(T.Resize(64)(Image.open(celeba_root / "img_align_celeba" / "000001.jpg").convert("RGB")))
    assert np.array_equal(np.asarray(ref), cache.images[0])

    ds = make_split_dataset(cache, SMILING, "train")
    item = ds[0]
    assert item["image"].shape == (3, 64, 64) and item["image"].dtype == torch.float32
    assert 0 <= item["image"].min() and item["image"].max() <= 1
    f = target_frame(cache, SMILING)
    assert f["label"].tolist() == cache.index["Smiling"].tolist()
    with pytest.raises(ValueError):
        target_frame(cache, EFFUSION)  # wrong dataset


def test_chest_no_finding_policy(chest_root, tmp_path):
    src = get_source("chestmnist", chest_root, size=64)
    cache = build_cache(src, tmp_path / "c", 64, "none", num_workers=2)
    assert cache.images.shape == (32, 64, 64, 1)
    f = target_frame(ImageCache(tmp_path / "c"), EFFUSION)
    tr = f[f.split == "train"]
    assert len(tr) == 20 - 7 and tr.label.sum() == 7  # atelectasis-only rows dropped
    ds = make_split_dataset(cache, EFFUSION, "train", subsample=0.5, seed=3)
    assert len(ds) == 4 + 3  # round(.5*7)=4 (python banker's rounding: 3.5 -> 4), round(.5*6)=3


def test_stratified_subsample_deterministic():
    y = np.array([0] * 80 + [1] * 20)
    a, b = stratified_subsample(y, 0.5, 7), stratified_subsample(y, 0.5, 7)
    assert np.array_equal(a, b) and y[a].sum() == 10 and len(a) == 50


@pytest.mark.parametrize("channels", [1, 3])
def test_classifier_roundtrip_and_gradients(channels, tmp_path):
    target = EFFUSION if channels == 1 else SMILING
    pre = PreprocessSpec.default_for(channels)
    model = build_classifier("densenet121", pre, pretrained=False, threshold=0.3).eval()
    meta = ClassifierMeta(target=target, preprocess=pre, threshold=0.3, seed=0)
    save_classifier(model, meta, tmp_path / "m0")
    loaded = load_classifier(tmp_path / "m0")
    assert loaded.meta.target == target and loaded.threshold == 0.3

    x = torch.rand(2, channels, 128, 128)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert loaded(x).shape == (2,)

    # white-box access: gradients w.r.t. the input, none w.r.t. frozen weights
    x256 = torch.rand(2, channels, 256, 256, requires_grad=True)  # e.g. GCD resolution
    loaded(x256).sum().backward()
    assert x256.grad is not None and x256.grad.abs().sum() > 0
    assert all(p.grad is None for p in loaded.parameters())

    # adapters
    with torch.no_grad():
        p = torch.sigmoid(loaded(x))
        assert torch.allclose(torch.softmax(TwoClassLogits(loaded)(x), 1)[:, 1], p, atol=1e-6)
        assert torch.allclose(FromMinusOneOne(loaded)(x * 2 - 1), loaded(x), atol=1e-5)

    with pytest.raises(ValueError):
        loaded(torch.rand(1, 4 - channels if channels == 3 else 3, 128, 128))


def test_threshold():
    p = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    y = np.array([1, 1, 0, 1, 0, 0])
    assert choose_threshold(p, y, "fixed") == 0.5
    assert choose_threshold(p, y, "youden") in (0.8, 0.4)
    t = choose_threshold(p, y, "f1")
    assert ((p >= t) == y.astype(bool)).sum() >= 4


def test_core_has_no_torch_import():
    import subprocess
    import sys

    code = ("import sys, robust_gcf.core, robust_gcf.core.paths; "
            "bad=[m for m in ('torch','lightning','pytorch_lightning','numpy') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)
