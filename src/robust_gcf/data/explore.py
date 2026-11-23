"""Dataset exploration: load a dataset, show both classes (cached, optionally also
DiME-transformed: default on for CelebA, off for ChestMNIST; `dime_transform=` /
`--[no-]dime_transform`),
and compute the statistics classifier training needs.

Notebook:
    from robust_gcf.data import explore
    cache  = explore.load_dataset("celeba", raw_root="data/raw/celeba", data_root="data")
    target = explore.load_target("celeba_smiling")
    fig    = explore.show_examples(cache, target, n_per_class=4)
    stats  = explore.compute_stats(cache, target)

CLI (writes examples.png + stats.json):
    python -m robust_gcf.data.explore --target celeba_smiling --raw-root data/raw/celeba
    python -m robust_gcf.data.explore --target chestmnist_effusion --raw-root data/raw/medmnist --download

Needs the `viz` extra (matplotlib, pyyaml). Matplotlib is imported lazily.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from robust_gcf.core.paths import cache_dir
from robust_gcf.core.schemas import PreprocessSpec, TargetSpec
from robust_gcf.data.datasets import target_frame, ImageCache, build_cache
from robust_gcf.data.preprocessing import dime_celeba_augmentation, to_float_chw, to_minus_one_one
from robust_gcf.data.datasets import get_reader

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CROP = {"celeba": "center", "chestmnist": "none"}


# ------------------------------------------------------------------------- loading


def load_dataset(dataset: str, raw_root, data_root="data", resolution: int = 128,
                 crop: Optional[str] = None, download: bool = False,
                 num_workers: int = 8, overwrite: bool = False) -> ImageCache:
    """Open the cache for `dataset`, building it from the raw files on first use."""
    crop = crop or DEFAULT_CROP[dataset]
    path = cache_dir(data_root, dataset, resolution, crop)
    if (path / "manifest.json").exists() and not overwrite:
        return ImageCache(path)
    kwargs = {"size": resolution, "download_if_missing": download} if dataset == "chestmnist" else {}
    reader = get_reader(dataset, raw_root, **kwargs)
    return build_cache(reader, path, resolution, crop, num_workers=num_workers, overwrite=overwrite)


def load_target(name_or_path) -> TargetSpec:
    """TargetSpec of a task: configs/task/<name>.yaml (its `target:` block), or a .yaml path
    to a task file or to a bare TargetSpec mapping. Reads the file directly, without Hydra."""
    import yaml

    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = REPO_ROOT / "configs" / "task" / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    return TargetSpec.from_dict(cfg["target"] if "target" in cfg else cfg)


# ---------------------------------------------------------------- DiME transforms


def dime_transform(dataset: str, train: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """DiME's CelebADataset transform applied to a cached [0,1] image; returns [-1, 1].

    The deterministic part (Resize -> CenterCrop) is already in the cache, so this adds
      train=True : RandomHorizontalFlip + RandomResizedCrop(scale=(0.95, 1)), then Normalize(0.5, 0.5)
      train=False: Normalize(0.5, 0.5) only
    ChestMNIST: DiME has no ChestMNIST pipeline; per the setup there is no augmentation, so
    only the [-1, 1] normalization is applied.
    """
    aug = dime_celeba_augmentation() if (train and dataset == "celeba") else None

    def apply(x: torch.Tensor) -> torch.Tensor:
        return to_minus_one_one(aug(x) if aug is not None else x)

    return apply


# ------------------------------------------------------------------ visualization


def sample_examples(cache: ImageCache, target: TargetSpec, n_per_class: int = 4,
                    split: str = "train", seed: int = 0) -> Dict[int, np.ndarray]:
    """{label: cache indices} with up to n_per_class random images per class (seeded)."""
    f = target_frame(cache, target)
    f = f[f["split"] == split]
    rng = np.random.default_rng(seed)
    out = {}
    for label in (1, 0):
        idx = f.loc[f["label"] == label, "cache_index"].to_numpy()
        out[label] = np.sort(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    return out


def _imshow(ax, x: torch.Tensor) -> None:
    """Show a [0,1] CHW tensor."""
    a = x.clamp(0, 1).permute(1, 2, 0).numpy()
    if a.shape[-1] == 1:
        ax.imshow(a[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(a)
    ax.set_xticks([])
    ax.set_yticks([])


DIME_DEFAULT = {"celeba": True, "chestmnist": False}
_make_dime_transform = dime_transform  # the `dime_transform` argument below shadows the function


def show_examples(cache: ImageCache, target: TargetSpec, n_per_class: int = 4,
                  split: str = "train", seed: int = 0, dime_transform: Optional[bool] = None,
                  transform: Optional[Callable] = None,
                  transform_name: str = "DiME train transform"):
    """Grid of example images per class: the cached image, optionally with a transformed copy.

    dime_transform: show a second row per class with the transform applied.
          None (default) -> per dataset: on for CelebA, off for ChestMNIST (DiME has no
          ChestMNIST pipeline and the setup uses no augmentation there).
    transform: maps [0,1] -> [-1,1]; default `dime_transform(dataset, train=True)` (the module
          function). Passing one implies dime_transform=True unless dime_transform=False is
          given. Transformed images are mapped back to [0,1] for display.
    """
    import matplotlib.pyplot as plt

    show_dime = dime_transform
    if show_dime is None:
        show_dime = transform is not None or DIME_DEFAULT.get(target.dataset, False)
    if show_dime and transform is None:
        transform = _make_dime_transform(target.dataset, train=True)
    rows_per_class = 2 if show_dime else 1

    examples = sample_examples(cache, target, n_per_class, split, seed)
    names = {1: target.positive_name, 0: target.negative_name}
    ncols = max(max(len(v) for v in examples.values()), 1)
    nrows = 2 * rows_per_class
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.3 * nrows), squeeze=False)
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)  # reproducible augmentation draws
    try:
        for block, label in enumerate((1, 0)):
            r0 = block * rows_per_class
            for col in range(ncols):
                cells = [axes[r0 + r, col] for r in range(rows_per_class)]
                if col >= len(examples[label]):
                    for ax in cells:
                        ax.axis("off")
                    continue
                ci = int(examples[label][col])
                x = to_float_chw(cache[ci])
                _imshow(cells[0], x)
                cells[0].set_title(str(cache.index.at[ci, "image_id"]), fontsize=7)
                if show_dime:
                    _imshow(cells[1], (transform(x) + 1.0) / 2.0)
            axes[r0, 0].set_ylabel(f"{names[label]} (y={label})\ncached", fontsize=8)
            if show_dime:
                axes[r0 + 1, 0].set_ylabel(f"{names[label]}\n{transform_name}", fontsize=8)
    finally:
        torch.random.set_rng_state(gen_state)
    fig.suptitle(f"{target.name}: {split} split, {cache.manifest.resolution}px "
                 f"(crop={cache.manifest.crop})", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- statistics


def _pixel_stats(cache: ImageCache, cache_indices: np.ndarray, labels: np.ndarray,
                 max_images: int, seed: int, chunk: int = 256) -> dict:
    """Per-channel mean/std in [0,1] over a seeded sample of training images, overall and per class."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(cache_indices), size=min(max_images, len(cache_indices)), replace=False))
    ci, y = cache_indices[pick], labels[pick]
    c = cache.manifest.channels
    s1, s2 = np.zeros((2, c)), np.zeros((2, c))
    npx = np.zeros(2)
    for start in range(0, len(ci), chunk):
        a = cache.images[ci[start:start + chunk]].astype(np.float64) / 255.0  # (B, H, W, C)
        for label in (0, 1):
            m = y[start:start + chunk] == label
            if m.any():
                v = a[m].reshape(-1, c)
                s1[label] += v.sum(0)
                s2[label] += (v ** 2).sum(0)
                npx[label] += len(v)

    def ms(t1, t2, n):
        mean = t1 / max(n, 1)
        std = np.sqrt(np.maximum(t2 / max(n, 1) - mean ** 2, 0))
        return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]

    mean, std = ms(s1.sum(0), s2.sum(0), npx.sum())
    per_class = {}
    for label in (0, 1):
        m, s = ms(s1[label], s2[label], npx[label])
        per_class[str(label)] = {"mean": m, "std": s}
    return {"n_images": int(len(ci)), "mean": mean, "std": std, "per_class": per_class}


def _correlated_columns(cache: ImageCache, target: TargetSpec, frame, top_k: int) -> list:
    """Label columns most correlated (phi) with the target label on train: possible confounders."""
    idx = cache.index.set_index("cache_index").loc[frame["cache_index"].to_numpy()]
    y = frame["label"].to_numpy().astype(np.float64)
    out = []
    for col in cache.manifest.label_columns:
        if col == target.column:
            continue
        v = idx[col].to_numpy().astype(np.float64)
        if v.std() == 0 or y.std() == 0:
            continue
        out.append((col, float(np.corrcoef(y, v)[0, 1]), float(v.mean())))
    out.sort(key=lambda t: -abs(t[1]))
    return [{"column": c, "phi": round(r, 3), "prevalence": round(p, 3)} for c, r, p in out[:top_k]]


def compute_stats(cache: ImageCache, target: TargetSpec, batch_size: int = 64,
                  pixel_sample: int = 5000, seed: int = 0, top_k_correlated: int = 5) -> dict:
    """Statistics needed to configure classifier training for `target`.

    - split sizes, class counts, positive fraction, rows excluded by the negative policy
    - pos_weight (n_neg / n_pos on train) and sampler class weights
    - trivial baselines: majority-class accuracy per split (AUROC baseline is 0.5)
    - steps per epoch at `batch_size`
    - per-channel pixel mean/std on train (overall, per class) vs the normalization in use
    - label columns most correlated with the target (possible shortcuts / confounders)
    """
    f = target_frame(cache, target)
    splits = {}
    for split in ("train", "val", "test"):
        g = f[f["split"] == split]
        n, n_pos = len(g), int(g["label"].sum())
        splits[split] = {
            "n": n, "n_pos": n_pos, "n_neg": n - n_pos,
            "pos_frac": round(n_pos / max(n, 1), 4),
            "majority_acc": round(max(n_pos, n - n_pos) / max(n, 1), 4),
            "excluded": int(cache.manifest.splits.get(split, 0) - n),
        }
    tr = splits["train"]
    train = f[f["split"] == "train"]
    pre = PreprocessSpec.default_for(cache.manifest.channels)
    return {
        "dataset": target.dataset,
        "target": target.name,
        "classes": {"1": target.positive_name, "0": target.negative_name},
        "cf_direction": f"{target.positive_name} -> {target.negative_name}",
        "negative_policy": target.negative_policy,
        "imbalance_strategy": target.imbalance,
        "image": {"resolution": cache.manifest.resolution, "channels": cache.manifest.channels,
                  "crop": cache.manifest.crop, "images_in_cache": len(cache),
                  "images_in_task": int(len(f))},
        "splits": splits,
        "training": {
            "pos_weight": round(tr["n_neg"] / max(tr["n_pos"], 1), 4),
            "sampler_class_weights": {"0": round(1 / max(tr["n_neg"], 1), 8),
                                      "1": round(1 / max(tr["n_pos"], 1), 8)},
            "batch_size": batch_size,
            "steps_per_epoch": tr["n"] // batch_size,
        },
        "pixels_train": _pixel_stats(cache, train["cache_index"].to_numpy(),
                                     train["label"].to_numpy(), pixel_sample, seed),
        "normalization_in_use": {"mean": list(pre.mean), "std": list(pre.std)},
        "top_correlated_columns": _correlated_columns(cache, target, train, top_k_correlated),
    }


def format_stats(stats: dict) -> str:
    """Short human-readable summary."""
    lines = [f"{stats['target']}  ({stats['cf_direction']}, negatives: {stats['negative_policy']})",
             f"  images: {stats['image']['images_in_task']} in task / {stats['image']['images_in_cache']} cached, "
             f"{stats['image']['resolution']}px x{stats['image']['channels']}"]
    for split, s in stats["splits"].items():
        lines.append(f"  {split:5s} n={s['n']:7d}  pos={s['n_pos']:7d} ({s['pos_frac']:.1%})  "
                     f"majority acc={s['majority_acc']:.3f}  excluded={s['excluded']}")
    t, p = stats["training"], stats["pixels_train"]
    lines += [f"  pos_weight={t['pos_weight']}  steps/epoch@{t['batch_size']}={t['steps_per_epoch']}  "
              f"imbalance strategy={stats['imbalance_strategy']}",
              f"  pixel mean={p['mean']} std={p['std']} (n={p['n_images']}); "
              f"normalization in use mean={stats['normalization_in_use']['mean']}",
              "  most correlated: " + ", ".join(f"{c['column']} ({c['phi']:+.2f})"
                                               for c in stats["top_correlated_columns"])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- CLI


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, help="task name (configs/task/<name>.yaml) or .yaml path")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None, help="default: outputs/explore/<target>")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-per-class", type=int, default=4)
    ap.add_argument("--pixel-sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--download", action="store_true", help="ChestMNIST: download via medmnist")
    ap.add_argument("--dime_transform", action=argparse.BooleanOptionalAction, default=None,
                    help="show DiME-transformed rows in examples.png "
                         "(default: on for CelebA, off for ChestMNIST)")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    target = load_target(args.target)
    cache = load_dataset(target.dataset, args.raw_root, args.data_root, args.resolution,
                         args.crop, args.download, args.num_workers)
    out = Path(args.out or Path("outputs") / "dataset_stats" / target.name)
    out.mkdir(parents=True, exist_ok=True)

    fig = show_examples(cache, target, args.n_per_class, args.split, args.seed, dime_transform=args.dime_transform)
    fig.savefig(out / "examples.png", dpi=120)
    stats = compute_stats(cache, target, pixel_sample=args.pixel_sample, seed=args.seed)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(format_stats(stats))
    print(f"wrote {out / 'examples.png'} and {out / 'stats.json'}")
    return stats


if __name__ == "__main__":
    main()
