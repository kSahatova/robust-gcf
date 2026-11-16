"""Classifier artifacts: outputs/classifiers/{dataset}/{target}/{variant}/

    model.pt   {"state_dict": ClassifierWrapper.state_dict(), "format_version": 1}
               tensors only - no pickled classes, so no Lightning / Hydra / robust_gcf
               class is needed to read it, and `weights_only=True` loading works.
    meta.json  ClassifierMeta (arch, target, preprocessing, threshold, variant, metrics)
"""
from __future__ import annotations

import platform
from pathlib import Path

import torch

from robust_gcf.core.schemas import ClassifierMeta
from robust_gcf.models.factory import build_backbone
from robust_gcf.models.wrapper import ClassifierWrapper

MODEL_FILE, META_FILE = "model.pt", "meta.json"


def _versions() -> dict:
    v = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import torchvision

        v["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover
        pass
    return v


def save_classifier(model: ClassifierWrapper, meta: ClassifierMeta, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if abs(meta.threshold - model.threshold) > 1e-12:
        raise ValueError("meta.threshold and model.threshold disagree")
    if meta.preprocess != model.preprocess:
        raise ValueError("meta.preprocess and model.preprocess disagree")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "format_version": 1}, out_dir / MODEL_FILE)
    meta.replace(lib_versions={**_versions(), **meta.lib_versions}).save(out_dir / META_FILE)
    return out_dir


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=map_location)


def load_classifier(path, map_location="cpu", freeze: bool = True) -> ClassifierWrapper:
    """Rebuild the wrapper from meta.json + model.pt. Result has `.meta` (ClassifierMeta)."""
    path = Path(path)
    meta = ClassifierMeta.load(path / META_FILE)
    backbone = build_backbone(meta.arch, meta.preprocess.channels, pretrained=False)
    model = ClassifierWrapper(backbone, meta.preprocess, meta.threshold)
    model.load_state_dict(_torch_load(path / MODEL_FILE, map_location)["state_dict"], strict=True)
    model.meta = meta
    return model.freeze() if freeze else model
