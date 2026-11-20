"""Train one classifier (task x variant) from a composed Hydra config.

Steps:
  1. open the image cache (built from the raw files on first use)
  2. fit `LitClassifier` (DenseNet121 + `ClassifierWrapper`) with checkpointing and early
     stopping on `train.monitor` (val AUROC)
  3. reload the best checkpoint, choose the decision threshold on the validation set
  4. evaluate on the test split at that threshold (`trainer.test`)
  5. write outputs/classifiers/{dataset}/{target}/{variant}/:
       model.pt, meta.json   classifier artifact (load with robust_gcf.models.load_classifier)
       metrics.json          val/test metrics, threshold, training summary
       config.yaml           fully resolved config of this run
       lightning/best.ckpt   Lightning checkpoint (resume / debugging only)
       logs/metrics.csv      per-epoch logs (CSVLogger)

Entry point: scripts/train_classifier.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

from robust_gcf.core.paths import classifier_dir
from robust_gcf.core.schemas import ClassifierMeta, PreprocessSpec, TargetSpec
from robust_gcf.data.lit_datamodule import BinaryTargetDataModule
from robust_gcf.data.explore import load_dataset
from robust_gcf.models import build_classifier, save_classifier
from robust_gcf.models.checkpoint import MODEL_FILE
from robust_gcf.training.lit_classifier import LitClassifier, binary_metrics
from robust_gcf.training.utils import choose_threshold


def build_preprocess(cfg: DictConfig) -> PreprocessSpec:
    """PreprocessSpec from the dataset (channels) and model (size, interpolation, mean/std) configs."""
    pre = PreprocessSpec.default_for(int(cfg.dataset.channels), int(cfg.model.input_size))
    changes = {"interpolation": cfg.model.interpolation, "antialias": bool(cfg.model.antialias)}
    if cfg.model.get("mean") is not None:
        changes["mean"] = tuple(cfg.model.mean)
    if cfg.model.get("std") is not None:
        changes["std"] = tuple(cfg.model.std)
    return pre.replace(**changes)


def _precision(cfg: DictConfig) -> str:
    if cfg.train.precision != "auto":
        return str(cfg.train.precision)
    gpu = torch.cuda.is_available() and cfg.train.accelerator in ("auto", "gpu", "cuda")
    return "16-mixed" if gpu else "32-true"


def _threshold_method(cfg: DictConfig, target: TargetSpec) -> str:
    method = cfg.train.threshold_method
    if method == "auto":
        return "youden" if target.imbalance == "threshold" else "fixed"
    return method


def _predict(trainer: pl.Trainer, lit: LitClassifier, loader):
    out = trainer.predict(lit, dataloaders=loader)
    return torch.cat([b["proba"] for b in out]), torch.cat([b["label"] for b in out])


def _clean(d):
    """NaN -> None, so metrics.json is valid JSON."""
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    return None if isinstance(d, float) and math.isnan(d) else d


def train(cfg: DictConfig) -> Path: 
    # creates a TargetSpec from the dict in the config file, and checks that the dataset matches the config
    target = TargetSpec.from_dict(OmegaConf.to_container(cfg.target, resolve=True))
    if target.dataset != cfg.dataset.name:
        raise ValueError(f"target {target.name} is for {target.dataset}, dataset config is {cfg.dataset.name}")
    if cfg.get("epochs") is not None:  # `epochs=N` on the command line overrides train.max_epochs
        cfg.train.max_epochs = int(cfg.epochs)
    out = classifier_dir(cfg.paths.outputs_root, target.dataset, target.name, cfg.variant.name)
    if (out / MODEL_FILE).exists() and not cfg.train.overwrite:
        raise FileExistsError(f"{out / MODEL_FILE} exists; pass train.overwrite=true to replace it")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    pl.seed_everything(int(cfg.seed), workers=True)

    # ---- data
    cache = load_dataset(target.dataset, cfg.dataset.raw_root, cfg.paths.data_root,
                         int(cfg.dataset.cache_resolution), cfg.dataset.crop,
                         download=bool(cfg.dataset.get("download", False)),
                         num_workers=max(int(cfg.train.num_workers), 1))
    dm = BinaryTargetDataModule(
        cache.path, target, batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size), num_workers=int(cfg.train.num_workers),
        train_subsample=cfg.variant.train_subsample, subsample_seed=int(cfg.variant.subsample_seed),
        balance=cfg.train.balance, augment=cfg.dataset.augment,
        pin_memory=torch.cuda.is_available(), sampler_seed=int(cfg.seed))
    dm.setup()
    n_neg, n_pos = dm.train_counts()
    pos_weight = dm.pos_weight if target.imbalance == "pos_weight" else None

    # ---- model
    preprocess = build_preprocess(cfg)
    model = build_classifier(cfg.model.arch, preprocess, pretrained=bool(cfg.model.pretrained))
    lit = LitClassifier(model, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
                        pos_weight=pos_weight, scheduler=cfg.train.scheduler)

    # ---- fit
    ckpt = ModelCheckpoint(dirpath=out / "lightning", filename="best", monitor=cfg.train.monitor,
                           mode="max", save_top_k=1, enable_version_counter=False)
    stop = EarlyStopping(monitor=cfg.train.monitor, mode="max",
                         patience=int(cfg.train.early_stopping_patience))
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs), accelerator=cfg.train.accelerator,
        devices=cfg.train.devices, precision=_precision(cfg), deterministic=cfg.train.deterministic,
        gradient_clip_val=cfg.train.gradient_clip_val, callbacks=[ckpt, stop],
        logger=CSVLogger(save_dir=str(out), name="logs", version=""),
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        limit_test_batches=cfg.train.limit_test_batches,
        log_every_n_steps=10)
    trainer.fit(lit, datamodule=dm)
    epochs_run = int(trainer.current_epoch)

    if ckpt.best_model_path:
        best_model = build_classifier(cfg.model.arch, preprocess, pretrained=False)
        lit = LitClassifier.load_from_checkpoint(ckpt.best_model_path, model=best_model)

    # ---- threshold on validation, then test
    method = _threshold_method(cfg, target)
    val_p, val_y = _predict(trainer, lit, dm.val_dataloader())
    threshold = choose_threshold(val_p.numpy(), val_y.numpy(), method)
    lit.model.threshold = threshold
    val_metrics = binary_metrics(val_p, val_y, threshold)
    trainer.test(lit, dataloaders=dm.test_dataloader(), verbose=False)
    test_metrics = lit.test_metrics

    # ---- save artifact + metrics
    best_score = ckpt.best_model_score
    summary = {
        "epochs_run": epochs_run,
        "best_epoch_val_auroc": None if best_score is None else float(best_score),
        "train_n_neg": n_neg, "train_n_pos": n_pos,
        "pos_weight": pos_weight, "precision": _precision(cfg),
    }
    metrics = _clean({"threshold": threshold, "threshold_method": method,
                      "val": val_metrics, "test": test_metrics, "training": summary})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    flat = {f"{split}/{k}": float(v) for split in ("val", "test")
            for k, v in (metrics[split] or {}).items() if isinstance(v, (int, float))}
    meta = ClassifierMeta(
        arch=cfg.model.arch, target=target, preprocess=preprocess,
        threshold=threshold, threshold_method=method,
        variant=cfg.variant.name,
        variant_params=OmegaConf.to_container(cfg.variant, resolve=True),
        seed=int(cfg.seed), train_subsample=cfg.variant.train_subsample, pos_weight=pos_weight,
        cache_resolution=int(cache.manifest.resolution), metrics=flat)
    save_classifier(lit.model.cpu().eval(), meta, out)
    print(f"[train] {target.name}/{cfg.variant.name}: val AUROC={val_metrics['auroc']:.4f}  "
          f"test AUROC={test_metrics.get('auroc', math.nan):.4f}  threshold={threshold:.4f} ({method})")
    print(f"[train] saved to {out}")
    return out
