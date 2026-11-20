"""LightningModule around a ClassifierWrapper, plus the standard binary metrics.

Lightning is a training harness only: the artifact that leaves training is the plain
`ClassifierWrapper` (`lit.model`), saved with `robust_gcf.models.save_classifier`.
Lightning .ckpt files are for resuming runs, never consumed by CF methods.

Logged metrics (all computed by `binary_metrics`):
  train/loss
  val/loss, val/auroc, val/ap, val/accuracy, val/balanced_accuracy, ...   (each epoch)
  test/...                                                                (trainer.test)
Threshold-dependent metrics use `model.threshold` (0.5 until it is calibrated on validation).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from robust_gcf.models.wrapper import ClassifierWrapper


def binary_metrics(proba, labels, threshold: float = 0.5) -> Dict[str, float]:
    """Ranking metrics (AUROC, AP) and metrics at `threshold` (predict positive iff p >= t).

    AUROC / AP are NaN when only one class is present.
    """
    from torchmetrics.functional.classification import binary_auroc, binary_average_precision

    p = torch.as_tensor(np.array(proba, dtype=np.float32)).flatten()  # copies: inputs may be read-only
    y = torch.as_tensor(np.array(labels, dtype=np.int64)).flatten()
    n, n_pos = len(y), int(y.sum())
    n_neg = n - n_pos
    pred = (p >= threshold).long()
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp, fn = n_neg - tn, n_pos - tp
    both = n_pos > 0 and n_neg > 0
    tpr = tp / n_pos if n_pos else math.nan
    tnr = tn / n_neg if n_neg else math.nan
    prec = tp / (tp + fp) if tp + fp else math.nan
    return {
        "n": n,
        "pos_frac": n_pos / max(n, 1),
        "auroc": float(binary_auroc(p, y)) if both else math.nan,
        "ap": float(binary_average_precision(p, y)) if both else math.nan,
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(n, 1),
        "balanced_accuracy": (tpr + tnr) / 2 if both else math.nan,
        "tpr": tpr,
        "tnr": tnr,
        "precision": prec,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan,
    }


class LitClassifier(pl.LightningModule):
    def __init__(self, model: ClassifierWrapper, lr: float = 1e-4, weight_decay: float = 1e-4,
                 pos_weight: Optional[float] = None, scheduler: str = "cosine"):
        super().__init__()
        if scheduler not in ("cosine", "none"):
            raise ValueError("scheduler must be 'cosine' or 'none'")
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.register_buffer("pos_weight", None if pos_weight is None
                             else torch.tensor(float(pos_weight)))
        self._outputs = {"val": [], "test": []}
        self.test_metrics: Dict[str, float] = {}

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=self.pos_weight)

    # ------------------------------------------------------------------- training
    def training_step(self, batch, batch_idx):
        loss = self._loss(self(batch["image"]), batch["label"])
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch["label"]))
        return loss

    # ----------------------------------------------------------------- evaluation
    def _eval_step(self, stage: str, batch):
        logits = self(batch["image"])
        self.log(f"{stage}/loss", 
                 self._loss(logits, batch["label"]), 
                 prog_bar=stage == "val",
                 batch_size=len(batch["label"]))
        self._outputs[stage].append((torch.sigmoid(logits.detach().float()).cpu(), batch["label"].cpu()))

    def _eval_epoch_end(self, stage: str) -> Dict[str, float]:
        if not self._outputs[stage]:
            return {}
        proba, y = map(torch.cat, zip(*self._outputs[stage]))
        self._outputs[stage].clear()
        m = binary_metrics(proba, y, self.model.threshold)
        for k in ("auroc", "ap", "accuracy", "balanced_accuracy", "tpr", "tnr", "f1"):
            if not math.isnan(m[k]):
                self.log(f"{stage}/{k}", m[k], prog_bar=k == "auroc")
        return m

    def validation_step(self, batch, batch_idx):
        self._eval_step("val", batch)

    def on_validation_epoch_end(self):
        self._eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._eval_step("test", batch)

    def on_test_epoch_end(self):
        self.test_metrics = self._eval_epoch_end("test")

    def predict_step(self, batch, batch_idx):
        return {"proba": torch.sigmoid(self(batch["image"]).float()).cpu(),
                "label": batch.get("label"), "index": batch["index"]}

    # ------------------------------------------------------------------ optimizer
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        max_epochs = getattr(self.trainer, "max_epochs", None) if self._trainer else None
        if self.hparams.scheduler == "cosine" and max_epochs and max_epochs > 0:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt
