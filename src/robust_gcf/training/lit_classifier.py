"""LightningModule around a ClassifierWrapper.

Lightning is a training harness only: the artifact that leaves training is the plain
`ClassifierWrapper` (`lit.model`), saved with `robust_gcf.models.save_classifier`.
Lightning .ckpt files are for resuming runs, never consumed by CF methods.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from robust_gcf.models.wrapper import ClassifierWrapper


def _auroc(proba: torch.Tensor, labels: torch.Tensor) -> float:
    from torchmetrics.functional.classification import binary_auroc

    return float(binary_auroc(proba, labels.long()))


class LitClassifier(pl.LightningModule):
    def __init__(self, model: ClassifierWrapper, lr: float = 1e-4, weight_decay: float = 1e-4,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.register_buffer("pos_weight", None if pos_weight is None
                             else torch.tensor(float(pos_weight)))
        self._val = []

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=self.pos_weight)

    def training_step(self, batch, batch_idx):
        loss = self._loss(self(batch["image"]), batch["label"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["image"])
        self.log("val/loss", self._loss(logits, batch["label"]), prog_bar=True)
        self._val.append((logits.detach().float().cpu(), batch["label"].cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        logits, y = map(torch.cat, zip(*self._val))
        self._val.clear()
        self.log("val/auroc", _auroc(torch.sigmoid(logits), y), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return {"proba": torch.sigmoid(self(batch["image"]).float()).cpu(),
                "label": batch.get("label"), "index": batch["index"]}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)
