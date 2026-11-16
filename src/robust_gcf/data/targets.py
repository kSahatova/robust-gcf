"""Attribute/finding -> binary label policies (positive = CF source class = 1)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from robust_gcf.core.schemas import TargetSpec


def binary_labels(index: pd.DataFrame, target: TargetSpec,
                  label_columns: Sequence[str]) -> pd.DataFrame:
    """Return [cache_index, image_id, split, label] for the rows that belong to the task.

    "absent":     label = column; all rows kept.
    "no_finding": positives = column == 1; negatives = rows with every label column == 0;
                  rows with other findings only are dropped. (Effusion vs No Finding gives
                  ~18% positives on ChestX-ray14, matching the setup.)
    """
    if target.column not in index.columns:
        raise KeyError(f"label column {target.column!r} not in cache index")
    pos = index[target.column].to_numpy() == 1
    if target.negative_policy == "absent":
        keep = np.ones(len(index), dtype=bool)
    elif target.negative_policy == "no_finding":
        healthy = (index[list(label_columns)].to_numpy() == 0).all(axis=1)
        keep = pos | healthy
    else:  # pragma: no cover - guarded by the schema
        raise ValueError(target.negative_policy)
    out = index.loc[keep, ["cache_index", "image_id", "split"]].copy()
    out["label"] = pos[keep].astype(np.int64)
    return out.reset_index(drop=True)


def class_counts(frame: pd.DataFrame) -> dict:
    """{split: {"neg": n0, "pos": n1, "pos_frac": f}}."""
    out = {}
    for split, g in frame.groupby("split"):
        n1 = int(g["label"].sum())
        out[str(split)] = {"neg": len(g) - n1, "pos": n1, "pos_frac": n1 / max(len(g), 1)}
    return out
