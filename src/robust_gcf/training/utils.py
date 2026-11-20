"""Decision-threshold selection on the validation set (numpy only). Predict positive iff p >= t."""
from __future__ import annotations

import numpy as np

METHODS = ("fixed", "youden", "f1")


def choose_threshold(proba, labels, method: str = "youden", fixed: float = 0.5) -> float:
    if method == "fixed":
        return float(fixed)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    p = np.asarray(proba, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes to choose a threshold")
    order = np.argsort(-p, kind="mergesort")
    p, y = p[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    last_of_tie = np.r_[p[1:] != p[:-1], True]  # threshold = each distinct score
    tp, fp, thr = tp[last_of_tie], fp[last_of_tie], p[last_of_tie]
    tpr, fpr = tp / n_pos, fp / n_neg
    if method == "youden":
        score = tpr - fpr
    else:
        prec = tp / (tp + fp)
        score = 2 * prec * tpr / np.maximum(prec + tpr, 1e-12)
    return float(thr[int(np.argmax(score))])
