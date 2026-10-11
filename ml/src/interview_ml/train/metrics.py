"""评分一致性指标（纯 NumPy）：衡量模型分与人工标注分的一致程度。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BUCKET_EDGES = (30, 50, 70, 90)


def to_buckets(scores: Sequence[float]) -> np.ndarray:
    return np.digitize(np.asarray(scores, dtype=float), BUCKET_EDGES)


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # 并列取平均秩
    for v in np.unique(x):
        idx = x == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(_rank(np.asarray(a, float)), _rank(np.asarray(b, float)))


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int], num_classes: int = 5) -> float:
    """二次加权 Cohen's Kappa：有序分档一致性的常用指标（1 完全一致，0 相当于随机）。"""
    x, y = np.asarray(a, int), np.asarray(b, int)
    observed = np.zeros((num_classes, num_classes))
    for i, j in zip(x, y):
        observed[i, j] += 1
    weights = np.array([[(i - j) ** 2 for j in range(num_classes)] for i in range(num_classes)]) / (num_classes - 1) ** 2
    expected = np.outer(np.bincount(x, minlength=num_classes), np.bincount(y, minlength=num_classes)) / max(len(x), 1)
    denom = (weights * expected).sum()
    return 1.0 if denom == 0 else float(1 - (weights * observed).sum() / denom)


def agreement_report(pred: Sequence[float], human: Sequence[float]) -> dict[str, float]:
    p, h = np.asarray(pred, float), np.asarray(human, float)
    return {
        "n": int(len(p)),
        "pearson": round(pearson(p, h), 4),
        "spearman": round(spearman(p, h), 4),
        "qwk": round(quadratic_weighted_kappa(to_buckets(p), to_buckets(h)), 4),
        "mae": round(float(np.abs(p - h).mean()) if len(p) else 0.0, 2),
        "within_10": round(float((np.abs(p - h) <= 10).mean()) if len(p) else 0.0, 4),
        "bucket_accuracy": round(float((to_buckets(p) == to_buckets(h)).mean()) if len(p) else 0.0, 4),
    }
