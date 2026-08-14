from __future__ import annotations

from collections.abc import Iterable

import numpy as np

EPS = 1e-12


def as_float_vector(values, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def finite_pairs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = as_float_vector(y_true, "y_true")
    p = as_float_vector(y_pred, "y_pred")
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    keep = np.isfinite(y) & np.isfinite(p)
    return y[keep], p[keep]


def safe_div(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= EPS:
        return None
    return float(numerator / denominator)


def mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def population_variance(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    m = np.mean(x)
    return float(np.mean((x - m) ** 2))


def pearson_correlation(a, b) -> float | None:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Correlation inputs must have the same length >= 2")
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    den = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    if den <= EPS:
        return None
    return float(np.sum(dx * dy) / den)


def average_ranks(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(a, b) -> float | None:
    return pearson_correlation(average_ranks(a), average_ranks(b))


def stable_unique(values: Iterable) -> list:
    out: list = []
    for value in values:
        if not any(_equal(value, seen) for seen in out):
            out.append(value)
    return out


def _equal(a, b) -> bool:
    try:
        value = a == b
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)
    except (TypeError, ValueError):
        return str(a) == str(b)


def quantile(values, q: float) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("quantile requires at least one value")
    return float(np.quantile(x, q, method="linear"))
