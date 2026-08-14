from __future__ import annotations

import numpy as np

from ._numeric import EPS, quantile


def _shifted(mask: np.ndarray, offset: tuple[int, ...]) -> np.ndarray:
    pads = [(1, 1) for _ in range(mask.ndim)]
    padded = np.pad(mask, pads, mode="constant", constant_values=False)
    slices = tuple(slice(1 + delta, 1 + delta + size) for delta, size in zip(offset, mask.shape))
    return padded[slices]


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()
    # A surface voxel/pixel is one whose foreground does not continue in every
    # positive/negative axial direction. This matches the browser boundary rule.
    for axis in range(mask.ndim):
        for direction in (-1, 1):
            offset = [0] * mask.ndim
            offset[axis] = direction
            eroded &= _shifted(mask, tuple(offset))
    return mask & ~eroded


def _surface_points(mask: np.ndarray, spacing=None) -> np.ndarray:
    coords = np.argwhere(_surface(mask)).astype(float)
    if coords.size == 0:
        return coords.reshape(0, mask.ndim)
    if spacing is not None:
        scale = np.asarray(spacing, dtype=float).reshape(1, -1)
        if scale.shape[1] != mask.ndim:
            raise ValueError("spacing length must equal the number of mask dimensions")
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("spacing values must be positive finite numbers")
        coords *= scale
    return coords


def _nearest_distances(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if points.size == 0 or targets.size == 0:
        return np.array([], dtype=float)
    # Exact pairwise nearest-neighbour distances, chunked to bound temporary memory.
    target_cells = 2_000_000
    chunk = max(1, min(points.shape[0], target_cells // max(1, targets.shape[0])))
    out = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk]
        diff = block[:, None, :] - targets[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        out[start : start + block.shape[0]] = np.sqrt(np.min(sq, axis=1))
    return out


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing=None) -> tuple[np.ndarray, np.ndarray]:
    pa = _surface_points(a, spacing)
    pb = _surface_points(b, spacing)
    return _nearest_distances(pa, pb), _nearest_distances(pb, pa)


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(float(denominator)) <= EPS else float(numerator / denominator)


def binary_segmentation_metrics(y_true, y_pred, spacing=None) -> dict:
    a = np.asarray(y_true, dtype=bool)
    b = np.asarray(y_pred, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Masks must have identical shapes")
    if a.size == 0:
        raise ValueError("Masks must not be empty")

    tp = int(np.sum(a & b))
    fp = int(np.sum(~a & b))
    fn = int(np.sum(a & ~b))
    tn = int(np.sum(~a & ~b))
    va = int(np.sum(a))
    vb = int(np.sum(b))

    if va == 0 and vb == 0:
        dice = 1.0
        iou = 1.0
        volume_similarity = 1.0
        hausdorff = 0.0
        hd95 = 0.0
        assd = 0.0
        surface_status = "both_empty"
    elif va == 0 or vb == 0:
        dice = 0.0
        iou = 0.0
        volume_similarity = 0.0
        hausdorff = None
        hd95 = None
        assd = None
        surface_status = "one_empty"
    else:
        dice = float(2 * tp / (2 * tp + fp + fn))
        iou = float(tp / (tp + fp + fn))
        volume_similarity = float(1.0 - abs(va - vb) / (va + vb))
        d_ab, d_ba = _surface_distances(a, b, spacing)
        directed = np.concatenate([d_ab, d_ba])
        hausdorff = float(np.max(directed))
        hd95 = quantile(directed, 0.95)
        assd = float(np.mean(directed))
        surface_status = "exact"

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    balanced = None
    if sensitivity is not None and specificity is not None:
        balanced = float((sensitivity + specificity) / 2.0)

    return {
        "pixels": int(a.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": _ratio(tp, tp + fp),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "pixel_accuracy": float((tp + tn) / a.size),
        "balanced_accuracy": balanced,
        "volume_similarity": volume_similarity,
        "hausdorff": hausdorff,
        "hd95": hd95,
        "assd": assd,
        "surface_status": surface_status,
    }


def evaluate_segmentation(
    y_true,
    y_pred,
    labels=None,
    spacing=None,
    include_background: bool = False,
) -> dict:
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.shape != b.shape:
        raise ValueError("Masks must have identical shapes")
    if a.size == 0:
        raise ValueError("Masks must not be empty")

    if labels is None:
        labs = np.unique(np.concatenate([a.reshape(-1), b.reshape(-1)]))
    else:
        labs = np.asarray(labels)

    rows = []
    for label in labs:
        if not include_background and label == 0:
            continue
        metrics = binary_segmentation_metrics(a == label, b == label, spacing=spacing)
        metrics["class"] = str(label)
        rows.append(metrics)

    keys = [
        "dice",
        "iou",
        "precision",
        "recall_sensitivity",
        "specificity",
        "pixel_accuracy",
        "balanced_accuracy",
        "volume_similarity",
        "hausdorff",
        "hd95",
        "assd",
    ]
    macro = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None and np.isfinite(row[key])]
        macro[key] = float(np.mean(values)) if values else None

    return {
        "shape": list(a.shape),
        "labels": [str(value) for value in labs.tolist()],
        "per_class": rows,
        "macro": macro,
        "include_background": bool(include_background),
    }
