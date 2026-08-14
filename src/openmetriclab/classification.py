from __future__ import annotations

from itertools import pairwise

import numpy as np

from ._numeric import EPS, stable_unique


def _label_index(labels: list) -> dict:
    return {str(label): idx for idx, label in enumerate(labels)}


def _confusion_matrix(y_true, y_pred, labels: list) -> np.ndarray:
    index = _label_index(labels)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        ti = index.get(str(truth))
        pi = index.get(str(pred))
        if ti is None or pi is None:
            raise ValueError("All truth/prediction labels must be represented in classes")
        matrix[ti, pi] += 1
    return matrix


def _multiclass_mcc(matrix: np.ndarray) -> float:
    c = float(np.trace(matrix))
    s = float(np.sum(matrix))
    actual = np.sum(matrix, axis=1, dtype=float)
    predicted = np.sum(matrix, axis=0, dtype=float)
    numerator = c * s - float(np.dot(actual, predicted))
    denominator = np.sqrt(
        (s * s - float(np.dot(predicted, predicted)))
        * (s * s - float(np.dot(actual, actual)))
    )
    return 0.0 if denominator <= EPS else float(numerator / denominator)


def _cohen_kappa(matrix: np.ndarray) -> float:
    n = float(np.sum(matrix))
    if n <= 0:
        return 0.0
    observed = float(np.trace(matrix) / n)
    expected = float(np.dot(np.sum(matrix, axis=1), np.sum(matrix, axis=0)) / (n * n))
    return 0.0 if abs(1.0 - expected) <= EPS else float((observed - expected) / (1.0 - expected))


def _binary_curve(y_binary: np.ndarray, scores: np.ndarray) -> tuple[list[dict], list[dict]]:
    if y_binary.size != scores.size:
        raise ValueError("Binary labels and scores must have equal lengths")
    order = np.argsort(-scores, kind="mergesort")
    y = y_binary[order]
    s = scores[order]

    positives = int(np.sum(y))
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return [], []

    roc = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    pr = [{"recall": 0.0, "precision": 1.0, "threshold": float("inf")}]
    tp = 0
    fp = 0
    i = 0
    while i < y.size:
        threshold = s[i]
        j = i
        while j < y.size and s[j] == threshold:
            if y[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / positives
        fpr = fp / negatives
        precision = tp / (tp + fp)
        roc.append({"fpr": float(fpr), "tpr": float(tpr), "threshold": float(threshold)})
        pr.append(
            {
                "recall": float(tpr),
                "precision": float(precision),
                "threshold": float(threshold),
            }
        )
        i = j
    return roc, pr


def _roc_auc(points: list[dict]) -> float | None:
    if len(points) < 2:
        return None
    area = 0.0
    for a, b in pairwise(points):
        area += (b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2.0
    return float(area)


def _average_precision(points: list[dict]) -> float | None:
    if len(points) < 2:
        return None
    area = 0.0
    previous_recall = points[0]["recall"]
    for point in points[1:]:
        delta = point["recall"] - previous_recall
        if delta > 0:
            area += delta * point["precision"]
        previous_recall = point["recall"]
    return float(area)


def _normalize_probabilities(probabilities, n_rows: int, n_classes: int) -> tuple[np.ndarray, bool]:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.shape[0] != n_rows:
        raise ValueError("probabilities row count must match labels")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative")

    if n_classes == 2 and probs.shape[1] == 1:
        positive = probs[:, 0]
        if np.any(positive > 1.0 + 1e-12):
            raise ValueError("binary probabilities must lie in [0, 1]")
        probs = np.column_stack([1.0 - positive, positive])
    elif probs.shape[1] != n_classes:
        raise ValueError("probability column count must match the number of classes")

    row_sum = np.sum(probs, axis=1, keepdims=True)
    if np.any(row_sum <= EPS):
        raise ValueError("every probability row must have positive mass")
    normalized = not np.allclose(row_sum, 1.0, rtol=1e-9, atol=1e-12)
    probs = probs / row_sum
    return probs, normalized


def _probability_metrics(
    y_true,
    labels: list,
    probabilities,
    probability_classes: list | None,
) -> dict:
    prob_labels = list(probability_classes) if probability_classes is not None else list(labels)
    if len(prob_labels) != len(labels):
        raise ValueError("probability classes must match the classification classes")

    probs, was_normalized = _normalize_probabilities(probabilities, len(y_true), len(labels))
    prob_index = _label_index(prob_labels)
    label_index = _label_index(labels)
    if set(prob_index) != set(label_index):
        raise ValueError("probability classes and classification classes must contain the same labels")

    # Reorder columns into the classification-label order.
    reorder = [prob_index[str(label)] for label in labels]
    probs = probs[:, reorder]

    onehot = np.zeros_like(probs)
    for row, value in enumerate(y_true):
        onehot[row, label_index[str(value)]] = 1.0

    eps = 1e-15
    clipped = np.clip(probs, eps, 1.0)
    true_probability = clipped[onehot.astype(bool)]
    out = {
        "log_loss": float(-np.mean(np.log(true_probability))),
        "probabilities_normalized": bool(was_normalized),
    }

    if len(labels) == 2:
        positive_index = 1
        positive = labels[positive_index]
        scores = probs[:, positive_index]
        y01 = np.asarray([1 if str(v) == str(positive) else 0 for v in y_true], dtype=np.int8)
        roc, pr = _binary_curve(y01, scores)
        out.update(
            {
                "roc_auc": _roc_auc(roc),
                "average_precision": _average_precision(pr),
                "brier": float(np.mean((scores - y01) ** 2)),
                "positive_class": str(positive),
            }
        )
    else:
        aucs: list[float] = []
        aps: list[float] = []
        for class_index in range(len(labels)):
            binary = onehot[:, class_index].astype(np.int8)
            roc, pr = _binary_curve(binary, probs[:, class_index])
            auc = _roc_auc(roc)
            ap = _average_precision(pr)
            if auc is not None:
                aucs.append(auc)
            if ap is not None:
                aps.append(ap)
        out.update(
            {
                "roc_auc_ovr_macro": float(np.mean(aucs)) if aucs else None,
                "average_precision_macro": float(np.mean(aps)) if aps else None,
                "brier_multiclass": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
            }
        )
    return out


def evaluate_classification(y_true, y_pred, probabilities=None, classes=None) -> dict:
    """Compute classification metrics without external metric libraries."""

    y = np.asarray(y_true, dtype=object).reshape(-1)
    p = np.asarray(y_pred, dtype=object).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y.size < 2:
        raise ValueError("At least two samples are required")

    labels = list(classes) if classes is not None else stable_unique(list(y) + list(p))
    if not labels:
        raise ValueError("At least one class is required")

    matrix = _confusion_matrix(y, p, labels)
    n = int(np.sum(matrix))
    per_class = []
    recalls = []
    f1_values = []
    supports = []

    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        support = int(np.sum(matrix[idx, :]))
        predicted = int(np.sum(matrix[:, idx]))
        precision = 0.0 if predicted == 0 else tp / predicted
        recall = 0.0 if support == 0 else tp / support
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        per_class.append(
            {
                "class": str(label),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
        recalls.append(recall)
        f1_values.append(f1)
        supports.append(support)

    accuracy = float(np.trace(matrix) / n)
    macro_f1 = float(np.mean(f1_values))
    weighted_f1 = float(np.dot(f1_values, supports) / n)
    balanced_accuracy = float(np.mean(recalls))

    out = {
        "n": n,
        "classes": [str(label) for label in labels],
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "mcc": _multiclass_mcc(matrix),
        "cohen_kappa": _cohen_kappa(matrix),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
    }

    if probabilities is not None:
        out.update(_probability_metrics(y, labels, probabilities, list(classes) if classes is not None else None))
    return out
