from __future__ import annotations

import numpy as np

from ._numeric import (
    EPS,
    finite_pairs,
    pearson_correlation,
    population_variance,
    quantile,
    spearman_correlation,
)


def evaluate_regression(y_true, y_pred) -> dict:
    """Compute regression metrics directly from their mathematical definitions.

    Non-finite truth/prediction pairs are excluded together. Percentage metrics never add an
    arbitrary epsilon to a zero target: MAPE reports both the value and the fraction of rows on
    which it is mathematically defined.
    """

    y, p = finite_pairs(y_true, y_pred)
    if y.size < 2:
        raise ValueError("At least two finite pairs are required")

    error = p - y
    abs_error = np.abs(error)
    sq_error = error * error

    mae = float(np.mean(abs_error))
    mse = float(np.mean(sq_error))
    rmse = float(np.sqrt(mse))
    median_ae = float(np.median(abs_error))
    max_abs_error = float(np.max(abs_error))

    y_mean = float(np.mean(y))
    sse = float(np.sum(sq_error))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = None if sst <= EPS else float(1.0 - sse / sst)

    var_y = population_variance(y)
    var_error = population_variance(error)
    explained_variance = None if var_y <= EPS else float(1.0 - var_error / var_y)

    smape_den = np.abs(y) + np.abs(p)
    smape_terms = np.zeros_like(abs_error)
    valid_smape = smape_den > EPS
    smape_terms[valid_smape] = 2.0 * abs_error[valid_smape] / smape_den[valid_smape]
    smape = float(100.0 * np.mean(smape_terms))

    nonzero_target = np.abs(y) > EPS
    if np.any(nonzero_target):
        mape = float(100.0 * np.mean(abs_error[nonzero_target] / np.abs(y[nonzero_target])))
    else:
        mape = None

    abs_target_sum = float(np.sum(np.abs(y)))
    wape = None if abs_target_sum <= EPS else float(100.0 * np.sum(abs_error) / abs_target_sum)

    return {
        "n": int(y.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "median_ae": median_ae,
        "max_error": max_abs_error,
        "r2": r2,
        "explained_variance": explained_variance,
        "bias": float(np.mean(error)),
        "smape_percent": smape,
        "mape_percent": mape,
        "mape_coverage": float(np.mean(nonzero_target)),
        "wape_percent": wape,
        "pearson_r": pearson_correlation(y, p),
        "spearman_rho": spearman_correlation(y, p),
        "error_p50": quantile(abs_error, 0.50),
        "error_p90": quantile(abs_error, 0.90),
        "error_p95": quantile(abs_error, 0.95),
        "error_p99": quantile(abs_error, 0.99),
    }
