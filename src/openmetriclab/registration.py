from __future__ import annotations

import numpy as np

from ._numeric import EPS, quantile


def _gray(image) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        if array.shape[-1] < 3:
            raise ValueError("Color images must provide at least three channels")
        # ITU-R BT.709 luminance coefficients, matching the browser implementation.
        array = (
            0.2126 * array[..., 0]
            + 0.7152 * array[..., 1]
            + 0.0722 * array[..., 2]
        )
    if array.ndim != 2:
        raise ValueError("Registration metrics require a 2D grayscale or RGB image")
    if not np.all(np.isfinite(array)):
        raise ValueError("Registration images must contain finite values")
    return array


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    if size < 3 or size % 2 == 0:
        raise ValueError("Gaussian window size must be an odd integer >= 3")
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def _blur_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def _gaussian_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = _blur_axis(image, kernel, axis=0)
    return _blur_axis(out, kernel, axis=1)


def structural_similarity(
    fixed,
    registered,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute local Gaussian-window SSIM directly from the standard SSIM equation."""

    a = _gray(fixed)
    b = _gray(registered)
    if a.shape != b.shape:
        raise ValueError("SSIM images must have identical shapes")
    min_size = min(a.shape)
    if min_size < 3:
        raise ValueError("SSIM requires image dimensions of at least 3 pixels")
    size = min(window_size, min_size if min_size % 2 == 1 else min_size - 1)
    size = max(size, 3)
    kernel = _gaussian_kernel(size=size, sigma=min(sigma, max(0.5, size / 6.0)))

    mu_a = _gaussian_blur(a, kernel)
    mu_b = _gaussian_blur(b, kernel)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    var_a = np.maximum(0.0, _gaussian_blur(a * a, kernel) - mu_a2)
    var_b = np.maximum(0.0, _gaussian_blur(b * b, kernel) - mu_b2)
    cov_ab = _gaussian_blur(a * b, kernel) - mu_ab

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_ab + c1) * (2.0 * cov_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    score_map = numerator / np.maximum(denominator, np.finfo(float).tiny)
    return float(np.mean(score_map))


def normalized_mutual_information(fixed, registered, bins: int = 64) -> float | None:
    """Compute Studholme-style NMI: (H(A) + H(B)) / H(A,B)."""

    a = _gray(fixed).reshape(-1)
    b = _gray(registered).reshape(-1)
    if a.size != b.size:
        raise ValueError("NMI images must have identical sizes")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    joint, _, _ = np.histogram2d(a, b, bins=bins)
    total = float(np.sum(joint))
    if total <= 0:
        return None
    pxy = joint / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    def entropy(probabilities: np.ndarray) -> float:
        nonzero = probabilities[probabilities > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    h_x = entropy(px)
    h_y = entropy(py)
    h_xy = entropy(pxy.reshape(-1))
    return None if h_xy <= EPS else float((h_x + h_y) / h_xy)


def evaluate_registration(fixed, registered, moving=None, data_range=None) -> dict:
    a = _gray(fixed)
    b = _gray(registered)
    if a.shape != b.shape:
        raise ValueError("Fixed and registered images must have identical shapes")

    if data_range is None:
        data_range = float(max(np.max(a), np.max(b)) - min(np.min(a), np.min(b)))
        if data_range <= EPS:
            data_range = 1.0
    else:
        data_range = float(data_range)
        if not np.isfinite(data_range) or data_range <= 0:
            raise ValueError("data_range must be a positive finite number")

    error = b - a
    mse = float(np.mean(error * error))
    rmse = float(np.sqrt(mse))
    reference_rms = float(np.sqrt(np.mean(a * a)))
    nrmse = None if reference_rms <= EPS else float(rmse / reference_rms)
    psnr = float("inf") if mse <= EPS else float(20 * np.log10(data_range) - 10 * np.log10(mse))

    centered_a = a - np.mean(a)
    centered_b = b - np.mean(b)
    ncc_den = np.sqrt(np.sum(centered_a * centered_a) * np.sum(centered_b * centered_b))
    ncc = None if ncc_den <= EPS else float(np.sum(centered_a * centered_b) / ncc_den)

    ssim = structural_similarity(a, b, data_range=data_range)
    nmi = normalized_mutual_information(a, b, bins=64)

    out = {
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "psnr_db": psnr,
        "ssim": ssim,
        "ncc": ncc,
        "nmi": nmi,
    }

    if moving is not None:
        before = evaluate_registration(a, moving, None, data_range=data_range)
        out["before"] = before

        def gain(before_value, after_value, higher_is_better: bool):
            if before_value is None or after_value is None:
                return None
            return float(after_value - before_value) if higher_is_better else float(before_value - after_value)

        out["improvement"] = {
            "ssim": gain(before.get("ssim"), ssim, True),
            "ncc": gain(before.get("ncc"), ncc, True),
            "nmi": gain(before.get("nmi"), nmi, True),
            "rmse": gain(before.get("rmse"), rmse, False),
        }
    return out


def landmark_tre(fixed_xy, registered_xy, spacing=(1.0, 1.0)) -> dict:
    a = np.asarray(fixed_xy, dtype=float)
    b = np.asarray(registered_xy, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Landmarks must be Nx2 arrays")
    if a.shape[0] == 0:
        raise ValueError("At least one landmark pair is required")
    sp = np.asarray(spacing, dtype=float).reshape(1, 2)
    if np.any(sp <= 0) or not np.all(np.isfinite(sp)):
        raise ValueError("spacing must contain positive finite values")
    distances = np.sqrt(np.sum(((b - a) * sp) ** 2, axis=1))
    return {
        "n": len(distances),
        "mean_tre": float(np.mean(distances)),
        "median_tre": float(np.median(distances)),
        "rmse_tre": float(np.sqrt(np.mean(distances * distances))),
        "p95_tre": quantile(distances, 0.95),
        "max_tre": float(np.max(distances)),
    }
