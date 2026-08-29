from __future__ import annotations

import numpy as np


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _worst_window(error: np.ndarray, width: int) -> float:
    if not len(error):
        return float("nan")
    width = min(width, len(error))
    squared = np.square(error)
    sums = np.convolve(squared, np.ones(width), mode="valid")
    return float(np.sqrt(np.max(sums) / width))


def _best_lag(truth: np.ndarray, prediction: np.ndarray, limit: int = 48) -> tuple[int, float]:
    best_lag, best_corr = 0, float("-inf")
    for lag in range(-limit, limit + 1):
        if lag < 0:
            left, right = truth[-lag:], prediction[:lag]
        elif lag > 0:
            left, right = truth[:-lag], prediction[lag:]
        else:
            left, right = truth, prediction
        corr = _correlation(left, right)
        if np.isfinite(corr) and corr > best_corr:
            best_lag, best_corr = lag, corr
    return best_lag, best_corr


def compute_metrics(truth, prediction) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if truth.shape != prediction.shape or not len(truth):
        raise ValueError("truth and prediction must be equal non-empty vectors")
    error = prediction - truth
    absolute = np.abs(error)
    squared = np.square(error)
    mse = float(np.mean(squared))
    denominator = float(np.sum(np.square(truth - truth.mean())))
    q95 = float(np.quantile(absolute, 0.95))
    tail = absolute[absolute >= q95]
    ordered = np.sort(squared)[::-1]
    total_sse = float(ordered.sum())
    def share(fraction):
        count = max(1, int(np.ceil(len(ordered) * fraction)))
        return float(ordered[:count].sum() / total_sse) if total_sse else 0.0
    lag, lag_corr = _best_lag(truth, prediction)
    truth_delta = np.diff(truth)
    prediction_delta = np.diff(prediction)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(absolute)),
        "medae": float(np.median(absolute)),
        "r2": float(1.0 - squared.sum() / denominator) if denominator else float("nan"),
        "pearson": _correlation(truth, prediction),
        "std_ratio": float(np.std(prediction) / np.std(truth)) if np.std(truth) else float("nan"),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "p95_abs_error": q95,
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "cvar95": float(tail.mean()),
        "top_1pct_sse_share": share(0.01),
        "top_5pct_sse_share": share(0.05),
        "top_10pct_sse_share": share(0.10),
        "worst_24h_rmse": _worst_window(error, 24),
        "worst_72h_rmse": _worst_window(error, 72),
        "best_lag": int(lag),
        "lag_adjusted_pearson": float(lag_corr),
        "residual_mean": float(error.mean()),
        "residual_std": float(error.std()),
        "residual_std_ratio": float(error.std() / truth.std()) if truth.std() else float("nan"),
        "residual_pearson": _correlation(truth_delta, prediction_delta),
    }

