"""Regression metrics used throughout the project."""

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("Expected matching two-dimensional target and prediction arrays")
    if y_true.shape[0] < 2 or y_true.shape[1] == 0:
        raise ValueError("R-squared requires at least two observations and one output")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite")
    residual_ss = np.square(y_true - y_pred).sum(axis=0)
    total_ss = np.square(y_true - y_true.mean(axis=0)).sum(axis=0)
    # Match sklearn's force_finite convention for constant targets: 1 if exact, 0 otherwise.
    r2_bins = np.where(residual_ss == 0, 1.0, 0.0)
    nonconstant = total_ss > 0
    r2_bins[nonconstant] = 1.0 - residual_ss[nonconstant] / total_ss[nonconstant]
    mse = float(np.square(y_true - y_pred).mean())
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mean_r2": float(r2_bins.mean()),
        "r2_bins": r2_bins.tolist(),
    }
