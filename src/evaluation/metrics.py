import numpy as np


def compute_regression_metrics(y_true, y_pred, feature_count=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    n_samples = y_true.size
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    adjusted_r2 = None
    if feature_count is not None and feature_count >= 1:
        adjusted_r2 = 1.0 - ((1 - r2) * (n_samples - 1) / (n_samples - feature_count - 1)) if n_samples - feature_count - 1 > 0 else 1.0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "adjusted_r2": float(adjusted_r2) if adjusted_r2 is not None else None,
    }
