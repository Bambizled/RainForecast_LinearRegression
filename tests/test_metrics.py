import numpy as np

from src.evaluation.metrics import compute_regression_metrics


def test_compute_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = compute_regression_metrics(y_true, y_pred, feature_count=1)

    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["adjusted_r2"] == 1.0
