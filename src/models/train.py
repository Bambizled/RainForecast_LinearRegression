import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

from src.models.linear_regression import LinearRegressionFromScratch
from src.models.sklearn_linear_regression import SklearnLinearRegression
from src.evaluation.metrics import compute_regression_metrics


def main():
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if y_test.ndim == 2 and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    model_scratch = LinearRegressionFromScratch(learning_rate=0.0001, n_iterations=3000)
    model_scratch.fit(X_train, y_train)

    model_sklearn = SklearnLinearRegression()
    model_sklearn.fit(X_train, y_train)

    pred_scratch = model_scratch.predict(X_test)
    pred_sklearn = model_sklearn.predict(X_test)

    metrics_scratch = compute_regression_metrics(y_test, pred_scratch, feature_count=X_train.shape[1])
    metrics_sklearn = compute_regression_metrics(y_test, pred_sklearn, feature_count=X_train.shape[1])

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(pred_scratch, columns=["Predicted"]).to_csv(
        MODELS_OUTPUT_DIR / "predictions_scratch.csv", index=False
    )
    pd.DataFrame(pred_sklearn, columns=["Predicted"]).to_csv(
        MODELS_OUTPUT_DIR / "predictions_sklearn.csv", index=False
    )

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<15} | {'Custom (Scratch)':<16} | {'Scikit-Learn':<16}")
    print("-" * 55)
    for metric in ["mae", "mse", "rmse", "r2", "adjusted_r2"]:
        val_scratch = metrics_scratch.get(metric)
        val_sklearn = metrics_sklearn.get(metric)
        s_scratch = f"{val_scratch:.4f}" if val_scratch is not None else "N/A"
        s_sklearn = f"{val_sklearn:.4f}" if val_sklearn is not None else "N/A"
        print(f"{metric.upper():<15} | {s_scratch:<16} | {s_sklearn:<16}")
    print("=" * 55 + "\n")

    return metrics_scratch, metrics_sklearn


if __name__ == "__main__":
    main()

