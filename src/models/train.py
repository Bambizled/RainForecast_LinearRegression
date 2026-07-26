from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.linear_regression import LinearRegressionFromScratch
from src.models.sklearn_linear_regression import SklearnLinearRegression
from src.evaluation.metrics import compute_regression_metrics

def main():
    X_train_path = 'data/processed/X_train.csv'
    y_train_path = 'data/processed/y_train.csv'
    X_test_path = 'data/processed/X_test.csv'
    y_test_path = 'data/processed/y_test.csv'

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if y_test.ndim == 2 and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    model_scratch = LinearRegressionFromScratch(learning_rate=0.00001, n_iterations=5000)
    model_scratch.fit(X_train, y_train)
    model_sklearn = SklearnLinearRegression()
    model_sklearn.fit(X_train, y_train)

    pred_scratch = model_scratch.predict(X_test)
    pred_sklearn = model_sklearn.predict(X_test)

    metrics_scratch = compute_regression_metrics(y_test, pred_scratch, feature_count=X_train.shape[1])
    metrics_sklearn = compute_regression_metrics(y_test, pred_sklearn, feature_count=X_train.shape[1])

    pred_scratch_df = pd.DataFrame(pred_scratch, columns=['Predicted'])
    pred_sklearn_df = pd.DataFrame(pred_sklearn, columns=['Predicted'])
    pred_scratch_df.to_csv('outputs/models/predictions_scratch.csv', index=False)
    pred_sklearn_df.to_csv('outputs/models/predictions_sklearn.csv', index=False)

    print(f"Intercept (Scratch): {model_scratch.intercept_}")
    print(f"Coefficients (Scratch): {model_scratch.coef_}")
    print("Metrics (Scratch):")
    for name, value in metrics_scratch.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print("--------------------------------------------------")
    print(f"Intercept (Sklearn): {model_sklearn.model.intercept_}")
    print(f"Coefficients (Sklearn): {model_sklearn.model.coef_}")
    print("Metrics (Sklearn):")
    for name, value in metrics_sklearn.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")


if __name__ == "__main__":
    main()

