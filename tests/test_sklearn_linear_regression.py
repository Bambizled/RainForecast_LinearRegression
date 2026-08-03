from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sklearn_linear_regression import SklearnLinearRegression


def test_sklearn_linear_regression_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = SklearnLinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.allclose(predictions, y, atol=1e-5)
    assert np.isclose(model.model.intercept_, 1.0, atol=1e-5)
    assert np.isclose(model.model.coef_[0], 2.0, atol=1e-5)
