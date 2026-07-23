from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.linear_regression import LinearRegressionFromScratch


def test_fit_predict_on_linear_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=20000)
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.allclose(predictions, y, atol=0.2)
    assert np.isclose(model.intercept_, 1.0, atol=0.2)
    assert np.isclose(model.coef_[0], 2.0, atol=0.2)


def test_fit_accepts_dataframe_target():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = pd.DataFrame({"rain": [1.0, 3.0, 5.0, 7.0]})

    model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=20000)
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.allclose(predictions, y.values.ravel(), atol=0.2)
    assert np.isclose(model.intercept_, 1.0, atol=0.2)
    assert np.isclose(model.coef_[0], 2.0, atol=0.2)
