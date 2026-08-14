from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import (
    calculate_correlation,
    check_multicollinearity,
    select_features,
)


def test_calculate_correlation():
    X = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "f3": [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    corr = calculate_correlation(X, y)
    assert corr["f1"] == pytest.approx(1.0)
    assert corr["f2"] == pytest.approx(-1.0)


def test_check_multicollinearity():
    X = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "f3": [5.0, 1.0, 4.0, 2.0, 3.0]
    })
    collinear_pairs = check_multicollinearity(X, threshold=0.9)
    assert len(collinear_pairs) == 1
    assert collinear_pairs[0]["feature_1"] == "f1"
    assert collinear_pairs[0]["feature_2"] == "f2"
    assert collinear_pairs[0]["correlation"] == pytest.approx(1.0)


def test_select_features():
    # f1 and f2 are collinear, but f1 is more correlated with y than f2
    # f3 has zero correlation with y
    X = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [1.0, 2.0, 3.0, 4.0, 4.9],
        "f3": [3.0, 3.0, 3.0, 3.0, 3.0]
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    selected = select_features(X, y, min_corr=0.1, max_collinear=0.8)
    assert "f1" in selected
    assert "f2" not in selected  # dropped due to multicollinearity
    assert "f3" not in selected  # dropped due to low correlation
