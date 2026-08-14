from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    handle_missing_values,
    remove_duplicates,
    engineer_domain_features,
    handle_outliers_iqr,
    split_data,
    standardize_features,
)


def test_handle_missing_values():
    df = pd.DataFrame({
        "temperature": [20.0, np.nan, 22.0, np.nan],
        "humidity": [80.0, 85.0, np.nan, 90.0]
    })
    df_cleaned = handle_missing_values(df)
    assert df_cleaned.isnull().sum().sum() == 0
    assert df_cleaned.loc[1, "temperature"] == pytest.approx(21.0)


def test_remove_duplicates():
    df = pd.DataFrame({
        "temperature": [20.0, 20.0, 25.0],
        "humidity": [80.0, 80.0, 85.0]
    })
    df_dedup = remove_duplicates(df)
    assert len(df_dedup) == 2


def test_engineer_domain_features():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2023-01-15 10:00:00", "2023-06-20 14:00:00"]),
        "temperature": [30.0, 25.0],
        "temp_max": [32.0, 28.0],
        "temp_min": [24.0, 20.0],
        "humidity": [80.0, 70.0]
    })
    df_feat = engineer_domain_features(df)
    assert "temperature_range" in df_feat.columns
    assert "dew_point_approximation" in df_feat.columns
    assert "temperature_humidity_index" in df_feat.columns
    assert "month" in df_feat.columns
    assert "quarter" in df_feat.columns
    assert "hour" in df_feat.columns

    assert df_feat.loc[0, "temperature_range"] == pytest.approx(8.0)
    assert df_feat.loc[0, "month"] == 1
    assert df_feat.loc[1, "month"] == 6


def test_handle_outliers_iqr():
    X_train = pd.DataFrame({"temperature": [10.0, 20.0, 22.0, 24.0, 100.0]})
    X_test = pd.DataFrame({"temperature": [5.0, 20.0, 150.0]})
    
    X_train_capped, X_test_capped = handle_outliers_iqr(X_train, X_test, ["temperature"])
    assert X_train_capped["temperature"].max() < 100.0
    assert X_test_capped["temperature"].max() <= X_train_capped["temperature"].max()


def test_split_data():
    df = pd.DataFrame({
        "temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
        "humidity": [70.0, 75.0, 80.0, 85.0, 90.0],
        "precipitation": [0.0, 1.0, 0.5, 2.0, 0.0]
    })
    X_train, X_test, y_train, y_test = split_data(df, target_column="precipitation", test_size=0.4)
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2
    assert "precipitation" not in X_train.columns


def test_standardize_features():
    X_train = pd.DataFrame({"temperature": [10.0, 20.0, 30.0]})
    X_test = pd.DataFrame({"temperature": [20.0, 30.0]})
    
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    assert np.isclose(X_train_scaled["temperature"].mean(), 0.0)
    assert np.isclose(X_train_scaled["temperature"].std(ddof=0), 1.0)
