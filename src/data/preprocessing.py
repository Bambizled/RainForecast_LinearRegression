import os
import glob
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_data(data_dir: Path = RAW_DATA_DIR, target_max_rows: int = 250000) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "weather-vn-*.csv")))
    if not files:
        raise FileNotFoundError(f"No weather CSV files found in {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise ValueError("Combined raw dataset is empty.")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values(by="time").reset_index(drop=True)

    if target_max_rows and len(df) > target_max_rows:
        step = max(1, len(df) // target_max_rows)
        df = df.iloc[::step].reset_index(drop=True)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    if df_cleaned.isnull().sum().sum() > 0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(method="linear").ffill().bfill()
        df_cleaned = df_cleaned.ffill().bfill()
    return df_cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    if "temp_max" in df_feat.columns and "temp_min" in df_feat.columns:
        df_feat["temperature_range"] = df_feat["temp_max"] - df_feat["temp_min"]

    if "temperature" in df_feat.columns and "humidity" in df_feat.columns:
        df_feat["dew_point_approximation"] = df_feat["temperature"] - ((100 - df_feat["humidity"]) / 5)
        df_feat["temperature_humidity_index"] = df_feat["temperature"] - (
            0.55 - 0.55 * (df_feat["humidity"] / 100)
        ) * (df_feat["temperature"] - 14.5)

    if "time" in df_feat.columns and pd.api.types.is_datetime64_any_dtype(df_feat["time"]):
        df_feat["month"] = df_feat["time"].dt.month
        df_feat["quarter"] = df_feat["time"].dt.quarter
        df_feat["hour"] = df_feat["time"].dt.hour
        df_feat["day_of_year"] = df_feat["time"].dt.dayofyear

    return df_feat


def handle_outliers_iqr(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()

    for col in columns:
        if col not in X_train_capped.columns:
            continue
        q1 = X_train_capped[col].quantile(0.25)
        q3 = X_train_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        X_train_capped[col] = np.clip(X_train_capped[col], lower_bound, upper_bound)
        X_test_capped[col] = np.clip(X_test_capped[col], lower_bound, upper_bound)

    return X_train_capped, X_test_capped


def split_data(
    df: pd.DataFrame, 
    target_column: str = "precipitation", 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    y = df[target_column]
    
    cols_to_drop = [
        "time", "province", "city", "weather_main", "weather_description", 
        "weather_icon", "weather_code", target_column
    ]
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    X_features = df.drop(columns=existing_drops)

    numeric_features = X_features.select_dtypes(include=[np.number]).columns
    X_features = X_features[numeric_features]

    n_samples = len(df)
    split_idx = int(n_samples * (1 - test_size))
    X_train = X_features.iloc[:split_idx].copy()
    X_test = X_features.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled


def save_processed_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    output_dir: Path = PROCESSED_DATA_DIR
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False, float_format="%.5f", chunksize=50000)
    X_test.to_csv(output_dir / "X_test.csv", index=False, float_format="%.5f", chunksize=50000)
    if isinstance(y_train, pd.Series):
        y_train.to_frame(name="precipitation").to_csv(output_dir / "y_train.csv", index=False, float_format="%.5f", chunksize=50000)
    else:
        y_train.to_csv(output_dir / "y_train.csv", index=False, float_format="%.5f", chunksize=50000)
    if isinstance(y_test, pd.Series):
        y_test.to_frame(name="precipitation").to_csv(output_dir / "y_test.csv", index=False, float_format="%.5f", chunksize=50000)
    else:
        y_test.to_csv(output_dir / "y_test.csv", index=False, float_format="%.5f", chunksize=50000)


def main() -> None:
    df = load_data(RAW_DATA_DIR, target_max_rows=250000)
    df_cleaned = handle_missing_values(df)
    df_no_dup = remove_duplicates(df_cleaned)
    df_featured = engineer_domain_features(df_no_dup)
    
    target_col = "precipitation"
    X_train, X_test, y_train, y_test = split_data(
        df_featured, target_column=target_col, test_size=0.2
    )
    
    num_cols = ["temperature", "humidity", "pressure", "visibility", "cloudcover", "wind_speed"]
    num_cols = [c for c in num_cols if c in X_train.columns]
    X_train_capped, X_test_capped = handle_outliers_iqr(X_train, X_test, num_cols)
    X_train_scaled, X_test_scaled = standardize_features(X_train_capped, X_test_capped)
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()