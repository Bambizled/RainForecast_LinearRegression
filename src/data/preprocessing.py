import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Raw data file at {file_path} is empty.")
    print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    missing_counts = df_cleaned.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing > 0:
        print("Missing values detected:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  Column '{col}': {count} missing values")
        df_cleaned = df_cleaned.interpolate(method="linear").ffill().bfill()
        print("Missing values handled using linear interpolation and ffill/bfill.")
    else:
        print("No missing values detected.")
    return df_cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df_no_dup = df.copy()
    duplicates_count = df_no_dup.duplicated().sum()

    if duplicates_count > 0:
        df_no_dup = df_no_dup.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates_count} duplicate rows.")
    else:
        print("No duplicate rows detected.")
    return df_no_dup


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

        num_outliers_train = ((X_train_capped[col] < lower_bound) | (X_train_capped[col] > upper_bound)).sum()
        num_outliers_test = ((X_test_capped[col] < lower_bound) | (X_test_capped[col] > upper_bound)).sum()

        X_train_capped[col] = np.clip(X_train_capped[col], lower_bound, upper_bound)
        X_test_capped[col] = np.clip(X_test_capped[col], lower_bound, upper_bound)

        print(f"Column '{col}': Capped {num_outliers_train} train outliers and {num_outliers_test} test outliers using train IQR range [{lower_bound:.2f}, {upper_bound:.2f}].")

    return X_train_capped, X_test_capped


def split_data(
    df: pd.DataFrame, 
    target_column: str = "Precipitation", 
    test_size: float = 0.2, 
    chronological: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    cols_to_drop = [col for col in ["Year", "Month", "Day", "Date"] if col in df.columns]
    
    time_cols = [col for col in ["Year", "Month", "Day"] if col in df.columns]
    if chronological and time_cols:
        df_sorted = df.sort_values(by=time_cols).reset_index(drop=True)
    else:
        df_sorted = df.copy()

    X = df_sorted.drop(columns=[target_column])
    y = df_sorted[target_column]
    
    X_features = X.drop(columns=cols_to_drop)

    n_samples = len(df_sorted)
    if chronological:
        split_idx = int(n_samples * (1 - test_size))
        X_train, X_test = X_features.iloc[:split_idx].copy(), X_features.iloc[split_idx:].copy()
        y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
        print(f"Chronological split completed (Train size: {len(X_train)}, Test size: {len(X_test)}).")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42
        )
        print(f"Random split completed (Train size: {len(X_train)}, Test size: {len(X_test)}).")
    return X_train, X_test, y_train, y_test

def standardize_features( X_train: pd.DataFrame,  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame( scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns,   index=X_test.index )
    print("Features standardized successfully using StandardScaler (fitted on X_train).")
    return X_train_scaled, X_test_scaled
def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,  output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    if isinstance(y_train, pd.Series):
        y_train.to_frame(name="Precipitation").to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    else:
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    if isinstance(y_test, pd.Series):
        y_test.to_frame(name="Precipitation").to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    else:
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Processed CSV files saved successfully to {output_dir}/")
def main() -> None:
    raw_data_path = os.path.join("data", "raw", "weather.csv")
    processed_dir = os.path.join("data", "processed")
    try:
        df = load_data(raw_data_path)
        df_cleaned = handle_missing_values(df)
        df_no_dup = remove_duplicates(df_cleaned)
        target_col = "Precipitation"
        X_train, X_test, y_train, y_test = split_data( df_no_dup, target_column=target_col, test_size=0.2, chronological=True )
        num_cols = ["Specific Humidity", "Relative Humidity", "Temperature"]
        X_train_capped, X_test_capped = handle_outliers_iqr(X_train, X_test, num_cols)
        X_train_scaled, X_test_scaled = standardize_features(X_train_capped, X_test_capped)
        save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, processed_dir)
        print("Data Preprocessing pipeline completed successfully!")
    except Exception as e:
        print(f"Error executing preprocessing pipeline: {e}")
        raise
if __name__ == "__main__":
    main()

X, y = df.drop(columns=['Precipitation']), df['Precipitation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('data/raw/X_train.csv', index=False)
X_test.to_csv('data/raw/X_test.csv', index=False)
y_train.to_csv('data/raw/y_train.csv', index=False)
y_test.to_csv('data/raw/y_test.csv', index=False)