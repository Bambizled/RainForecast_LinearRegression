import os
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_processed_data(processed_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def calculate_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    combined = X.copy()
    combined["Target"] = y.values
    corr_matrix = combined.corr()
    target_corr = corr_matrix["Target"].drop(index="Target")
    sorted_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
    return sorted_corr


def check_multicollinearity(X: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            val = upper_tri.loc[idx, col]
            if not np.isnan(val) and val >= threshold:
                collinear_pairs.append({
                    "feature_1": idx,
                    "feature_2": col,
                    "correlation": float(val)
                })
    return collinear_pairs


def select_features(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    min_corr: float = 0.005, 
    max_collinear: float = 0.8
) -> List[str]:
    target_corr = calculate_correlation(X_train, y_train)
    selected_features = [col for col, val in target_corr.items() if abs(val) >= min_corr]
    if not selected_features:
        selected_features = list(X_train.columns)

    X_selected = X_train[selected_features]
    collinear_pairs = check_multicollinearity(X_selected, threshold=max_collinear)

    features_to_remove = set()
    for pair in collinear_pairs:
        f1 = pair["feature_1"]
        f2 = pair["feature_2"]
        if f1 in features_to_remove or f2 in features_to_remove:
            continue
        f1_corr = abs(target_corr.get(f1, 0))
        f2_corr = abs(target_corr.get(f2, 0))
        if f1_corr >= f2_corr:
            features_to_remove.add(f2)
        else:
            features_to_remove.add(f1)

    final_features = [f for f in selected_features if f not in features_to_remove]
    return final_features


def main() -> None:
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DATA_DIR)
    selected_cols = select_features(
        X_train, y_train, min_corr=0.005, max_collinear=0.8
    )
    X_train_filtered = X_train[selected_cols]
    X_test_filtered = X_test[selected_cols]

    X_train_filtered.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test_filtered.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)


if __name__ == "__main__":
    main()
