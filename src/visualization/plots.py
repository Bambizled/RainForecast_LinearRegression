import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures"

NUM_COLS = ["temperature", "humidity", "pressure", "visibility", "cloudcover", "wind_speed", "precipitation"]


def load_raw_sample(data_dir: Path = RAW_DATA_DIR, max_rows: int = 100000) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "weather-vn-*.csv")))
    if not files:
        raise FileNotFoundError(f"No weather CSV files found in {data_dir}")
    df = pd.read_csv(files[0])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    return df


def fig_histograms(raw_df, output_path: Path):
    cols = [c for c in NUM_COLS[:6] if c in raw_df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, col in enumerate(cols):
        sns.histplot(raw_df[col].dropna(), kde=True, ax=axes[idx], color="skyblue", bins=20)
        axes[idx].set_title(f"Distribution of {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
    fig.suptitle("Feature Distributions", y=0.98, fontsize=16)
    fig.tight_layout()
    path = output_path / "histogram_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_correlation_heatmap(raw_df, output_path: Path):
    cols = [c for c in NUM_COLS if c in raw_df.columns]
    corr_matrix = raw_df[cols].corr()
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of Weather Variables")
    plt.tight_layout()
    path = output_path / "correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)


def fig_boxplots(raw_df, output_path: Path):
    cols = [c for c in NUM_COLS[:6] if c in raw_df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 5))
    for idx, col in enumerate(cols):
        sns.boxplot(y=raw_df[col].dropna(), ax=axes[idx], color="lightgreen", width=0.4)
        axes[idx].set_title(col)
        axes[idx].set_ylabel("")
    fig.suptitle("Boxplots for Outlier Detection", y=0.98, fontsize=16)
    fig.tight_layout()
    path = output_path / "boxplot_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gradient_descent_with_history(X, y, learning_rate=0.0001, n_iterations=1000):
    n_samples, n_features = X.shape
    b, w = 0.0, np.zeros(n_features)
    losses = []
    for _ in range(n_iterations):
        pred = b + X @ w
        resid = pred - y
        losses.append(np.mean(resid ** 2))
        grad_w = (2 / n_samples) * (X.T @ resid)
        grad_b = (2 / n_samples) * np.sum(resid)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return b, w, losses


def fig_loss_curve(X_train, y_train, output_path: Path, learning_rate=0.0001, n_iterations=1000):
    if len(X_train) > 50000:
        indices = np.random.choice(len(X_train), size=50000, replace=False)
        X_sub, y_sub = X_train[indices], y_train[indices]
    else:
        X_sub, y_sub = X_train, y_train

    _, _, losses = _gradient_descent_with_history(X_sub, y_sub, learning_rate, n_iterations)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(losses, color="#c1440e", linewidth=1.6)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Gradient Descent Convergence Curve\n(learning_rate = {learning_rate}, n_iterations = {n_iterations})")
    fig.tight_layout()
    path = output_path / "fig_loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_predictions_comparison(y_true, y_pred_custom, y_pred_sklearn, output_path: Path):
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), size=5000, replace=False)
        y_true, y_pred_custom, y_pred_sklearn = y_true[idx], y_pred_custom[idx], y_pred_sklearn[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Actual vs Predicted Comparison", fontsize=14)

    plot_specs = [
        (axes[0], y_pred_custom, "#E08B75", "Custom Linear Regression (Gradient Descent)"),
        (axes[1], y_pred_sklearn, "#607B96", "scikit-learn Linear Regression"),
    ]
    for ax, y_pred, color, title in plot_specs:
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect Prediction")
        ax.scatter(y_true, y_pred, color=color, alpha=0.5, edgecolors="none", s=20, label="Predictions")
        ax.set_title(title)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()

    fig.tight_layout()
    path = output_path / "predictions_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_residuals(y_true, y_pred_custom, y_pred_sklearn, output_path: Path):
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), size=5000, replace=False)
        y_true, y_pred_custom, y_pred_sklearn = y_true[idx], y_pred_custom[idx], y_pred_sklearn[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_specs = [
        (axes[0], y_pred_custom, "#E08B75", "Custom Linear Regression"),
        (axes[1], y_pred_sklearn, "#607B96", "scikit-learn Linear Regression"),
    ]
    for ax, y_pred, color, title in plot_specs:
        resid = y_true - y_pred
        ax.scatter(y_pred, resid, alpha=0.5, color=color, edgecolor="none", s=20)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(title)
    fig.suptitle("Residual Plots")
    fig.tight_layout()
    path = output_path / "fig_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(output_path: Path = OUTPUT_PATH):
    os.makedirs(output_path, exist_ok=True)
    raw_df = load_raw_sample()

    fig_histograms(raw_df, output_path)
    fig_correlation_heatmap(raw_df, output_path)
    fig_boxplots(raw_df, output_path)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").values
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].values
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].values
    pred_scratch = pd.read_csv(PREDICTIONS_DIR / "predictions_scratch.csv").iloc[:, 0].values
    pred_sklearn = pd.read_csv(PREDICTIONS_DIR / "predictions_sklearn.csv").iloc[:, 0].values

    fig_loss_curve(X_train, y_train, output_path)
    fig_predictions_comparison(y_test, pred_scratch, pred_sklearn, output_path)
    fig_residuals(y_test, pred_scratch, pred_sklearn, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()
    main(Path(args.output_path))