import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


RAW_DATA_PATH = os.path.join("data", "raw", "weather.csv")
PROCESSED_DIR = os.path.join("data", "processed")
PREDICTIONS_DIR = os.path.join("outputs", "models")

OUTPUT_PATH = os.path.join("outputs", "figures") 

NUM_COLS = ["Specific Humidity", "Relative Humidity", "Temperature", "Precipitation"]


GD_LEARNING_RATE = 0.00001
GD_N_ITERATIONS = 5000


# Histogram
def fig_histograms(raw_df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(NUM_COLS):
        if _HAS_SEABORN:
            sns.histplot(raw_df[col], kde=True, ax=axes[idx], color="skyblue", bins=20)
        else:
            axes[idx].hist(raw_df[col], bins=20, color="skyblue", edgecolor="white")
        axes[idx].set_title(f"Phân phối của {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Tần suất")
    fig.suptitle("Biểu đồ phân phối các biến thời tiết", y=0.98, fontsize=16)
    fig.tight_layout()
    path = os.path.join(output_path, "histogram_features.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu {path}")


# Correlation Heatmap
def fig_correlation_heatmap(raw_df, output_path):
    corr_matrix = raw_df[NUM_COLS].corr()
    fig = plt.figure(figsize=(8, 6))
    if _HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f",
                    vmin=-1, vmax=1, linewidths=0.5)
    else:
        ax = plt.gca()
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_yticklabels(corr_matrix.columns)
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}", ha="center", va="center")
        fig.colorbar(im)
    plt.title("Ma Trận Tương Quan Của Các Biến Số Thời Tiết")
    plt.tight_layout()
    path = os.path.join(output_path, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu {path}")


# outliers Detection Boxplots
def fig_boxplots(raw_df, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for idx, col in enumerate(NUM_COLS):
        if _HAS_SEABORN:
            sns.boxplot(y=raw_df[col], ax=axes[idx], color="lightgreen", width=0.4)
        else:
            axes[idx].boxplot(raw_df[col], patch_artist=True,
                               boxprops=dict(facecolor="lightgreen"))
        axes[idx].set_title(col)
        axes[idx].set_ylabel("")
    fig.suptitle("Biểu đồ hộp phát hiện Outliers", y=0.98, fontsize=16)
    fig.tight_layout()
    path = os.path.join(output_path, "boxplot_features.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu {path}")


# Gradient Descent Loss Curve
def _gradient_descent_with_history(X, y, learning_rate, n_iterations):
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


def fig_loss_curve(X_train, y_train, output_path,
                    learning_rate=GD_LEARNING_RATE, n_iterations=GD_N_ITERATIONS):
    _, _, losses = _gradient_descent_with_history(X_train, y_train, learning_rate, n_iterations)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(losses, color="#c1440e", linewidth=1.6)
    ax.set_xlabel("Số vòng lặp (iteration)")
    ax.set_ylabel("MSE trên tập huấn luyện")
    ax.set_title(f"Đường cong hội tụ của Gradient Descent\n"
                 f"(learning_rate = {learning_rate}, n_iterations = {n_iterations})")
    fig.tight_layout()
    path = os.path.join(output_path, "fig_loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu {path}  (MSE đầu={losses[0]:.2f}, MSE cuối={losses[-1]:.2f})")


# Actual vs Predicted Comparison Plot
def fig_predictions_comparison(y_true, y_pred_custom, y_pred_sklearn, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Giá trị thực tế và giá trị dự đoán trên tập kiểm tra", fontsize=14)

    plot_specs = [
        (axes[0], y_pred_custom, "#E08B75", "Tự cài đặt (Gradient Descent)"),
        (axes[1], y_pred_sklearn, "#607B96", "scikit-learn"),
    ]
    for ax, y_pred, color, title in plot_specs:
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="Dự đoán hoàn hảo")
        ax.scatter(y_true, y_pred, color=color, alpha=0.7, edgecolors="white", s=50,
                    label="Dữ liệu dự đoán")
        ax.set_title(title)
        ax.set_xlabel("Giá trị thực tế")
        ax.set_ylabel("Giá trị dự đoán")
        ax.legend()

    fig.tight_layout()
    path = os.path.join(output_path, "predictions_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu {path}")


# Residual Plot
def fig_residuals(y_true, y_pred_custom, y_pred_sklearn, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_specs = [
        (axes[0], y_pred_custom, "#E08B75", "Tự cài đặt (Gradient Descent)"),
        (axes[1], y_pred_sklearn, "#607B96", "scikit-learn"),
    ]
    for ax, y_pred, color, title in plot_specs:
        resid = y_true - y_pred
        ax.scatter(y_pred, resid, alpha=0.7, color=color, edgecolor="white", s=50)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Giá trị dự đoán (mm)")
        ax.set_ylabel("Phần dư (mm)")
        ax.set_title(title)
    fig.suptitle("Biểu đồ phần dư (Residual Plot) trên tập kiểm tra")
    fig.tight_layout()
    path = os.path.join(output_path, "fig_residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu {path}")


def main(output_path=OUTPUT_PATH):
    os.makedirs(output_path, exist_ok=True)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_df["Date"] = pd.to_datetime(raw_df[["Year", "Month", "Day"]])

    fig_histograms(raw_df, output_path)
    fig_correlation_heatmap(raw_df, output_path)
    fig_boxplots(raw_df, output_path)

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).iloc[:, 0].values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).iloc[:, 0].values
    pred_scratch = pd.read_csv(os.path.join(PREDICTIONS_DIR, "predictions_scratch.csv")).iloc[:, 0].values
    pred_sklearn = pd.read_csv(os.path.join(PREDICTIONS_DIR, "predictions_sklearn.csv")).iloc[:, 0].values

    fig_loss_curve(X_train, y_train, output_path)
    fig_predictions_comparison(y_test, pred_scratch, pred_sklearn, output_path)
    fig_residuals(y_test, pred_scratch, pred_sklearn, output_path)

    print(f"\nHoàn tất! Toàn bộ 8 biểu đồ đã được lưu vào: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH,
                         help="Thư mục sẽ chứa các file PNG xuất ra (mặc định: outputs/figures_report)")
    args = parser.parse_args()
    main(args.output_path)