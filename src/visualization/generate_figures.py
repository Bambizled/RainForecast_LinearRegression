import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_sample_raw_data(data_dir: Path = RAW_DATA_DIR, sample_size: int = 100000) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "weather-vn-*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw weather CSV files found in {data_dir}")
    df = pd.read_csv(files[0])
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_sample_raw_data()
    num_cols = ["temperature", "humidity", "pressure", "visibility", "cloudcover", "wind_speed", "precipitation"]
    num_cols = [c for c in num_cols if c in df.columns]

    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of Weather Variables")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols[:6]):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color="skyblue", bins=20)
        axes[idx].set_title(f"Distribution of {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
    plt.suptitle("Feature Distributions", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "histogram_features.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 6, figsize=(18, 5))
    for idx, col in enumerate(num_cols[:6]):
        sns.boxplot(y=df[col].dropna(), ax=axes[idx], color="lightgreen", width=0.4)
        axes[idx].set_title(col)
        axes[idx].set_ylabel("")
    plt.suptitle("Boxplots for Outlier Detection", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplot_features.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
