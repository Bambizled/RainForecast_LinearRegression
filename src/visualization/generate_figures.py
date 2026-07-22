"""Script to generate and save EDA figures to outputs/figures/."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})


def main():
    data_path = os.path.join("data", "raw", "weather.csv")
    figures_dir = os.path.join("outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    num_cols = ["Specific Humidity", "Relative Humidity", "Temperature", "Precipitation"]

    # 1. Correlation Heatmap
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f",
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of Weather Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved correlation_heatmap.png")

    # 2. Histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color="skyblue", bins=20)
        axes[idx].set_title(f"Distribution of {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
    plt.suptitle("Feature Distributions", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "histogram_features.png"), dpi=150)
    plt.close()
    print("Saved histogram_features.png")

    # 3. Boxplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for idx, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[idx], color="lightgreen", width=0.4)
        axes[idx].set_title(col)
        axes[idx].set_ylabel("")
    plt.suptitle("Boxplots for Outlier Detection", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "boxplot_features.png"), dpi=150)
    plt.close()
    print("Saved boxplot_features.png")

    print("All figures generated successfully!")


if __name__ == "__main__":
    main()
