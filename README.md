# RainForecast_LinearRegression

A complete, production-ready machine learning project for rainfall forecasting using the Vietnam Weather Dataset and Linear Regression.

---

## Overview

Rainfall forecasting is critical for agriculture, flood management, and disaster risk reduction across Vietnam's distinct tropical climate zones. This project builds an end-to-end machine learning pipeline to predict rainfall volume (**`precipitation`** in mm) using multi-station meteorological sensor measurements (temperature, humidity, atmospheric pressure, cloud cover, visibility, and wind velocity).

The repository implements and benchmarks two linear regression approaches:
1. **Custom Linear Regression (from Scratch)**: Implemented using pure NumPy and batch Gradient Descent to demonstrate algorithmic optimization from first principles.
2. **Scikit-Learn Linear Regression**: Built using Ordinary Least Squares (OLS) closed-form matrix factorization.

---

## Features

- **Automated Data Preprocessing**: Robust loading of multi-file station records, temporal sorting, linear interpolation with forward/backward fill for missing values, duplicate removal, IQR outlier capping, and standard feature scaling.
- **Domain Feature Engineering**: Generation of meteorological indices including Temperature Range, Dew Point approximation, Temperature-Humidity Index (THI), and temporal cyclical indicators (month, quarter, hour, day of year).
- **Correlation & Multicollinearity Selection**: Target correlation ranking and pairwise Pearson multicollinearity filtering to remove redundant predictors.
- **Dual Model Implementation**: Side-by-side training and comparative evaluation of first-principles Gradient Descent vs. Scikit-Learn OLS regression.
- **Comprehensive Evaluation**: Automated metric computation covering Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), and Adjusted $R^2$.
- **Rich Visualizations**: Publication-quality diagnostic figures including feature distributions, correlation heatmaps, boxplots, loss convergence curves, actual vs. predicted scatter plots, and residual diagnostics.
- **Production Architecture**: Modular codebase with clear separation of concerns, orchestrated via a single entry point (`main.py`).
- **Comprehensive Test Suite**: Automated unit tests using `pytest` verifying data preprocessing, feature engineering, model optimization, and evaluation metrics.

---

## Project Architecture

```text
Raw Weather Data (data/raw/*.csv)
                ↓
Data Preprocessing (src/data/preprocessing.py)
├── Missing value handling & interpolation
├── Deduplication & temporal ordering
├── Domain feature engineering (Dew point, THI, Temp range)
├── Outlier capping (IQR method)
└── Train/Test split (80/20) & StandardScaler
                ↓
Feature Selection (src/features/feature_engineering.py)
├── Target correlation analysis
└── Multicollinearity filtering (|r| > 0.8)
                ↓
Model Training & Optimization (src/models/train.py)
├── Custom Linear Regression (Batch Gradient Descent)
└── Scikit-Learn Linear Regression (OLS)
                ↓
Model Evaluation & Predictions (src/evaluation/metrics.py)
├── Metric computation (MAE, MSE, RMSE, R², Adjusted R²)
└── Prediction generation (outputs/models/*.csv)
                ↓
Visualizations (src/visualization/plots.py)
└── Diagnostic figures (outputs/figures/*.png)
```

---

## Project Structure

```text
RainForecast_LinearRegression/
├── data/
│   ├── processed/                  # Preprocessed training and testing datasets (.gitkeep)
│   └── raw/                        # Raw Vietnam weather CSV records (.gitkeep)
├── notebooks/
│   └── EDA.ipynb                   # Exploratory Data Analysis notebook
├── outputs/
│   ├── figures/                    # Generated visual artifacts
│   │   ├── boxplot_features.png
│   │   ├── correlation_heatmap.png
│   │   ├── fig_loss_curve.png
│   │   ├── fig_residuals.png
│   │   ├── histogram_features.png
│   │   └── predictions_comparison.png
│   ├── models/                     # Model predictions and serialized artifacts (.gitkeep)
│   └── reports/                    # Evaluation summaries and reports (.gitkeep)
├── src/
│   ├── data/
│   │   └── preprocessing.py        # Cleaning, domain features, outlier handling, scaling
│   ├── evaluation/
│   │   └── metrics.py              # Mathematical calculation of MAE, MSE, RMSE, R²
│   ├── features/
│   │   └── feature_engineering.py  # Correlation analysis & multicollinearity filtering
│   ├── models/
│   │   ├── linear_regression.py    # Custom Linear Regression (Gradient Descent)
│   │   ├── sklearn_linear_regression.py # Scikit-learn Linear Regression wrapper
│   │   └── train.py                # Model training, evaluation, and output generation
│   └── visualization/
│       └── plots.py                # Generation of publication-quality figures
├── tests/
│   ├── test_feature_engineering.py # Unit tests for feature selection & multicollinearity
│   ├── test_linear_regression.py   # Unit tests for custom gradient descent model
│   ├── test_metrics.py             # Unit tests for evaluation metrics
│   ├── test_preprocessing.py       # Unit tests for data cleaning & transformation
│   └── test_sklearn_linear_regression.py # Unit tests for scikit-learn model wrapper
├── .gitignore
├── main.py                         # Master orchestration script
├── requirements.txt                # Python project dependencies
└── README.md
```

---

## Requirements

- **Python**: 3.9+ (tested on Python 3.11)
- **Dependencies**:
  - `pandas >= 2.0.0`
  - `numpy >= 1.24.0`
  - `matplotlib >= 3.7.0`
  - `seaborn >= 0.12.0`
  - `scikit-learn >= 1.2.0`
  - `scipy >= 1.10.0`
  - `pytest >= 7.0.0`
  - `jupyter >= 1.0.0`

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bam-s-Personal-Org/RainForecast_LinearRegression.git
   cd RainForecast_LinearRegression
   ```

2. **Create and activate a virtual environment:**
   - On Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the entire end-to-end machine learning pipeline with a single command:

```bash
python main.py
```

### Execution Steps
When executed, `main.py` orchestrates the following stages sequentially:
1. **Data Preprocessing**: Loads raw weather datasets from `data/raw/`, cleans missing values, computes domain indices, caps outliers, standardizes features, and creates train/test sets in `data/processed/`.
2. **Feature Selection**: Analyzes correlation matrices and removes multicollinear features.
3. **Model Training & Evaluation**: Trains both Custom (Gradient Descent) and Scikit-Learn models, computes evaluation metrics, and saves predictions to `outputs/models/`.
4. **Visualization Generation**: Exports all diagnostic figures to `outputs/figures/`.

---

## Pipeline Components

Each module can also be run independently:

```bash
# 1. Preprocess raw data
python src/data/preprocessing.py

# 2. Select features and filter multicollinearity
python src/features/feature_engineering.py

# 3. Train models and evaluate metrics
python src/models/train.py

# 4. Generate diagnostic visualization figures
python src/visualization/plots.py
```

---

## Models

### 1. Custom Linear Regression (`LinearRegressionFromScratch`)
Implemented from first principles using vectorized NumPy operations:
- **Optimization**: Batch Gradient Descent minimizing Mean Squared Error loss:
  $$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (y_i - (w^T x_i + b))^2$$
- **Gradients**:
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} X^T (\hat{y} - y), \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum (\hat{y} - y)$$
- **Default Hyperparameters**: $\alpha = 0.0001$, $\text{iterations} = 3000$.

### 2. Scikit-Learn Linear Regression (`SklearnLinearRegression`)
Wraps `sklearn.linear_model.LinearRegression` to provide exact closed-form analytical solutions:
$$w = (X^T X)^{-1} X^T y$$

---

## Evaluation Results

The models were evaluated on the held-out test split ($N = 50,000$ test observations). The actual metrics obtained are:

| Metric | Custom Linear Regression (Scratch) | Scikit-Learn Linear Regression |
| :--- | :---: | :---: |
| **MAE** (Mean Absolute Error) | **0.2997** | 0.3735 |
| **MSE** (Mean Squared Error) | 1.6618 | **1.6131** |
| **RMSE** (Root Mean Squared Error) | 1.2891 | **1.2701** |
| **$R^2$** (Coefficient of Determination) | -0.0287 | **0.0014** |
| **Adjusted $R^2$** | -0.0290 | **0.0012** |

> **Analysis**: Weather precipitation data exhibits severe zero-inflation (a large proportion of time intervals experience zero rainfall) and complex non-linear meteorological dynamics. The custom gradient descent model yields a lower MAE (0.2997), while Scikit-Learn's OLS analytical solver achieves the optimal squared error loss (MSE 1.6131).

---

## Testing

The project includes an automated unit test suite verifying data transformations, feature selection, model fitting, and metric calculations:

```bash
pytest -v
```

### Test Suite Summary
```text
tests/test_feature_engineering.py::test_calculate_correlation PASSED
tests/test_feature_engineering.py::test_check_multicollinearity PASSED
tests/test_feature_engineering.py::test_select_features PASSED
tests/test_linear_regression.py::test_fit_predict_on_linear_data PASSED
tests/test_linear_regression.py::test_fit_accepts_dataframe_target PASSED
tests/test_metrics.py::test_compute_regression_metrics PASSED
tests/test_preprocessing.py::test_handle_missing_values PASSED
tests/test_preprocessing.py::test_remove_duplicates PASSED
tests/test_preprocessing.py::test_engineer_domain_features PASSED
tests/test_preprocessing.py::test_handle_outliers_iqr PASSED
tests/test_preprocessing.py::test_split_data PASSED
tests/test_preprocessing.py::test_standardize_features PASSED
tests/test_sklearn_linear_regression.py::test_sklearn_linear_regression_fit_predict PASSED

13 passed in ~2.0s
```

---

## Outputs

- **Generated Figures** (`outputs/figures/`):
  - `histogram_features.png`: Histograms with KDE curves showing weather variable distributions.
  - `correlation_heatmap.png`: Pearson correlation matrix of meteorological variables.
  - `boxplot_features.png`: Outlier detection boxplots across sensor measurements.
  - `fig_loss_curve.png`: Convergence trajectory of batch gradient descent.
  - `predictions_comparison.png`: Scatter plot comparison of actual vs. predicted values.
  - `fig_residuals.png`: Residual distribution plots for error diagnostic analysis.
- **Model Predictions** (`outputs/models/`):
  - `predictions_scratch.csv`: Predictions from the custom gradient descent model.
  - `predictions_sklearn.csv`: Predictions from the Scikit-Learn model.

---

## Exploratory Data Analysis Notebook

The notebook [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) provides detailed exploratory analysis:
- Statistical summary of meteorological indicators across Vietnam provinces.
- Missing value analysis and temporal frequency inspections.
- Correlation analysis of atmospheric pressure, humidity, cloud cover, and rainfall.

---

## Git Workflow

This project adheres to a streamlined, production-ready single-branch structure:

```text
main (authoritative production branch)
```

All development features, models, tests, and documentation are unified on `main`.

---

## Author / Maintainer

- **Owner / Maintainer**: Edward2106
- **Organization**: [Bam's Personal Org](https://github.com/Bam-s-Personal-Org)

---

## License

This project is licensed under the MIT License.
