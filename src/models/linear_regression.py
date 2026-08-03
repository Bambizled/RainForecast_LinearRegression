import numpy as np


class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        if y.ndim != 1:
            raise ValueError("y must be a 1D target vector")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n_samples, n_features = X.shape
        self.intercept_ = 0.0
        self.coef_ = np.zeros(n_features, dtype=float)

        for _ in range(self.n_iterations):
            predictions = self.intercept_ + X @ self.coef_
            residuals = predictions - y
            gradient_w = (2 / n_samples) * (X.T @ residuals)
            gradient_b = (2 / n_samples) * np.sum(residuals)
            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_
