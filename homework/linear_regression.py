"""Simple univariate linear regression implementation.

This module provides a tiny, dependency-free LinearRegression class
that can fit a line y = intercept + coef * x using ordinary least
squares. It's intentionally minimal so it's easy to read and works
in the autograder environment without external packages.
"""
from typing import Sequence, List


class LinearRegression:
    """Univariate linear regression (ordinary least squares).

    Attributes:
        coef_ (float): slope of the fitted line.
        intercept_ (float): intercept of the fitted line.
        fitted (bool): whether the model has been fit.
    """

    def __init__(self) -> None:
        self.coef_: float = 0.0
        self.intercept_: float = 0.0
        self.fitted: bool = False

    def fit(self, X: Sequence[float], y: Sequence[float]) -> None:
        """Fit the model to 1D data X and target y.

        Args:
            X: sequence of feature values (numeric)
            y: sequence of target values (numeric)

        Raises:
            ValueError: if inputs are empty, lengths mismatch, or X has zero variance.
        """
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y must have same non-zero length")

        n = len(X)
        mean_x = sum(X) / n
        mean_y = sum(y) / n

        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(X, y))
        den = sum((xi - mean_x) ** 2 for xi in X)

        if den == 0:
            raise ValueError("Variance of X is zero; cannot fit linear model")

        self.coef_ = num / den
        self.intercept_ = mean_y - self.coef_ * mean_x
        self.fitted = True

    def predict(self, X: Sequence[float]) -> List[float]:
        """Predict targets for 1D sequence X.

        Raises:
            ValueError: if model is not fitted.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted")
        return [self.intercept_ + self.coef_ * xi for xi in X]


def fit_predict(X: Sequence[float], y: Sequence[float], X_pred: Sequence[float]) -> List[float]:
    """Convenience: fit model on (X, y) and predict on X_pred."""
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_pred)