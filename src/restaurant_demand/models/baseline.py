"""Baseline demand models for benchmark comparison.

Establishes an empirical baseline using historical entity-level central tendency
(center_id + meal_id median) with fallback to global training median.

Why this baseline is essential:
In restaurant fulfillment operations, managers often use historical typical volume
for a specific meal at a specific kitchen as a rule of thumb. Evaluating against this
domain heuristic proves whether machine learning algorithms genuinely learn price elasticity,
promotional lift, and seasonality beyond static historical averages.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from restaurant_demand.exceptions import ModelTrainingError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


class BaselineDemandModel(BaseEstimator, RegressorMixin):
    """Predicts demand using historical center-meal group medians with global fallback.

    Implements the standard scikit-learn Regressor interface (fit and predict).
    """

    def __init__(self, group_cols: list[str] | None = None, strategy: str = "median"):
        """Initializes the baseline model.

        Args:
            group_cols: Columns used to group historical demand. Defaults to ['center_id', 'meal_id'].
            strategy: Aggregation strategy ('median' or 'mean').
        """
        self.group_cols = group_cols or ["center_id", "meal_id"]
        self.strategy = strategy
        self.group_estimates_: dict[tuple[Any, ...], float] = {}
        self.global_estimate_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "BaselineDemandModel":
        """Fits baseline predictions by computing group aggregations on training records.

        Args:
            X: Training feature DataFrame containing group_cols.
            y: Target demand series or array.

        Returns:
            self: The fitted baseline estimator.

        Raises:
            ModelTrainingError: If group_cols are missing or y length mismatches.
        """
        if not isinstance(X, pd.DataFrame):
            raise ModelTrainingError(
                "Baseline model requires X to be a pandas DataFrame with named columns."
            )

        missing_group_cols = set(self.group_cols) - set(X.columns)
        if missing_group_cols:
            raise ModelTrainingError(
                f"Missing required grouping columns in X: {missing_group_cols}"
            )

        if len(X) != len(y):
            raise ModelTrainingError(
                f"Mismatch between X rows ({len(X)}) and y target values ({len(y)})."
            )

        df = X[self.group_cols].copy()
        df["target"] = np.asarray(y)

        # Compute global fallback estimate
        if self.strategy == "mean":
            self.global_estimate_ = float(df["target"].mean())
            grouped = df.groupby(self.group_cols)["target"].mean()
        else:
            self.global_estimate_ = float(df["target"].median())
            grouped = df.groupby(self.group_cols)["target"].median()

        # Convert multi-index to lookup dictionary
        self.group_estimates_ = {
            k if isinstance(k, tuple) else (k,): float(v) for k, v in grouped.to_dict().items()
        }

        self.is_fitted_ = True
        logger.info(
            "Fitted BaselineDemandModel: %d group combinations, global fallback=%.2f",
            len(self.group_estimates_),
            self.global_estimate_,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates baseline demand predictions for input rows.

        Args:
            X: Input DataFrame containing grouping columns.

        Returns:
            np.ndarray: Predicted demand values.
        """
        if not self.is_fitted_:
            raise ModelTrainingError(
                "BaselineDemandModel is not fitted. Call fit() before predict()."
            )

        if not isinstance(X, pd.DataFrame):
            raise ModelTrainingError("Baseline model predict expects a pandas DataFrame.")

        predictions = np.empty(len(X), dtype=np.float64)

        # Vectorized or fast dictionary lookup
        if all(col in X.columns for col in self.group_cols):
            keys = list(zip(*(X[col] for col in self.group_cols), strict=False))
            for i, k in enumerate(keys):
                predictions[i] = self.group_estimates_.get(k, self.global_estimate_)
        else:
            # If grouping columns are completely absent, fall back to global estimate
            logger.warning(
                "Grouping columns absent in predict input; defaulting to global estimate."
            )
            predictions.fill(self.global_estimate_)

        return predictions
