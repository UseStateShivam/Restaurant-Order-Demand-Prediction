"""Model evaluation and comprehensive error analysis module.

Calculates key performance metrics (MAE, RMSE, R2, MedAE, WAPE), performs
demand-segment slicing (low, medium, high), inspects residual biases, and generates
diagnostic charts saved to reports/figures/.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from restaurant_demand.config import get_project_root
from restaurant_demand.exceptions import ModelEvaluationError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|y_true - y_pred|) / sum(y_true) * 100
    Unlike MAPE, WAPE does not divide by individual zeros or near-zero observations,
    making it well-behaved for retail and restaurant demand forecasting.

    Args:
        y_true: Array of ground-truth target values.
        y_pred: Array of predicted values.

    Returns:
        float: WAPE percentage.
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)


class ModelEvaluator:
    """Computes regression metrics, demand-tier segmentation, and residual plots."""

    def __init__(self, figures_dir: Path | None = None):
        """Initializes the evaluator.

        Args:
            figures_dir: Optional path for saving visualization figures.
        """
        self.root = get_project_root()
        self.figures_dir = figures_dir or (self.root / "reports" / "figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
    ) -> dict[str, float]:
        """Calculates standard regression evaluation metrics.

        Args:
            y_true: Ground truth target demand.
            y_pred: Model predictions.
            model_name: Identifier for logging.

        Returns:
            Dict[str, float]: Calculated metric values.

        Raises:
            ModelEvaluationError: If input arrays are invalid.
        """
        if len(y_true) != len(y_pred):
            raise ModelEvaluationError(
                f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(root_mean_squared_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))
        medae = float(median_absolute_error(y_t, y_p))
        wape = calculate_wape(y_t, y_p)

        metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "medae": round(medae, 2),
            "wape": round(wape, 2),
        }

        logger.info(
            "%s Metrics: MAE=%.2f | RMSE=%.2f | R2=%.4f | MedAE=%.2f | WAPE=%.2f%%",
            model_name,
            mae,
            rmse,
            r2,
            medae,
            wape,
        )
        return metrics

    def analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df_context: pd.DataFrame | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Performs deep error analysis, segmentation, and bias checks.

        Args:
            y_true: Ground truth demand.
            y_pred: Predicted demand.
            df_context: Optional context DataFrame for joining metadata to worst errors.
            top_k: Number of largest errors to retrieve.

        Returns:
            Dict containing bias statistics, segment breakdown, and worst outliers.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        residuals = y_t - y_p  # positive means underpredicted, negative means overpredicted
        abs_errors = np.abs(residuals)

        underpredicted_mask = residuals > 0
        overpredicted_mask = residuals < 0
        exact_mask = residuals == 0

        # Demand tiers: Low (<100), Medium (100-500), High (>500)
        low_mask = y_t < 100
        med_mask = (y_t >= 100) & (y_t <= 500)
        high_mask = y_t > 500

        def segment_stats(mask: np.ndarray) -> dict[str, float]:
            if not np.any(mask):
                return {"count": 0, "mae": 0.0, "wape": 0.0}
            return {
                "count": int(np.sum(mask)),
                "mae": round(float(np.mean(abs_errors[mask])), 2),
                "wape": round(calculate_wape(y_t[mask], y_p[mask]), 2),
            }

        # Retrieve top-k largest absolute errors
        worst_indices = np.argsort(abs_errors)[-top_k:][::-1]
        worst_cases = []
        for idx in worst_indices:
            case = {
                "actual": float(y_t[idx]),
                "predicted": round(float(y_p[idx]), 1),
                "absolute_error": round(float(abs_errors[idx]), 1),
                "residual": round(float(residuals[idx]), 1),
            }
            if df_context is not None and idx < len(df_context):
                row = df_context.iloc[idx]
                for col in [
                    "center_id",
                    "meal_id",
                    "week",
                    "category",
                    "cuisine",
                    "checkout_price",
                ]:
                    if col in row:
                        val = row[col]
                        if hasattr(val, "item"):
                            val = val.item()
                        case[col] = val
            worst_cases.append(case)

        return {
            "bias_summary": {
                "mean_residual": round(float(np.mean(residuals)), 2),
                "median_residual": round(float(np.median(residuals)), 2),
                "std_residual": round(float(np.std(residuals)), 2),
                "underpredicted_count": int(np.sum(underpredicted_mask)),
                "underpredicted_pct": round(float(np.mean(underpredicted_mask) * 100), 2),
                "overpredicted_count": int(np.sum(overpredicted_mask)),
                "overpredicted_pct": round(float(np.mean(overpredicted_mask) * 100), 2),
                "exact_count": int(np.sum(exact_mask)),
            },
            "demand_tiers": {
                "low_demand_under_100": segment_stats(low_mask),
                "medium_demand_100_to_500": segment_stats(med_mask),
                "high_demand_above_500": segment_stats(high_mask),
            },
            "worst_predictions": worst_cases,
        }

    def generate_evaluation_plots(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "model"
    ) -> dict[str, str]:
        """Generates and persists diagnostic residual charts.

        Args:
            y_true: Actual demand values.
            y_pred: Predicted demand values.
            prefix: Filename prefix for outputs.

        Returns:
            Dict mapping plot name to saved filepath.
        """
        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )
        saved_plots = {}

        # 1. Actual vs Predicted Scatter Plot (sample for performance if dataset is large)
        n_samples = min(len(y_true), 5000)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(y_true), size=n_samples, replace=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            y_true[sample_idx], y_pred[sample_idx], alpha=0.3, edgecolors="none", c="#1f77b4"
        )
        max_val = max(np.percentile(y_true, 99.5), np.percentile(y_pred, 99.5))
        ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect 1:1 Fit")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("Actual Weekly Orders")
        ax.set_ylabel("Predicted Weekly Orders")
        ax.set_title("Actual vs Predicted Demand (Held-Out Test Set)")
        ax.legend()
        plt.tight_layout()
        act_pred_path = self.figures_dir / f"{prefix}_actual_vs_predicted.png"
        fig.savefig(act_pred_path, dpi=200)
        plt.close(fig)
        saved_plots["actual_vs_predicted"] = str(act_pred_path)

        # 2. Residual Distribution Plot
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        # Clip display range to 1st and 99th percentiles for visual clarity
        low_p, high_p = np.percentile(residuals, 1), np.percentile(residuals, 99)
        clipped_residuals = residuals[(residuals >= low_p) & (residuals <= high_p)]
        sns.histplot(clipped_residuals, kde=True, ax=ax, color="#2ca02c", bins=50)
        ax.axvline(0, color="red", linestyle="--", lw=2, label="Zero Bias Line")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Error Distribution (1st-99th percentile)")
        ax.legend()
        plt.tight_layout()
        resid_path = self.figures_dir / f"{prefix}_residuals_distribution.png"
        fig.savefig(resid_path, dpi=200)
        plt.close(fig)
        saved_plots["residuals_distribution"] = str(resid_path)

        logger.info("Saved evaluation charts to %s", self.figures_dir)
        return saved_plots
