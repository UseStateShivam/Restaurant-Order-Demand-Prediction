"""Data and feature drift detection module for Restaurant Order Demand Prediction.

Detects statistical distribution shifts between baseline training data and
incoming production inference requests using the Kolmogorov-Smirnov (KS) two-sample
test and Population Stability Index (PSI).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between two samples.

    PSI benchmarks:
    - PSI < 0.1: No significant distribution change.
    - 0.1 <= PSI < 0.2: Moderate shift; monitor closely.
    - PSI >= 0.2: Significant drift; consider retraining pipeline.

    Args:
        baseline: Reference sample values (e.g., training distribution).
        current: Current observation sample values (e.g., inference distribution).
        num_bins: Number of quantile bins to partition data.

    Returns:
        float: Calculated PSI score.
    """
    b_clean = baseline[~np.isnan(baseline)]
    c_clean = current[~np.isnan(current)]

    if len(b_clean) < 10 or len(c_clean) < 10:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(b_clean, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    b_counts, _ = np.histogram(b_clean, bins=bin_edges)
    c_counts, _ = np.histogram(c_clean, bins=bin_edges)

    # Convert to proportions with small epsilon to prevent division by zero
    eps = 1e-4
    b_prop = (b_counts + eps) / (len(b_clean) + eps * num_bins)
    c_prop = (c_counts + eps) / (len(c_clean) + eps * num_bins)

    psi_val = np.sum((c_prop - b_prop) * np.log(c_prop / b_prop))
    return float(np.round(psi_val, 4))


class DriftDetector:
    """Monitors feature and prediction distribution changes between reference and current sets."""

    def __init__(self, reference_df: pd.DataFrame | None = None):
        """Initializes DriftDetector.

        Args:
            reference_df: Reference training baseline DataFrame.
        """
        self.reference_df = reference_df

    def set_reference(self, reference_df: pd.DataFrame) -> None:
        """Sets the baseline reference dataset.

        Args:
            reference_df: Baseline DataFrame.
        """
        self.reference_df = reference_df

    def check_drift(
        self,
        current_df: pd.DataFrame,
        features_to_monitor: list[str] | None = None,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        """Performs statistical drift tests on numerical and categorical features.

        Args:
            current_df: Incoming inference dataset to check.
            features_to_monitor: Optional list of column names to monitor.
            significance_level: p-value threshold below which KS test indicates drift.

        Returns:
            Dict containing per-feature drift results and high-level alert status.
        """
        if self.reference_df is None:
            return {"error": "No reference baseline set for drift comparison."}

        cols = features_to_monitor or [
            "checkout_price",
            "base_price",
            "op_area",
            "emailer_for_promotion",
            "homepage_featured",
        ]

        results = {}
        drift_detected_count = 0

        for col in cols:
            if col not in self.reference_df.columns or col not in current_df.columns:
                continue

            ref_vals = pd.to_numeric(self.reference_df[col], errors="coerce").dropna().values
            cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue

            # Two-sample Kolmogorov-Smirnov test for continuous distributions
            ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
            psi = calculate_psi(ref_vals, cur_vals)

            is_drifted = bool(p_val < significance_level and psi >= 0.1)
            if is_drifted:
                drift_detected_count += 1

            results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
                "psi": psi,
                "drift_detected": is_drifted,
                "ref_mean": round(float(np.mean(ref_vals)), 2),
                "cur_mean": round(float(np.mean(cur_vals)), 2),
            }

        report = {
            "monitored_features_count": len(results),
            "drifted_features_count": drift_detected_count,
            "overall_drift_alert": drift_detected_count > 0,
            "feature_metrics": results,
        }

        logger.info(
            "Drift check completed: %d of %d features flagged.",
            drift_detected_count,
            len(results),
        )
        return report
