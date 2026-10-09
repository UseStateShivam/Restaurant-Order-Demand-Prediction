"""Feature engineering module for Restaurant Order Demand Prediction.

Transforms raw transaction and catalog attributes into highly predictive signals:
1. Pricing and Discount Dynamics:
   - discount_amount: Absolute monetary reduction (base_price - checkout_price).
   - discount_ratio: Relative discount fraction, capturing consumer price sensitivity.
   - price_ratio: Effective checkout price relative to reference base price.
2. Promotional Interactions:
   - has_any_promotion: Binary flag indicating active email marketing or homepage featuring.
   - dual_promotion: Binary flag when both email and homepage promotions are active simultaneously.
3. Temporal and Seasonal Cycles:
   - week_of_year: Cyclical calendar week (1 to 52).
   - week_sin, week_cos: Fourier sinusoidal encodings of seasonality, preserving continuous distance
     between week 52 and week 1.
   - quarter: Fiscal/calendar quarter proxy (1 to 4).
4. Referential Metadata Lookup:
   - If center or meal descriptive attributes are absent at inference time, they are safely looked up
     from local catalog reference tables.

Leakage Prevention Design:
All engineered features rely exclusively on information available at or before the decision cutoff.
No future sales targets, future promotions, or reverse-aggregated statistics from validation/test
intervals are used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from restaurant_demand.config import DataConfig, get_project_root
from restaurant_demand.exceptions import FeatureEngineeringError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


class FeatureEngineer:
    """Computes engineered features for training and real-time inference."""

    def __init__(self, config: DataConfig | None = None, base_dir: Path | None = None):
        """Initializes the FeatureEngineer.

        Args:
            config: Optional DataConfig instance.
            base_dir: Optional project root override.
        """
        self.config = config or DataConfig()
        self.root = base_dir or get_project_root()
        self._center_catalog: pd.DataFrame | None = None
        self._meal_catalog: pd.DataFrame | None = None

    def _load_catalogs(self) -> None:
        """Loads static center and meal metadata catalogs for inference enrichment."""
        if self._center_catalog is None:
            c_path = self.root / self.config.raw.center_path
            if c_path.exists():
                self._center_catalog = pd.read_csv(c_path)
            else:
                logger.warning(
                    "Center catalog not found at %s. Inference must supply metadata.", c_path
                )

        if self._meal_catalog is None:
            m_path = self.root / self.config.raw.meal_path
            if m_path.exists():
                self._meal_catalog = pd.read_csv(m_path)
            else:
                logger.warning(
                    "Meal catalog not found at %s. Inference must supply metadata.", m_path
                )

    def enrich_from_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches DataFrame with center and meal metadata if not already present.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Enriched DataFrame.
        """
        result = df.copy()
        self._load_catalogs()

        # Check and join center metadata if center_type or op_area is absent
        if (
            "center_type" not in result.columns or "op_area" not in result.columns
        ) and self._center_catalog is not None:
            if "center_id" in result.columns:
                result = result.merge(self._center_catalog, on="center_id", how="left")

        # Check and join meal metadata if category or cuisine is absent
        if (
            "category" not in result.columns or "cuisine" not in result.columns
        ) and self._meal_catalog is not None:
            if "meal_id" in result.columns:
                result = result.merge(self._meal_catalog, on="meal_id", how="left")

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates all domain-engineered features from raw or merged columns.

        Args:
            df: Raw or merged DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing original and newly computed features.

        Raises:
            FeatureEngineeringError: If required basic columns are absent.
        """
        required = {
            "checkout_price",
            "base_price",
            "week",
            "emailer_for_promotion",
            "homepage_featured",
        }
        missing = required - set(df.columns)
        if missing:
            raise FeatureEngineeringError(
                f"Cannot generate features; missing required input columns: {missing}"
            )

        data = self.enrich_from_catalog(df)

        try:
            # 1. Price and discount signals
            # Base price is the benchmark non-discounted catalog price.
            # Checkout price is what the consumer actually paid after promotional discounting.
            base_safe = np.maximum(data["base_price"].values, 1e-4)
            data["discount_amount"] = np.maximum(
                0.0, data["base_price"].values - data["checkout_price"].values
            )
            data["discount_ratio"] = np.clip(data["discount_amount"].values / base_safe, 0.0, 1.0)
            data["price_ratio"] = np.clip(data["checkout_price"].values / base_safe, 0.0, 2.0)

            # 2. Promotional interactions
            # Both promotions simultaneously often exhibit non-linear compounding effects on order spikes.
            emailer = data["emailer_for_promotion"].astype(int).values
            featured = data["homepage_featured"].astype(int).values
            data["has_any_promotion"] = ((emailer == 1) | (featured == 1)).astype(int)
            data["dual_promotion"] = ((emailer == 1) & (featured == 1)).astype(int)

            # 3. Cyclical annual seasonality
            # The dataset spans weeks 1 to 145. 52 weeks represent an approximate annual cycle.
            # Fourier sine and cosine encodings prevent artificial discontinuity between week 52 and week 1.
            week_num = data["week"].astype(int).values
            week_of_year = ((week_num - 1) % 52) + 1
            data["week_of_year"] = week_of_year
            data["week_sin"] = np.sin(2.0 * np.pi * week_of_year / 52.0)
            data["week_cos"] = np.cos(2.0 * np.pi * week_of_year / 52.0)
            data["quarter"] = ((week_of_year - 1) // 13) + 1

            # Clean any potential NaN or Inf from divisions
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)

            logger.debug(
                "Successfully transformed %d rows into %d features", len(data), len(data.columns)
            )
            return data

        except Exception as exc:
            raise FeatureEngineeringError(f"Error during feature generation: {exc}") from exc
