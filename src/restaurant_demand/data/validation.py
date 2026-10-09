"""Data validation module for Restaurant Order Demand Prediction.

Implements rigorous schema validation, boundary checks, referential integrity tests,
and data quality reporting. Distinguishes between fatal schema violations and
recoverable data quality warnings.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from restaurant_demand.config import DataConfig
from restaurant_demand.exceptions import DataValidationError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


@dataclass
class ValidationReport:
    """Encapsulates comprehensive data quality and validation statistics."""

    is_valid: bool = True
    row_count: int = 0
    column_count: int = 0
    missing_counts: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    numeric_summaries: dict[str, dict[str, float]] = field(default_factory=dict)
    categorical_cardinalities: dict[str, int] = field(default_factory=dict)
    target_summary: dict[str, float] | None = None
    week_range: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts the report to a serializable dictionary."""
        return asdict(self)


class DataValidator:
    """Validates raw and processed DataFrames against required operational constraints."""

    REQUIRED_RAW_COLUMNS = {
        "id",
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "num_orders",
    }

    REQUIRED_MERGED_COLUMNS = REQUIRED_RAW_COLUMNS | {
        "city_code",
        "region_code",
        "center_type",
        "op_area",
        "category",
        "cuisine",
    }

    REQUIRED_INFERENCE_COLUMNS = {
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
    }

    VALID_CENTER_TYPES = {"TYPE_A", "TYPE_B", "TYPE_C"}
    VALID_CUISINES = {"Continental", "Indian", "Italian", "Thai"}

    def __init__(self, config: DataConfig | None = None):
        """Initializes the DataValidator with configuration parameters."""
        self.config = config or DataConfig()

    def validate_raw_schema(self, df: pd.DataFrame) -> ValidationReport:
        """Validates the schema and domain boundaries of raw training data.

        Args:
            df: Raw DataFrame containing demand transactions.

        Returns:
            ValidationReport: Report containing validation status, errors, and warnings.

        Raises:
            DataValidationError: If fatal validation errors are present.
        """
        report = ValidationReport()

        if df.empty:
            report.is_valid = False
            report.errors.append("Dataset is empty (0 rows).")
            raise DataValidationError("Raw dataset validation failed: dataset is empty.")

        report.row_count = len(df)
        report.column_count = len(df.columns)
        report.missing_counts = df.isnull().sum().to_dict()

        # Check required columns
        missing_cols = self.REQUIRED_RAW_COLUMNS - set(df.columns)
        if missing_cols:
            report.is_valid = False
            report.errors.append(f"Missing required columns: {sorted(missing_cols)}")

        # Check duplicate rows based on primary transaction key (id or center_id + meal_id + week)
        if "id" in df.columns:
            id_duplicates = df["id"].duplicated().sum()
            if id_duplicates > 0:
                report.is_valid = False
                report.errors.append(
                    f"Duplicate primary key 'id' found: {id_duplicates} occurrences."
                )

        composite_duplicates = df.duplicated(subset=["week", "center_id", "meal_id"]).sum()
        if composite_duplicates > 0:
            report.warnings.append(
                f"Duplicate composite key (week, center_id, meal_id) found: {composite_duplicates} rows."
            )
        report.duplicate_rows = composite_duplicates

        # Check target variable validity
        if "num_orders" in df.columns:
            if df["num_orders"].isnull().any():
                report.is_valid = False
                report.errors.append("Target 'num_orders' contains null values.")
            if (df["num_orders"] <= 0).any():
                report.is_valid = False
                report.errors.append(
                    "Target 'num_orders' contains non-positive values (must be > 0)."
                )
            if not np.issubdtype(df["num_orders"].dtype, np.number):
                report.is_valid = False
                report.errors.append("Target 'num_orders' is not a numeric datatype.")
            else:
                report.target_summary = {
                    "count": float(df["num_orders"].count()),
                    "mean": float(df["num_orders"].mean()),
                    "std": float(df["num_orders"].std()),
                    "min": float(df["num_orders"].min()),
                    "median": float(df["num_orders"].median()),
                    "max": float(df["num_orders"].max()),
                }

        # Validate pricing boundaries
        if "checkout_price" in df.columns and "base_price" in df.columns:
            if (df["checkout_price"] <= 0).any():
                report.is_valid = False
                report.errors.append("Found non-positive values in 'checkout_price'.")
            if (df["base_price"] <= 0).any():
                report.is_valid = False
                report.errors.append("Found non-positive values in 'base_price'.")
            # Unusually large checkout_price > base_price * 2 is abnormal in retail demand
            anomalous_markup = (df["checkout_price"] > df["base_price"] * 1.5).sum()
            if anomalous_markup > 0:
                report.warnings.append(
                    f"{anomalous_markup} rows exhibit checkout_price significantly exceeding base_price."
                )

        # Validate binary indicator values (must be 0 or 1)
        for binary_col in ["emailer_for_promotion", "homepage_featured"]:
            if binary_col in df.columns:
                invalid_binary = (~df[binary_col].isin([0, 1])).sum()
                if invalid_binary > 0:
                    report.is_valid = False
                    report.errors.append(
                        f"Column '{binary_col}' contains invalid values outside [0, 1]."
                    )

        # Validate temporal boundaries
        if "week" in df.columns:
            report.week_range = {
                "min_week": int(df["week"].min()),
                "max_week": int(df["week"].max()),
                "total_weeks": int(df["week"].nunique()),
            }

        # Check for infinite numbers in any numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if np.isinf(df[c]).any():
                report.is_valid = False
                report.errors.append(f"Column '{c}' contains infinite numeric values.")

        if not report.is_valid:
            logger.error("Validation failed with %d errors: %s", len(report.errors), report.errors)
            raise DataValidationError(f"Data validation failed: {'; '.join(report.errors)}")

        logger.info(
            "Data validation passed successfully (%d rows, %d columns)",
            report.row_count,
            report.column_count,
        )
        return report

    def validate_inference_input(self, df: pd.DataFrame) -> None:
        """Validates incoming feature payloads before model prediction.

        Args:
            df: DataFrame containing input features for prediction.

        Raises:
            DataValidationError: If schema or value constraints are violated.
        """
        if df.empty:
            raise DataValidationError("Inference input is empty.")

        missing_cols = self.REQUIRED_INFERENCE_COLUMNS - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required inference columns: {sorted(missing_cols)}. "
                f"Required columns are: {sorted(self.REQUIRED_INFERENCE_COLUMNS)}"
            )

        if (df["checkout_price"] <= 0).any():
            raise DataValidationError("checkout_price must be strictly positive.")

        if (df["base_price"] <= 0).any():
            raise DataValidationError("base_price must be strictly positive.")

        if (df["week"] <= 0).any():
            raise DataValidationError("week must be a positive integer.")

        for b in ["emailer_for_promotion", "homepage_featured"]:
            if not df[b].isin([0, 1]).all():
                raise DataValidationError(f"Indicator column '{b}' must contain only 0 or 1.")
