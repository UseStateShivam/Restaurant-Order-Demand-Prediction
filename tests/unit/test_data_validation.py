"""Unit tests for the data validation module."""

import pandas as pd
import pytest

from restaurant_demand.data.validation import DataValidator
from restaurant_demand.exceptions import DataValidationError


def test_validate_raw_schema_success(sample_raw_train_df):
    """Verifies that valid raw DataFrame passes schema validation."""
    validator = DataValidator()
    report = validator.validate_raw_schema(sample_raw_train_df)

    assert report.is_valid is True
    assert report.row_count == len(sample_raw_train_df)
    assert report.target_summary is not None
    assert report.target_summary["min"] >= 20


def test_validate_raw_schema_empty():
    """Verifies that empty DataFrame raises DataValidationError."""
    validator = DataValidator()
    with pytest.raises(DataValidationError, match="dataset is empty"):
        validator.validate_raw_schema(pd.DataFrame())


def test_validate_raw_schema_missing_column(sample_raw_train_df):
    """Verifies that missing required column raises DataValidationError."""
    validator = DataValidator()
    df_missing = sample_raw_train_df.drop(columns=["num_orders"])

    with pytest.raises(DataValidationError, match="Missing required columns"):
        validator.validate_raw_schema(df_missing)


def test_validate_raw_schema_negative_target(sample_raw_train_df):
    """Verifies that negative or zero target values fail validation."""
    validator = DataValidator()
    df_invalid = sample_raw_train_df.copy()
    df_invalid.loc[0, "num_orders"] = -5

    with pytest.raises(DataValidationError, match="non-positive"):
        validator.validate_raw_schema(df_invalid)


def test_validate_raw_schema_invalid_binary(sample_raw_train_df):
    """Verifies that promotional flags outside [0, 1] fail validation."""
    validator = DataValidator()
    df_invalid = sample_raw_train_df.copy()
    df_invalid.loc[0, "emailer_for_promotion"] = 2

    with pytest.raises(DataValidationError, match="invalid values outside"):
        validator.validate_raw_schema(df_invalid)


def test_validate_inference_input_success(sample_inference_record):
    """Verifies that valid inference DataFrame passes validation."""
    validator = DataValidator()
    df = pd.DataFrame([sample_inference_record])
    # Should not raise exception
    validator.validate_inference_input(df)


def test_validate_inference_input_missing_column():
    """Verifies that missing columns during inference raise DataValidationError."""
    validator = DataValidator()
    df = pd.DataFrame([{"week": 140, "center_id": 55}])

    with pytest.raises(DataValidationError, match="Missing required inference columns"):
        validator.validate_inference_input(df)


def test_validate_inference_input_negative_price(sample_inference_record):
    """Verifies that negative checkout price during inference raises DataValidationError."""
    validator = DataValidator()
    bad_record = sample_inference_record.copy()
    bad_record["checkout_price"] = -10.0
    df = pd.DataFrame([bad_record])

    with pytest.raises(DataValidationError, match="checkout_price must be strictly positive"):
        validator.validate_inference_input(df)
