"""Unit tests for feature engineering transformations."""

import pandas as pd
import pytest

from restaurant_demand.exceptions import FeatureEngineeringError
from restaurant_demand.features.build_features import FeatureEngineer


def test_feature_engineer_transformation(sample_merged_df):
    """Verifies calculated features for pricing, interactions, and seasonality."""
    engineer = FeatureEngineer()
    df_transformed = engineer.transform(sample_merged_df)

    # 1. Verify discount calculations
    assert "discount_amount" in df_transformed.columns
    assert "discount_ratio" in df_transformed.columns
    assert "price_ratio" in df_transformed.columns
    assert (df_transformed["discount_amount"] >= 0.0).all()
    assert (df_transformed["discount_ratio"] >= 0.0).all()
    assert (df_transformed["discount_ratio"] <= 1.0).all()

    # 2. Verify promotional interaction flags
    assert "has_any_promotion" in df_transformed.columns
    assert "dual_promotion" in df_transformed.columns
    assert df_transformed["has_any_promotion"].isin([0, 1]).all()
    assert df_transformed["dual_promotion"].isin([0, 1]).all()

    # 3. Verify cyclical seasonal encodings
    assert "week_sin" in df_transformed.columns
    assert "week_cos" in df_transformed.columns
    assert (df_transformed["week_sin"] >= -1.0).all()
    assert (df_transformed["week_sin"] <= 1.0).all()
    assert (df_transformed["week_cos"] >= -1.0).all()
    assert (df_transformed["week_cos"] <= 1.0).all()


def test_feature_engineer_missing_required_column():
    """Verifies that missing pricing or temporal columns raise FeatureEngineeringError."""
    engineer = FeatureEngineer()
    df_bad = pd.DataFrame({"week": [1, 2], "center_id": [1, 2]})

    with pytest.raises(FeatureEngineeringError, match="missing required input columns"):
        engineer.transform(df_bad)
