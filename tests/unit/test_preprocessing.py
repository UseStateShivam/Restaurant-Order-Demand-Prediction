"""Unit tests for data preprocessing and partitioning."""

import numpy as np

from restaurant_demand.config import DataConfig
from restaurant_demand.data.preprocessing import DataPreprocessor


def test_chronological_split(sample_merged_df):
    """Verifies that split partitions preserve chronological ordering without leakage."""
    config = DataConfig()
    # Configure test bounds matching synthetic data
    config.split.train_week_max = 80
    config.split.val_week_max = 110
    config.split.test_week_max = 145

    preprocessor = DataPreprocessor(config=config)
    train_df, val_df, test_df = preprocessor.chronological_split(sample_merged_df)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert train_df["week"].max() <= 80
    assert val_df["week"].min() > 80
    assert val_df["week"].max() <= 110
    assert test_df["week"].min() > 110


def test_column_transformer_fit_and_transform(sample_merged_df):
    """Verifies ColumnTransformer preprocessing pipeline handles mixed types."""
    preprocessor = DataPreprocessor()
    df_featured = preprocessor.feature_engineer.transform(sample_merged_df)

    num_cols, cat_cols, bin_cols = preprocessor.get_feature_lists(df_featured)
    transformer = preprocessor.build_transformer(num_cols, cat_cols, bin_cols)

    feature_cols = num_cols + cat_cols + bin_cols
    X = df_featured[feature_cols]

    # Fit transformer
    transformer.fit(X)
    X_transformed = transformer.transform(X)

    assert X_transformed.shape[0] == len(X)
    assert not np.isnan(X_transformed).any()


def test_column_transformer_unseen_category(sample_merged_df):
    """Verifies that OneHotEncoder handles unknown categories during inference without error."""
    preprocessor = DataPreprocessor()
    df_featured = preprocessor.feature_engineer.transform(sample_merged_df)

    num_cols, cat_cols, bin_cols = preprocessor.get_feature_lists(df_featured)
    transformer = preprocessor.build_transformer(num_cols, cat_cols, bin_cols)

    feature_cols = num_cols + cat_cols + bin_cols
    transformer.fit(df_featured[feature_cols])

    # Create unseen test row
    df_unseen = df_featured.iloc[:1][feature_cols].copy()
    df_unseen["center_type"] = "TYPE_Z_UNSEEN"
    df_unseen["cuisine"] = "Martian_Cuisine"

    # Should transform without raising KeyError / ValueError
    X_unseen_transformed = transformer.transform(df_unseen)
    assert X_unseen_transformed.shape[0] == 1
    assert not np.isnan(X_unseen_transformed).any()
