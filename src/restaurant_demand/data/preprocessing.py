"""Data preprocessing and partitioning module for Restaurant Order Demand Prediction.

Implements chronological train-validation-test data splitting to avoid temporal leakage,
and builds an end-to-end scikit-learn ColumnTransformer fitted exclusively on training data.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from restaurant_demand.config import DataConfig, get_project_root
from restaurant_demand.exceptions import PreprocessingError
from restaurant_demand.features.build_features import FeatureEngineer
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


class DataPreprocessor:
    """Manages dataset partitioning, feature cleaning, and ColumnTransformer pipeline creation."""

    def __init__(self, config: DataConfig | None = None, base_dir: Path | None = None):
        """Initializes DataPreprocessor with configuration settings.

        Args:
            config: Optional DataConfig instance.
            base_dir: Optional project root override.
        """
        self.config = config or DataConfig()
        self.root = base_dir or get_project_root()
        self.feature_engineer = FeatureEngineer(config=self.config, base_dir=self.root)
        self.column_transformer: ColumnTransformer | None = None
        self.feature_names_in_: list[str] | None = None

    def chronological_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions the dataset chronologically into train, validation, and test subsets.

        Rationale:
        In demand forecasting, standard random cross-validation leaks future patterns into
        past evaluations, artificially inflating offline metrics. A strict chronological
        holdout accurately simulates production deployment where models predict future unseen demand.

        Args:
            df: Merged transactional DataFrame.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            PreprocessingError: If week bounds or records are invalid.
        """
        if "week" not in df.columns:
            raise PreprocessingError("Cannot perform chronological split: 'week' column not found.")

        train_max = self.config.split.train_week_max
        val_max = self.config.split.val_week_max
        test_max = self.config.split.test_week_max

        train_df = df[df["week"] <= train_max].copy()
        val_df = df[(df["week"] > train_max) & (df["week"] <= val_max)].copy()
        test_df = df[(df["week"] > val_max) & (df["week"] <= test_max)].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise PreprocessingError(
                f"Split resulted in empty partitions! Train: {len(train_df)}, "
                f"Val: {len(val_df)}, Test: {len(test_df)}. "
                f"Check week ranges in data.yaml vs actual data."
            )

        logger.info(
            "Chronological split complete: Train=%d rows (weeks 1-%d), "
            "Val=%d rows (weeks %d-%d), Test=%d rows (weeks %d-%d)",
            len(train_df),
            train_max,
            len(val_df),
            train_max + 1,
            val_max,
            len(test_df),
            val_max + 1,
            test_max,
        )

        return train_df, val_df, test_df

    def build_transformer(
        self,
        numerical_cols: list[str],
        categorical_cols: list[str],
        binary_cols: list[str],
    ) -> ColumnTransformer:
        """Constructs a scikit-learn ColumnTransformer.

        Design Principles:
        - Numerical: Median imputation prevents outlier contamination, followed by StandardScaler.
        - Categorical: Handled with OneHotEncoder(handle_unknown='ignore') to gracefully accommodate
          unseen categories at inference time without raising exceptions.
        - Binary: Imputed with most frequent value and passed through without scaling to preserve
          binary interpretability.

        Args:
            numerical_cols: List of numeric feature names.
            categorical_cols: List of categorical feature names.
            binary_cols: List of binary indicator column names.

        Returns:
            ColumnTransformer: Configured preprocessor.
        """
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        bin_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols),
                ("bin", bin_pipeline, binary_cols),
            ],
            remainder="drop",
        )
        return transformer

    def get_feature_lists(self, sample_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """Identifies active feature subsets available in the transformed dataset.

        Args:
            sample_df: DataFrame with engineered features.

        Returns:
            Tuple of (numerical_features, categorical_features, binary_features).
        """
        # Base numericals plus engineered signals
        candidate_num = [
            "checkout_price",
            "base_price",
            "discount_amount",
            "discount_ratio",
            "price_ratio",
            "op_area",
            "week_sin",
            "week_cos",
            "week_of_year",
            "quarter",
        ]
        candidate_cat = ["center_type", "category", "cuisine", "city_code", "region_code"]
        candidate_bin = [
            "emailer_for_promotion",
            "homepage_featured",
            "has_any_promotion",
            "dual_promotion",
        ]

        num_cols = [c for c in candidate_num if c in sample_df.columns]
        cat_cols = [c for c in candidate_cat if c in sample_df.columns]
        bin_cols = [c for c in candidate_bin if c in sample_df.columns]

        return num_cols, cat_cols, bin_cols

    def prepare_data(
        self, raw_df: pd.DataFrame, save_processed: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Applies feature engineering, splits datasets, and optionally persists processed partitions.

        Args:
            raw_df: Merged raw DataFrame.
            save_processed: Whether to persist processed CSV files to disk.

        Returns:
            Tuple of (train_df, val_df, test_df) with engineered features.
        """
        logger.info("Generating engineered features on full merged dataset...")
        df_featured = self.feature_engineer.transform(raw_df)

        train_df, val_df, test_df = self.chronological_split(df_featured)

        if save_processed:
            proc_dir = self.root / Path(self.config.processed.train_path).parent
            proc_dir.mkdir(parents=True, exist_ok=True)

            train_df.to_csv(self.root / self.config.processed.train_path, index=False)
            val_df.to_csv(self.root / self.config.processed.val_path, index=False)
            test_df.to_csv(self.root / self.config.processed.test_path, index=False)
            logger.info("Saved processed datasets to %s", proc_dir)

        return train_df, val_df, test_df
