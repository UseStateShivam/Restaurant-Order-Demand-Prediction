"""Inference and prediction module for Restaurant Order Demand Prediction.

Provides robust single-observation and batch prediction capabilities using
the persisted model pipeline without retraining.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from restaurant_demand.config import DataConfig, ModelSettings, get_project_root
from restaurant_demand.data.validation import DataValidator
from restaurant_demand.exceptions import ModelInferenceError
from restaurant_demand.features.build_features import FeatureEngineer
from restaurant_demand.logging_config import configure_logger
from restaurant_demand.models.registry import ModelRegistry

logger = configure_logger(__name__)


class DemandPredictor:
    """Manages inference requests, schema validation, and prediction generation."""

    def __init__(
        self,
        model_filename: str | None = None,
        base_dir: Path | None = None,
        model_dir: Path | None = None,
    ):
        """Initializes the predictor by loading the persisted model artifact.

        Args:
            model_filename: Optional name of the saved model pipeline.
            base_dir: Optional project root path override.
            model_dir: Optional explicit model directory path override.
        """
        self.root = base_dir or get_project_root()
        self.data_config = DataConfig()
        self.model_config = ModelSettings()
        resolved_model_dir = model_dir or (self.root / self.model_config.artifacts.model_dir)
        self.registry = ModelRegistry(model_dir=resolved_model_dir)
        self.validator = DataValidator(config=self.data_config)
        self.feature_engineer = FeatureEngineer(config=self.data_config, base_dir=self.root)

        filename = model_filename or self.model_config.artifacts.pipeline_filename
        self.pipeline, self.metadata = self.registry.load_artifact(
            model_filename=filename,
            metadata_filename=self.model_config.artifacts.metadata_filename,
        )
        self.feature_names = self.metadata.get("feature_names", [])
        logger.info(
            "Initialized DemandPredictor with %d expected features", len(self.feature_names)
        )

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates order demand predictions for a DataFrame.

        Preserves row alignment and outputs predicted demand with metadata.

        Args:
            df: DataFrame containing input transaction rows.

        Returns:
            pd.DataFrame: DataFrame containing input columns plus predicted_demand.

        Raises:
            ModelInferenceError: If input is invalid or scoring fails.
        """
        if df.empty:
            raise ModelInferenceError("Input DataFrame is empty.")

        try:
            # 1. Validate incoming inference schema
            self.validator.validate_inference_input(df)

            # 2. Enrich with center and meal catalogs, generate features
            df_featured = self.feature_engineer.transform(df)

            # 3. Check for any missing features expected by the fitted pipeline
            if self.feature_names:
                missing = [col for col in self.feature_names if col not in df_featured.columns]
                if missing:
                    raise ModelInferenceError(
                        f"Missing required features for model pipeline: {missing}"
                    )
                X = df_featured[self.feature_names]
            else:
                X = df_featured

            # 4. Generate raw predictions from scikit-learn pipeline
            raw_preds = self.pipeline.predict(X)

            # 5. Enforce non-negativity constraint
            # Order demand cannot physically be negative; clip to 1.0 minimum order
            clean_preds = np.maximum(1.0, np.round(raw_preds, 1))

            result_df = df.copy()
            result_df["predicted_demand"] = clean_preds
            result_df["target_units"] = "orders_per_week"
            result_df["model_version"] = self.metadata.get("version", "1.0.0")

            return result_df

        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            raise ModelInferenceError(f"Failed to generate predictions: {exc}") from exc

    def predict_single(self, record: dict[str, Any]) -> dict[str, Any]:
        """Generates a prediction for a single transaction dictionary.

        Args:
            record: Dictionary of input feature values.

        Returns:
            Dict[str, Any]: Prediction result containing predicted_demand and metadata.
        """
        df = pd.DataFrame([record])
        scored_df = self.predict_dataframe(df)
        row = scored_df.iloc[0]

        return {
            "center_id": int(row.get("center_id", 0)),
            "meal_id": int(row.get("meal_id", 0)),
            "week": int(row.get("week", 1)),
            "predicted_demand": float(row["predicted_demand"]),
            "target_units": "orders_per_week",
            "model_version": row["model_version"],
            "model_name": self.metadata.get("model_name", "HistGradientBoostingRegressor"),
        }

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generates batch predictions for a list of transaction dictionaries.

        Args:
            records: List of input feature dictionaries.

        Returns:
            List[Dict[str, Any]]: List of prediction result dictionaries.
        """
        df = pd.DataFrame(records)
        scored_df = self.predict_dataframe(df)

        results = []
        for _, row in scored_df.iterrows():
            results.append(
                {
                    "center_id": int(row.get("center_id", 0)),
                    "meal_id": int(row.get("meal_id", 0)),
                    "week": int(row.get("week", 1)),
                    "predicted_demand": float(row["predicted_demand"]),
                    "target_units": "orders_per_week",
                    "model_version": row["model_version"],
                }
            )
        return results

    def predict_file(
        self, input_csv_path: str | Path, output_csv_path: str | Path | None = None
    ) -> pd.DataFrame:
        """Reads a CSV file, generates predictions, and optionally writes output to disk.

        Args:
            input_csv_path: Path to input CSV file.
            output_csv_path: Optional path to output CSV file.

        Returns:
            pd.DataFrame: Scored DataFrame with predicted_demand column.
        """
        input_p = Path(input_csv_path)
        if not input_p.exists():
            raise ModelInferenceError(f"Input file not found: {input_p}")

        logger.info("Reading input data from %s", input_p)
        df = pd.read_csv(input_p)
        scored = self.predict_dataframe(df)

        if output_csv_path:
            out_p = Path(output_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(out_p, index=False)
            logger.info("Saved batch predictions to %s (%d rows)", out_p, len(scored))

        return scored
