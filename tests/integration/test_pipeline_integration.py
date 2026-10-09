"""Integration tests for the complete machine learning lifecycle.

Executes end-to-end integration:
Data Loading -> Validation -> Partitioning -> Model Fitting -> Evaluation -> Persistence -> Inference.
"""

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from restaurant_demand.config import DataConfig
from restaurant_demand.data.preprocessing import DataPreprocessor
from restaurant_demand.data.validation import DataValidator
from restaurant_demand.models.baseline import BaselineDemandModel
from restaurant_demand.models.evaluate import ModelEvaluator
from restaurant_demand.models.predict import DemandPredictor
from restaurant_demand.models.registry import ModelRegistry


def test_end_to_end_pipeline_integration(tmp_path: Path, sample_merged_df):
    """Executes a complete mini-pipeline on synthetic fixture to verify component interoperability."""
    # 1. Validate Schema
    validator = DataValidator()
    report = validator.validate_raw_schema(sample_merged_df)
    assert report.is_valid is True

    # 2. Chronological Split
    config = DataConfig()
    config.split.train_week_max = 80
    config.split.val_week_max = 110
    config.split.test_week_max = 145

    preprocessor = DataPreprocessor(config=config, base_dir=tmp_path)
    train_df, val_df, test_df = preprocessor.prepare_data(sample_merged_df, save_processed=False)

    y_train = train_df["num_orders"].values
    y_test = test_df["num_orders"].values

    # 3. Fit Baseline Model
    baseline = BaselineDemandModel(group_cols=["center_id", "meal_id"], strategy="median")
    baseline.fit(train_df, y_train)
    baseline_preds = baseline.predict(test_df)
    assert len(baseline_preds) == len(test_df)

    # 4. Fit Candidate Pipeline
    num_cols, cat_cols, bin_cols = preprocessor.get_feature_lists(train_df)
    feature_cols = num_cols + cat_cols + bin_cols

    transformer = preprocessor.build_transformer(num_cols, cat_cols, bin_cols)
    regressor = HistGradientBoostingRegressor(max_iter=20, max_depth=5, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("regressor", regressor),
        ]
    )

    pipeline.fit(train_df[feature_cols], y_train)

    # 5. Evaluate
    evaluator = ModelEvaluator(figures_dir=tmp_path / "figures")
    test_preds = np.maximum(1.0, pipeline.predict(test_df[feature_cols]))
    metrics = evaluator.compute_metrics(y_test, test_preds, model_name="Integration_Test_HGB")

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "wape" in metrics

    # 6. Save Artifact
    registry = ModelRegistry(model_dir=tmp_path / "models")
    model_p, meta_p = registry.save_artifact(
        pipeline=pipeline,
        model_name="Integration_Test_HGB",
        feature_names=feature_cols,
        metrics=metrics,
        hyperparameters={"max_iter": 20},
        split_summary={"train_rows": len(train_df)},
    )
    assert model_p.exists()
    assert meta_p.exists()

    # 7. Reload and Predict
    predictor = DemandPredictor(base_dir=tmp_path)
    predictor.registry = registry
    predictor.pipeline = pipeline
    predictor.feature_names = feature_cols
    predictor.metadata = {"version": "1.0.0", "model_name": "Integration_Test_HGB"}

    sample_test_row = test_df.iloc[0].to_dict()
    prediction_result = predictor.predict_single(sample_test_row)

    assert "predicted_demand" in prediction_result
    assert prediction_result["predicted_demand"] >= 1.0
    assert prediction_result["target_units"] == "orders_per_week"
