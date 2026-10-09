"""Unit tests for inference and prediction pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from restaurant_demand.exceptions import ModelInferenceError
from restaurant_demand.models.predict import DemandPredictor
from restaurant_demand.models.registry import ModelRegistry


@pytest.fixture
def mock_trained_predictor(tmp_path: Path, sample_inference_record):
    """Creates a lightweight mock DemandPredictor backed by a dummy pipeline."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(model_dir=model_dir)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", DummyRegressor(strategy="constant", constant=150.0)),
        ]
    )

    feature_names = ["checkout_price", "base_price"]
    # Fit dummy on dummy 2D array
    pipeline.fit(np.array([[100.0, 120.0], [200.0, 220.0]]), np.array([150.0, 150.0]))

    registry.save_artifact(
        pipeline=pipeline,
        model_name="MockPipeline",
        feature_names=feature_names,
        metrics={"mae": 10.0},
        hyperparameters={},
        split_summary={},
    )

    predictor = DemandPredictor(base_dir=tmp_path)
    predictor.registry = registry
    predictor.pipeline = pipeline
    predictor.feature_names = feature_names
    predictor.metadata = {"model_name": "MockPipeline", "version": "1.0.0"}
    return predictor


def test_predict_single(mock_trained_predictor, sample_inference_record):
    """Verifies single transaction prediction output schema."""
    result = mock_trained_predictor.predict_single(sample_inference_record)

    assert "predicted_demand" in result
    assert result["predicted_demand"] == 150.0
    assert result["target_units"] == "orders_per_week"
    assert result["center_id"] == sample_inference_record["center_id"]


def test_predict_dataframe(mock_trained_predictor, sample_inference_record):
    """Verifies batch DataFrame prediction preserves row alignment."""
    records = [sample_inference_record, sample_inference_record.copy()]
    df = pd.DataFrame(records)

    scored_df = mock_trained_predictor.predict_dataframe(df)
    assert len(scored_df) == 2
    assert "predicted_demand" in scored_df.columns
    assert (scored_df["predicted_demand"] == 150.0).all()


def test_predict_empty_dataframe(mock_trained_predictor):
    """Verifies that an empty DataFrame raises ModelInferenceError."""
    with pytest.raises(ModelInferenceError, match="empty"):
        mock_trained_predictor.predict_dataframe(pd.DataFrame())
