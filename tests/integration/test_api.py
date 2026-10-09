"""Integration tests for FastAPI REST endpoints."""

import numpy as np
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from restaurant_demand.api.app import app
from restaurant_demand.models.predict import DemandPredictor
from restaurant_demand.models.registry import ModelRegistry


def test_api_health_endpoint():
    """Verifies that the /health endpoint returns operational status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "restaurant-order-demand-prediction"


def test_api_predict_endpoints(tmp_path):
    """Verifies single and batch prediction endpoints with mock predictor."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(model_dir=model_dir)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", DummyRegressor(strategy="constant", constant=225.0)),
        ]
    )
    feature_names = ["checkout_price", "base_price"]
    pipeline.fit(np.array([[100.0, 120.0], [200.0, 220.0]]), np.array([225.0, 225.0]))

    registry.save_artifact(
        pipeline=pipeline,
        model_name="MockCandidate",
        feature_names=feature_names,
        metrics={"mae": 8.5},
        hyperparameters={},
        split_summary={},
    )

    predictor = DemandPredictor(base_dir=tmp_path)
    predictor.registry = registry
    predictor.pipeline = pipeline
    predictor.feature_names = feature_names
    predictor.metadata = {
        "model_name": "MockCandidate",
        "version": "1.0.0",
        "target_variable": "num_orders",
        "target_units": "orders_per_week",
        "test_metrics": {"mae": 8.5, "rmse": 12.0},
        "created_at_utc": "2026-10-09T00:00:00Z",
        "expected_input_schema": {},
    }

    # Inject mock into app state
    app.state.predictor = predictor
    client = TestClient(app)

    # 1. Model Info
    info_resp = client.get("/model-info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert info_data["model_name"] == "MockCandidate"
    assert info_data["version"] == "1.0.0"

    # 2. Single Prediction
    payload = {
        "week": 140,
        "center_id": 55,
        "meal_id": 1885,
        "checkout_price": 136.83,
        "base_price": 152.29,
        "emailer_for_promotion": 0,
        "homepage_featured": 0,
    }
    pred_resp = client.post("/predict", json=payload)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["predicted_demand"] == 225.0
    assert pred_data["target_units"] == "orders_per_week"

    # 3. Batch Prediction
    batch_payload = {"orders": [payload, payload]}
    batch_resp = client.post("/predict/batch", json=batch_payload)
    assert batch_resp.status_code == 200
    batch_data = batch_resp.json()
    assert batch_data["total_records"] == 2
    assert len(batch_data["predictions"]) == 2

    # 4. Input validation failure (missing required checkout_price)
    invalid_payload = {"week": 140, "center_id": 55}
    invalid_resp = client.post("/predict", json=invalid_payload)
    assert invalid_resp.status_code == 422
