"""FastAPI application for serving Restaurant Order Demand predictions.

Provides high-performance RESTful inference endpoints with automatic OpenAPI
documentation, structured Pydantic input validation, and graceful error handling.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from restaurant_demand.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderDemandPredictionRequest,
    OrderDemandPredictionResponse,
)
from restaurant_demand.exceptions import ModelInferenceError
from restaurant_demand.logging_config import configure_logger
from restaurant_demand.models.predict import DemandPredictor

logger = configure_logger(__name__)

# Global singleton predictor instance to prevent reloading weights per request
predictor: DemandPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle: loads model artifact once at startup."""
    logger.info("Initializing Restaurant Order Demand Prediction API...")
    if not hasattr(app.state, "predictor") or app.state.predictor is None:
        try:
            app.state.predictor = DemandPredictor()
            logger.info("Model pipeline loaded successfully into app state.")
        except Exception as exc:
            logger.warning(
                "Could not pre-load model artifact at startup: %s. "
                "Endpoints requiring model will return 503 until model is trained.",
                exc,
            )
            app.state.predictor = None
    yield
    logger.info("Shutting down Restaurant Order Demand Prediction API.")


app = FastAPI(
    title="Restaurant Order Demand Prediction API",
    description=(
        "Production REST API for forecasting weekly meal order demand "
        "across restaurant fulfillment centers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_active_predictor() -> DemandPredictor:
    """Helper to retrieve the active predictor or raise 503."""
    pred = getattr(app.state, "predictor", None)
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Train the model first.",
        )
    return pred


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns the operational status of the service."""
    pred = getattr(app.state, "predictor", None)
    return HealthResponse(
        status="healthy" if pred is not None else "degraded",
        service="restaurant-order-demand-prediction",
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Information"])
async def get_model_info() -> ModelInfoResponse:
    """Returns metadata, versioning, and test evaluation metrics for the active model."""
    pred = get_active_predictor()
    meta = pred.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "HistGradientBoostingRegressor"),
        version=meta.get("version", "1.0.0"),
        target_variable=meta.get("target_variable", "num_orders"),
        target_units=meta.get("target_units", "orders_per_week"),
        test_metrics=meta.get("test_metrics", {}),
        created_at_utc=meta.get("created_at_utc", "unknown"),
        expected_input_schema=meta.get("expected_input_schema", {}),
    )


@app.post(
    "/predict",
    response_model=OrderDemandPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_single_order(
    request: OrderDemandPredictionRequest,
) -> OrderDemandPredictionResponse:
    """Generates demand forecast for a single meal-center-week transaction."""
    pred = get_active_predictor()
    try:
        input_dict = request.model_dump(exclude_none=True)
        result = pred.predict_single(input_dict)
        return OrderDemandPredictionResponse(**result)
    except ModelInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unhandled error during prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_batch_orders(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Generates demand forecasts for a batch of meal-center-week transactions."""
    pred = get_active_predictor()
    try:
        input_records = [r.model_dump(exclude_none=True) for r in request.orders]
        results = pred.predict_batch(input_records)
        items = [
            OrderDemandPredictionResponse(
                center_id=r["center_id"],
                meal_id=r["meal_id"],
                week=r["week"],
                predicted_demand=r["predicted_demand"],
                target_units=r["target_units"],
                model_version=r["model_version"],
                model_name=pred.metadata.get("model_name", "HistGradientBoostingRegressor"),
            )
            for r in results
        ]
        return BatchPredictionResponse(
            predictions=items,
            total_records=len(items),
            model_version=pred.metadata.get("version", "1.0.0"),
        )
    except ModelInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unhandled error during batch prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference execution failed: {exc}",
        ) from exc
