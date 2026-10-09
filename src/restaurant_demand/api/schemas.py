"""Pydantic schemas for FastAPI request and response validation.

Enforces strict input boundary validation on incoming prediction requests,
rejecting malformed parameters before model execution.
"""

from typing import Any

from pydantic import BaseModel, Field


class OrderDemandPredictionRequest(BaseModel):
    """Schema for a single restaurant order demand prediction request."""

    week: int = Field(..., ge=1, description="Planning week number (e.g., 140)")
    center_id: int = Field(..., ge=1, description="Fulfillment center identifier")
    meal_id: int = Field(..., ge=1, description="Meal catalog identifier")
    checkout_price: float = Field(..., gt=0.0, description="Discounted transaction checkout price")
    base_price: float = Field(..., gt=0.0, description="Base catalog reference price")
    emailer_for_promotion: int = Field(
        ..., ge=0, le=1, description="Email marketing campaign active (0 or 1)"
    )
    homepage_featured: int = Field(
        ..., ge=0, le=1, description="Featured on website homepage (0 or 1)"
    )

    # Optional metadata overrides (if omitted, looked up automatically from catalog tables)
    city_code: int | None = Field(None, description="City code of the center")
    region_code: int | None = Field(None, description="Region code of the center")
    center_type: str | None = Field(
        None, description="Center layout type ('TYPE_A', 'TYPE_B', 'TYPE_C')"
    )
    op_area: float | None = Field(None, gt=0.0, description="Operational area of center in sq km")
    category: str | None = Field(None, description="Food category (e.g., 'Beverages', 'Rice Bowl')")
    cuisine: str | None = Field(None, description="Cuisine type (e.g., 'Indian', 'Italian')")

    model_config = {
        "json_schema_extra": {
            "example": {
                "week": 140,
                "center_id": 55,
                "meal_id": 1885,
                "checkout_price": 136.83,
                "base_price": 152.29,
                "emailer_for_promotion": 0,
                "homepage_featured": 0,
                "center_type": "TYPE_C",
                "op_area": 2.0,
                "category": "Beverages",
                "cuisine": "Thai",
            }
        }
    }


class OrderDemandPredictionResponse(BaseModel):
    """Schema for single prediction output response."""

    center_id: int
    meal_id: int
    week: int
    predicted_demand: float = Field(..., description="Forecasted order volume")
    target_units: str = "orders_per_week"
    model_version: str
    model_name: str


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    orders: list[OrderDemandPredictionRequest] = Field(
        ..., min_length=1, description="List of meal-center-week transaction records"
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: list[OrderDemandPredictionResponse]
    total_records: int
    model_version: str


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    """Model metadata and governance specification response."""

    model_name: str
    version: str
    target_variable: str
    target_units: str
    test_metrics: dict[str, float]
    created_at_utc: str
    expected_input_schema: dict[str, Any]
