"""Pytest fixtures and test configuration for Restaurant Order Demand Prediction.

Provides deterministic synthetic fixtures for units and integration tests,
avoiding network calls and isolating temporary filesystem artifacts.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_train_df() -> pd.DataFrame:
    """Creates a deterministic synthetic transaction DataFrame matching train.csv."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "id": np.arange(1000, 1000 + n),
            "week": rng.choice(np.arange(1, 146), size=n),
            "center_id": rng.choice([10, 11, 13, 55], size=n),
            "meal_id": rng.choice([1885, 1993, 2539], size=n),
            "checkout_price": rng.uniform(100.0, 300.0, size=n).round(2),
            "base_price": rng.uniform(150.0, 350.0, size=n).round(2),
            "emailer_for_promotion": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "homepage_featured": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "num_orders": rng.integers(20, 800, size=n),
        }
    )


@pytest.fixture
def sample_center_df() -> pd.DataFrame:
    """Creates synthetic fulfillment center metadata matching fulfilment_center_info.csv."""
    return pd.DataFrame(
        {
            "center_id": [10, 11, 13, 55],
            "city_code": [647, 679, 590, 647],
            "region_code": [56, 56, 56, 77],
            "center_type": ["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_A"],
            "op_area": [3.8, 4.2, 2.5, 6.7],
        }
    )


@pytest.fixture
def sample_meal_df() -> pd.DataFrame:
    """Creates synthetic meal catalog metadata matching meal_info.csv."""
    return pd.DataFrame(
        {
            "meal_id": [1885, 1993, 2539],
            "category": ["Beverages", "Beverages", "Rice Bowl"],
            "cuisine": ["Thai", "Thai", "Indian"],
        }
    )


@pytest.fixture
def sample_merged_df(sample_raw_train_df, sample_center_df, sample_meal_df) -> pd.DataFrame:
    """Returns merged DataFrame combining demand transactions and entity metadata."""
    merged = sample_raw_train_df.merge(sample_center_df, on="center_id", how="left")
    merged = merged.merge(sample_meal_df, on="meal_id", how="left")
    return merged


@pytest.fixture
def sample_inference_record() -> dict:
    """Valid single inference payload dictionary."""
    return {
        "week": 140,
        "center_id": 55,
        "meal_id": 1885,
        "checkout_price": 136.83,
        "base_price": 152.29,
        "emailer_for_promotion": 0,
        "homepage_featured": 0,
        "center_type": "TYPE_A",
        "op_area": 3.8,
        "category": "Beverages",
        "cuisine": "Thai",
        "city_code": 647,
        "region_code": 77,
    }
