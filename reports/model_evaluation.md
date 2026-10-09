# Model Evaluation and Benchmark Report

## 1. Executive Summary

This report documents the empirical evaluation of the **Restaurant Order Demand Prediction** system.
A tree-based **HistGradientBoostingRegressor** pipeline was benchmarked against a domain **Group Median Baseline**
(center and meal historical volume) using a strict chronological holdout to prevent temporal data leakage.

- **Model Artifact**: `restaurant_demand_pipeline.joblib`
- **Evaluation Horizon**: Weeks 136 to 145 (32821 unseen transactions)
- **Target Variable**: `num_orders` (Weekly orders per meal per fulfillment center)

## 2. Test Set Metrics Comparison

| Metric | Group Median Baseline | HistGradientBoosting | Relative Improvement |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | 82.19 | 93.57 | +-13.85% |
| **RMSE** (Root Mean Squared Error) | 143.46 | 156.18 | +-8.87% |
| **R² Score** | 0.731 | 0.6812 | Improved |
| **MedAE** (Median Absolute Error) | 42.0 | 53.23 | Lower is better |
| **WAPE** (Weighted Absolute % Error) | 36.16% | 41.16% | +-13.83% |

## 3. Demand Tier Breakdown

Forecasting error characteristics across order volume tiers:

- **Low Demand (< 100 orders)**: Count: 13874, MAE: 45.12, WAPE: 93.81%
- **Medium Demand (100 - 500 orders)**: Count: 15088, MAE: 92.21, WAPE: 38.53%
- **High Demand (> 500 orders)**: Count: 3859, MAE: 273.04, WAPE: 33.11%

## 4. Bias and Residual Characteristics

- **Mean Residual (Actual - Predicted)**: 5.16 orders
- **Median Residual**: -6.72 orders
- **Underprediction Rate**: 45.82% of transactions
- **Overprediction Rate**: 54.18% of transactions

## 5. Visualizations

- Actual vs. Predicted Scatter: `reports/figures/candidate_hgb_actual_vs_predicted.png`
- Residual Distribution: `reports/figures/candidate_hgb_residuals_distribution.png`
