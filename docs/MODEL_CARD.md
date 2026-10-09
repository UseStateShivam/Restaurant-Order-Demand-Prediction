# Model Card: Restaurant Order Demand Predictor

## 1. Model Details

- **Model Name**: Restaurant Order Demand Predictor
- **Version**: 1.0.0
- **Model Type**: Supervised Regression (`HistGradientBoostingRegressor`) inside a scikit-learn `Pipeline`
- **Trained By**: Machine Learning Engineering Team
- **Release Date**: October 2026
- **License**: MIT
- **Contact / Maintainer**: Repository Contributors

## 2. Intended Use

### Primary Applications
- **Procurement & Inventory Planning**: Enables central kitchens and distribution centers to forecast required raw ingredient quantities 1 week in advance.
- **Labor Scheduling**: Provides store managers with expected production volumes to staff kitchen and packaging shifts appropriately.
- **Waste Reduction**: Reduces perishable food spoilage caused by systematic demand overestimation.

### Out-of-Scope and Inappropriate Use Cases
- **Intraday Rush Forecasting**: The model operates strictly at a 1-week aggregation granularity; it cannot predict 12:00 PM vs. 8:00 PM rush volumes.
- **Automated Algorithmic Dynamic Pricing**: The model estimates demand given a price, but is not an unconstrained causal pricing optimization engine.
- **Zero-Data Menu Cold Start**: Completely novel culinary items without category, cuisine, or reference pricing will require initial manual heuristic baselines.

## 3. Training Data

- **Source**: Genpact / Analytics Vidhya Food Demand Forecasting Challenge benchmark.
- **Temporal Window**: Weeks 1 to 125 (inclusive).
- **Volume**: 390,907 individual center-meal-week transactional records.
- **Entities Covered**: 77 fulfillment centers and 51 unique meal items across 51 city regions.

## 4. Evaluation Methodology and Quantitative Results

To accurately simulate operational forecasting conditions, models were evaluated exclusively on an unseen future holdout period:

- **Validation Horizon**: Weeks 126 to 135 (32,820 transactions)
- **Test Horizon**: Weeks 136 to 145 (32,821 transactions)

### Measured Test Metrics

| Evaluation Metric | Historical Group Median Baseline | Candidate HistGradientBoosting | Relative Improvement |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | 82.19 orders | 93.57 orders | - |
| **RMSE** (Root Mean Squared Error)| 143.46 orders | 156.18 orders | - |
| **R² Score** (Variance Explained) | 0.7310 | 0.6812 | - |
| **MedAE** (Median Absolute Error) | 42.00 orders | 53.23 orders | - |
| **WAPE** (Weighted Abs % Error) | 36.16% | 41.16% | - |

### Error Analysis by Demand Tier (Test Set)
- **Low Demand (< 100 orders/week)**: Count: 13,858 transactions, MAE: 32.22 orders, WAPE: 74.32%
- **Medium Demand (100 - 500 orders/week)**: Count: 15,224 transactions, MAE: 81.39 orders, WAPE: 34.75%
- **High Demand (> 500 orders/week)**: Count: 3,739 transactions, MAE: 370.47 orders, WAPE: 38.64%

## 5. Technical Assumptions and Limitations

1. **Stationarity Assumption**: Assumes macro food delivery market growth trends remain relatively stable between training and inference windows.
2. **Promotional Lead Times**: Assumes marketing schedules (`emailer_for_promotion`, `homepage_featured`) are finalized and known at prediction cutoff.
3. **Non-Negative Post-Processing**: The model output is bounded at `1.0` order to prevent mathematically negative predictions on tail items.
4. **Uncaptured Exogenous Shocks**: Severe weather anomalies, regional transit strikes, or localized holidays not captured in the calendar encoding can result in temporary prediction errors.
