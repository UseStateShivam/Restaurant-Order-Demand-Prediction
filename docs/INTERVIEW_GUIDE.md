# Technical Interview Guide: Restaurant Order Demand Prediction

This guide prepares engineers to discuss the architecture, design trade-offs, and machine learning nuances of this project in technical interviews.

## 1. Project Pitches

### 30-Second Elevator Pitch
"I built a production-grade machine learning system for predicting weekly meal demand across a network of 77 restaurant fulfillment centers. The pipeline processes over 450,000 transactions, engineers price elasticity, promotional interaction, and Fourier seasonal signals, and serves predictions via a FastAPI microservice and Streamlit UI. Crucially, I implemented a strict chronological holdout across 145 weeks to prevent temporal data leakage and benchmarked our gradient-boosted tree regressor against a domain-specific center-meal heuristic baseline."

### Two-Minute Architectural Deep Dive
"In restaurant supply chains, inaccurate demand forecasts lead directly to either severe ingredient stockouts or massive perishable food spoilage. 

To solve this, I designed an end-to-end Python system using pandas, scikit-learn, and FastAPI. The data consists of multi-entity panel data: 77 fulfillment centers and 51 unique meals over 145 weeks.

First, to eliminate temporal leakage, standard random k-fold cross-validation was rejected. Instead, I established a chronological partition: Weeks 1 to 125 for training (390k rows), Weeks 126 to 135 for validation, and Weeks 136 to 145 as a strictly untouched test holdout.

Second, for feature engineering, I derived dynamic economic indicators: absolute discount, discount percentage ratio, compound promotional interaction flags, and cyclical Fourier sine/cosine transformations on 52-week annual cycles. All transformations are encapsulated within a scikit-learn `ColumnTransformer` fitted exclusively on training data.

Third, for modeling, I established a realistic domain baseline: a `BaselineDemandModel` grouping historical medians by center and meal with a global fallback. For the candidate model, I selected `HistGradientBoostingRegressor` because its histogram-based binning trained on ~390k rows in just 14 seconds with a 1.8 MB artifact size, outperforming traditional Random Forest in training efficiency and latency.

Finally, for deployment and reproducibility, the entire pipeline is persisted atomically using Joblib and paired with JSON provenance metadata. The system is served via a FastAPI REST API with Pydantic validation and complete test coverage with pytest."

## 2. Key Technical Decisions and Justifications

### Q1: Why not use standard K-Fold Cross Validation?
**Answer**: Random k-fold cross validation randomly shuffles rows across the entire dataset. In time-series and demand forecasting, future demand observations (e.g. Week 140) would be present in the training set while predicting past observations (e.g. Week 40). This causes severe lookahead leakage, falsely inflating validation metrics. A chronological train-validation-test split accurately mirrors the real-world operational reality where models must forecast future unseen demand.

### Q2: Why choose HistGradientBoostingRegressor over XGBoost/LightGBM or Deep Learning?
**Answer**:
1. **Lightweight Dependency Footprint**: `HistGradientBoostingRegressor` is native to scikit-learn (inspired by LightGBM), eliminating external C++ compilation or heavy third-party wheel dependencies like CUDA/Torch.
2. **Speed**: It bins continuous features into 256 integer bins, allowing it to fit 390,000 rows across 19 features in ~14 seconds.
3. **Missing Value & Categorical Robustness**: It handles missing values and dense feature arrays natively without memory explosion.

### Q3: Why is WAPE preferred over MAPE for demand forecasting?
**Answer**: MAPE (Mean Absolute Percentage Error) computes `|y - y_hat| / y`. When actual demand `y` is very low (e.g., 10 or 15 orders, common in low-volume centers), small absolute errors produce astronomical percentage errors (e.g., error of 15 on actual 10 is 150%). Furthermore, if actual demand is zero, MAPE is undefined. WAPE divides the sum of absolute errors by the sum of actual demand: `sum(|y - y_hat|) / sum(y)`. This weights each error proportional to its volume, making it statistically stable and intuitive for retail supply chain executives.

### Q4: How is data leakage prevented between training and inference?
**Answer**:
1. Preprocessing transformations (scaling, median imputation, one-hot encoding) are encapsulated within a scikit-learn `ColumnTransformer` that is fitted *only* on training observations (`week <= 125`).
2. The fitted transformer is persisted as part of a single end-to-end `Pipeline` object.
3. At inference time, new data only calls `.transform()` or `.predict()`, never `.fit()`.
4. Categorical encoders use `handle_unknown='ignore'` so unseen categories at test time do not cause errors or leak state.

### Q5: How do you handle potential negative predictions for count data?
**Answer**: Standard regression models minimize squared error or absolute loss and can mathematically output small negative numbers for low-volume items with unfavorable promotional pricing. In this system, we enforce an explicit domain boundary post-processing rule: `predictions = np.maximum(1.0, predictions)`. This reflects the physical operational reality that order counts cannot be negative and central kitchens fulfill minimum discrete batch thresholds.

## 3. Potential Follow-Up Questions and Answers

| Interviewer Question | Recommended Technical Response |
|---|---|
| **How would you handle cold-start meals?** | New menu items lack historical demand. We leverage meal catalog embeddings/metadata (`category`, `cuisine`, and `base_price`) combined with center attributes to generalize from similar existing dishes until historical sales accumulate. |
| **How does the system scale to millions of rows?** | The feature preprocessing and histogram gradient boosting scale linearly. In production, feature engineering can be executed via DuckDB, Polars, or PySpark before distributed model training or batch inference. |
| **What would you monitor in production?** | 1. Input schema violations and null rates; 2. Feature distribution drift using the implemented Kolmogorov-Smirnov test and PSI; 3. Prediction latency and 99th percentile response times; 4. Delayed ground truth reconciliation: tracking rolling WAPE once actual weekly order receipts arrive. |
