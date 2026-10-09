# Data Dictionary

This document details the schema, definitions, measurement units, roles, and prediction-time availability for all columns within the Restaurant Order Demand Prediction dataset.

## 1. Raw Transactional Fields (`train.csv`)

| Field Name | Storage Type | Domain Type | Meaning / Description | Role | Units | Availability at Prediction Time | Missing Handling |
|---|---|---|---|---|---|---|---|
| `id` | Integer | Identifier | Unique transaction record identifier | Identifier | None | Excluded from features | Unique primary key; duplicates rejected |
| `week` | Integer | Temporal | Chronological week number since tracking began (weeks 1 to 145) | Feature / Split Key | Integer weeks | Known (forecast planning week) | Required; must be >= 1 |
| `center_id` | Integer | Identifier | Fulfillment center facility identifier (77 unique facilities) | Entity Feature | Integer ID | Known | Required; matched against center catalog |
| `meal_id` | Integer | Identifier | Menu item identifier (51 unique meal options) | Entity Feature | Integer ID | Known | Required; matched against meal catalog |
| `checkout_price` | Float | Continuous | Final transaction price charged to customer after promotions | Numeric Feature | Monetary currency | Known (established menu price) | Required; must be > 0.0 |
| `base_price` | Float | Continuous | Standard baseline catalog price before promotional discounting | Numeric Feature | Monetary currency | Known (catalog reference price) | Required; must be > 0.0 |
| `emailer_for_promotion`| Integer | Binary | Whether an email newsletter campaign featured this meal item | Categorical / Binary | Flag (0 or 1) | Known (marketing campaign schedule) | Required; validated in {0, 1} |
| `homepage_featured` | Integer | Binary | Whether the meal was prominently displayed on the ordering homepage | Categorical / Binary | Flag (0 or 1) | Known (placement schedule) | Required; validated in {0, 1} |
| `num_orders` | Integer | Discrete | Total number of meal orders fulfilled during the specified week | **Target Variable** | Count of orders | **Unavailable (Target)** | Must be non-null and > 0 during training |

## 2. Fulfillment Center Metadata (`fulfilment_center_info.csv`)

| Field Name | Storage Type | Meaning / Description | Role | Units | Availability at Prediction Time |
|---|---|---|---|---|---|
| `center_id` | Integer | Primary key matching transaction records | Join Key | None | Known |
| `city_code` | Integer | Anonymized city location code | Categorical Feature | Integer Code | Known (static facility attribute) |
| `region_code` | Integer | Anonymized regional administrative territory | Categorical Feature | Integer Code | Known (static facility attribute) |
| `center_type` | String | Facility operational classification (`TYPE_A`, `TYPE_B`, `TYPE_C`) | Categorical Feature | Category Name | Known (static facility attribute) |
| `op_area` | Float | Catchment operations surface area serviced by kitchen | Numeric Feature | Square Kilometers | Known (static facility attribute) |

## 3. Meal Catalog Metadata (`meal_info.csv`)

| Field Name | Storage Type | Meaning / Description | Role | Units | Availability at Prediction Time |
|---|---|---|---|---|---|
| `meal_id` | Integer | Primary key matching transaction records | Join Key | None | Known |
| `category` | String | Meal food classification (Beverages, Rice Bowl, Sandwich, etc.) | Categorical Feature | Category Name | Known (static menu item attribute) |
| `cuisine` | String | Culinary style (Continental, Indian, Italian, Thai) | Categorical Feature | Cuisine Name | Known (static menu item attribute) |

## 4. Derived Engineered Features

| Field Name | Formula / Derivation | Meaning | Units | Role |
|---|---|---|---|---|
| `discount_amount` | `max(0.0, base_price - checkout_price)` | Absolute promotional discount | Currency | Numeric Feature |
| `discount_ratio` | `clip(discount_amount / base_price, 0.0, 1.0)` | Relative discount percentage | Ratio [0, 1] | Numeric Feature |
| `price_ratio` | `clip(checkout_price / base_price, 0.0, 2.0)` | Ratio of final price to base price | Ratio | Numeric Feature |
| `has_any_promotion` | `(emailer_for_promotion == 1) \| (homepage_featured == 1)` | Any marketing campaign active | Binary [0, 1] | Binary Feature |
| `dual_promotion` | `(emailer_for_promotion == 1) & (homepage_featured == 1)` | Compound promotion active | Binary [0, 1] | Binary Feature |
| `week_of_year` | `((week - 1) % 52) + 1` | Annual calendar week proxy | Integer [1, 52] | Temporal Feature |
| `week_sin` | `sin(2 * pi * week_of_year / 52.0)` | Cyclical harmonic seasonal feature | Float [-1, 1] | Numeric Feature |
| `week_cos` | `cos(2 * pi * week_of_year / 52.0)` | Cyclical harmonic seasonal feature | Float [-1, 1] | Numeric Feature |
| `quarter` | `((week_of_year - 1) // 13) + 1` | Quarter of the year | Integer [1, 4] | Categorical Feature |
