"""Streamlit Web Interface for Restaurant Order Demand Prediction.

Provides an interactive user interface for single transaction forecasting,
batch CSV scoring, and model performance inspection.
"""

import pandas as pd
import streamlit as st

from restaurant_demand.exceptions import ModelInferenceError, ModelNotFoundError
from restaurant_demand.logging_config import configure_logger
from restaurant_demand.models.predict import DemandPredictor

logger = configure_logger(__name__)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Restaurant Order Demand Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor():
    """Caches the predictor in memory to avoid reloading per interaction."""
    try:
        return DemandPredictor()
    except ModelNotFoundError:
        return None
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


def main():
    """Main Streamlit application layout and logic."""
    st.title("Restaurant Order Demand Prediction System")
    st.caption(
        "Forecasting weekly order demand across restaurant fulfillment centers using Machine Learning"
    )

    predictor = load_predictor()

    if predictor is None:
        st.warning(
            "Model artifact not found. Please train the model first by running:\n"
            "```bash\npython -m restaurant_demand.cli train\n```"
        )
        return

    # Sidebar: Model and System Governance Information
    st.sidebar.header("Model Specifications")
    meta = predictor.metadata
    st.sidebar.text(f"Model: {meta.get('model_name', 'HistGradientBoosting')}")
    st.sidebar.text(f"Version: {meta.get('version', '1.0.0')}")
    st.sidebar.text(
        f"Target: {meta.get('target_variable', 'num_orders')} ({meta.get('target_units', 'orders_per_week')})"
    )

    metrics = meta.get("test_metrics", {})
    if metrics:
        st.sidebar.subheader("Held-Out Test Performance")
        st.sidebar.metric(label="MAE (Orders)", value=metrics.get("mae", "N/A"))
        st.sidebar.metric(label="RMSE (Orders)", value=metrics.get("rmse", "N/A"))
        st.sidebar.metric(label="WAPE Error", value=f"{metrics.get('wape', 'N/A')}%")
        st.sidebar.metric(label="R-Squared", value=metrics.get("r2", "N/A"))

    tabs = st.tabs(["Single Order Forecast", "Batch CSV Scoring", "Model Card & Limitations"])

    # TAB 1: Single Prediction
    with tabs[0]:
        st.subheader("Forecast Individual Meal-Center Demand")
        st.markdown(
            "Enter the operational parameters for an upcoming week to predict expected demand."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            week = st.number_input(
                "Forecast Planning Week", min_value=1, max_value=250, value=146, step=1
            )
            center_id = st.number_input(
                "Fulfillment Center ID", min_value=1, max_value=200, value=55, step=1
            )
            meal_id = st.number_input("Meal ID", min_value=1, max_value=5000, value=1885, step=1)

        with col2:
            base_price = st.number_input(
                "Base Catalog Price (USD/INR)",
                min_value=10.0,
                max_value=2000.0,
                value=152.29,
                step=1.0,
            )
            checkout_price = st.number_input(
                "Checkout Price (after discount)",
                min_value=10.0,
                max_value=2000.0,
                value=136.83,
                step=1.0,
            )

        with col3:
            emailer = st.selectbox(
                "Email Promotion Campaign",
                options=[0, 1],
                format_func=lambda x: "Active (1)" if x == 1 else "Inactive (0)",
            )
            featured = st.selectbox(
                "Homepage Featured Placement",
                options=[0, 1],
                format_func=lambda x: "Active (1)" if x == 1 else "Inactive (0)",
            )

        # Optional metadata inputs
        with st.expander("Optional Metadata Overrides (Leave blank to use catalog defaults)"):
            c_type = st.selectbox(
                "Center Type", options=["Default from catalog", "TYPE_A", "TYPE_B", "TYPE_C"]
            )
            cat = st.selectbox(
                "Meal Category",
                options=[
                    "Default from catalog",
                    "Beverages",
                    "Rice Bowl",
                    "Sandwich",
                    "Starters",
                    "Desert",
                    "Salad",
                    "Pasta",
                    "Pizza",
                ],
            )
            cui = st.selectbox(
                "Cuisine",
                options=["Default from catalog", "Thai", "Indian", "Italian", "Continental"],
            )

        if st.button("Generate Demand Forecast", type="primary"):
            record = {
                "week": int(week),
                "center_id": int(center_id),
                "meal_id": int(meal_id),
                "checkout_price": float(checkout_price),
                "base_price": float(base_price),
                "emailer_for_promotion": int(emailer),
                "homepage_featured": int(featured),
            }
            if c_type != "Default from catalog":
                record["center_type"] = c_type
            if cat != "Default from catalog":
                record["category"] = cat
            if cui != "Default from catalog":
                record["cuisine"] = cui

            try:
                result = predictor.predict_single(record)
                pred_val = result["predicted_demand"]

                st.success("Demand prediction generated successfully.")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Predicted Demand", f"{pred_val:.1f} orders")
                res_col2.metric("Target Granularity", "Center x Meal x Week")
                discount = max(0.0, base_price - checkout_price)
                res_col3.metric(
                    "Effective Discount", f"${discount:.2f} ({discount / base_price * 100:.1f}%)"
                )

                st.info(
                    f"Operational Guidance: Maintain ingredient inventory and preparation capacity "
                    f"for approximately {int(round(pred_val))} units during Week {week} at Center {center_id}."
                )
            except ModelInferenceError as err:
                st.error(f"Prediction failed: {err}")

    # TAB 2: Batch Scoring
    with tabs[1]:
        st.subheader("Batch Order Demand Scoring")
        st.markdown(
            "Upload a CSV file containing transactions. Required columns: "
            "`week`, `center_id`, `meal_id`, `checkout_price`, `base_price`, "
            "`emailer_for_promotion`, `homepage_featured`."
        )

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(input_df)} rows. Preview:")
                st.dataframe(input_df.head(5))

                if st.button("Score Batch Dataset"):
                    with st.spinner("Scoring records..."):
                        scored_df = predictor.predict_dataframe(input_df)
                    st.success(f"Successfully scored {len(scored_df)} records.")
                    st.dataframe(
                        scored_df[
                            ["week", "center_id", "meal_id", "checkout_price", "predicted_demand"]
                        ].head(10)
                    )

                    csv_data = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Scored Predictions (CSV)",
                        data=csv_data,
                        file_name="scored_restaurant_demand.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Error processing batch file: {exc}")

    # TAB 3: Model Card & Limitations
    with tabs[2]:
        st.subheader("Operational Limitations and Model Governance")
        st.markdown(
            """
            ### Intended Use
            - Forecasts expected weekly meal demand for restaurant fulfillment centers.
            - Guides staffing, ingredient purchasing, and central kitchen prep schedules.

            ### Known Limitations
            1. **Macro Shock Invariance**: Does not account for sudden external disruptions (extreme weather, local transit strikes, pandemics).
            2. **Lead Time Assumption**: Assumes pricing and marketing decisions are established at least one week prior to fulfillment.
            3. **Non-Negative Bounding**: Order volume is lower-bounded at 1.0 order, matching historical retail ordering minimums.
            4. **Granularity**: Operates strictly on a weekly aggregation level. Intraday hour-by-hour spikes require sub-daily replenishment models.

            ### Data Provenance
            - Trained on benchmark transactions across 77 fulfillment centers and 51 meal items.
            - Evaluated strictly using chronological holdout partitions (Weeks 136-145).
            """
        )


if __name__ == "__main__":
    main()
