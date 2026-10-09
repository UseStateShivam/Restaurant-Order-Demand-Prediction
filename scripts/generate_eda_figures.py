"""Script to generate publication-grade EDA visualizations.

Saves figures to reports/figures/:
- demand_target_distribution.png
- demand_by_category_and_cuisine.png
- promotional_impact_comparison.png
- weekly_demand_trend.png
- price_vs_demand_elasticity.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from restaurant_demand.config import get_project_root


def main():
    root = get_project_root()
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    interim_path = root / "data" / "interim" / "merged_data.csv"
    if not interim_path.exists():
        print("Interim merged data not found; run ingestion first.")
        return

    print("Loading interim merged data for EDA figures...")
    df = pd.read_csv(interim_path)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Target Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["num_orders"], bins=60, ax=ax1, color="#1f77b4", kde=False)
    ax1.set_title("Target 'num_orders' Distribution (Raw Scale)")
    ax1.set_xlabel("Weekly Order Count")
    ax1.set_ylabel("Frequency")

    sns.histplot(np.log1p(df["num_orders"]), bins=60, ax=ax2, color="#2ca02c", kde=True)
    ax2.set_title("Target Distribution (Log1p Transformed)")
    ax2.set_xlabel("log(1 + num_orders)")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(figures_dir / "demand_target_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Demand by Category and Cuisine
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cat_order = df.groupby("category")["num_orders"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="category", y="num_orders", order=cat_order, ax=ax1, palette="Blues_r")
    ax1.set_title("Mean Weekly Orders by Meal Category")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Mean Orders")
    ax1.tick_params(axis="x", rotation=45)

    cui_order = df.groupby("cuisine")["num_orders"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="cuisine", y="num_orders", order=cui_order, ax=ax2, palette="Greens_r")
    ax2.set_title("Mean Weekly Orders by Cuisine")
    ax2.set_xlabel("Cuisine")
    ax2.set_ylabel("Mean Orders")
    plt.tight_layout()
    fig.savefig(figures_dir / "demand_by_category_and_cuisine.png", dpi=200)
    plt.close(fig)

    # 3. Promotional Impact
    fig, ax = plt.subplots(figsize=(9, 6))
    promo_groups = df.groupby(["emailer_for_promotion", "homepage_featured"])["num_orders"].mean().reset_index()
    promo_groups["promo_type"] = promo_groups.apply(
        lambda r: "Both (Email + Homepage)" if r["emailer_for_promotion"] and r["homepage_featured"]
        else "Email Only" if r["emailer_for_promotion"]
        else "Homepage Only" if r["homepage_featured"]
        else "No Promotion",
        axis=1
    )
    sns.barplot(data=promo_groups, x="promo_type", y="num_orders", palette="mako", ax=ax)
    ax.set_title("Promotional Strategy Impact on Mean Weekly Demand")
    ax.set_xlabel("Promotional Condition")
    ax.set_ylabel("Mean Weekly Orders")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_dir / "promotional_impact_comparison.png", dpi=200)
    plt.close(fig)

    # 4. Weekly Demand Trend
    fig, ax = plt.subplots(figsize=(14, 5))
    weekly_agg = df.groupby("week")["num_orders"].sum().reset_index()
    ax.plot(weekly_agg["week"], weekly_agg["num_orders"], color="#d62728", lw=2)
    ax.axvline(125, color="blue", linestyle="--", label="Train / Val Split (Week 125)")
    ax.axvline(135, color="green", linestyle="--", label="Val / Test Split (Week 135)")
    ax.set_title("Total System-Wide Weekly Order Demand Across 145 Weeks")
    ax.set_xlabel("Chronological Week Number")
    ax.set_ylabel("Total Orders")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "weekly_demand_trend.png", dpi=200)
    plt.close(fig)

    # 5. Price Elasticity
    fig, ax = plt.subplots(figsize=(9, 6))
    sample_elasticity = df.sample(n=min(5000, len(df)), random_state=42)
    sns.scatterplot(data=sample_elasticity, x="checkout_price", y="num_orders", hue="has_any_promotion" if "has_any_promotion" in df.columns else None, alpha=0.4, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Checkout Price vs Demand (Log-Scale Order Count)")
    ax.set_xlabel("Checkout Price (Monetary Units)")
    ax.set_ylabel("log(num_orders)")
    plt.tight_layout()
    fig.savefig(figures_dir / "price_vs_demand_elasticity.png", dpi=200)
    plt.close(fig)

    print("Saved all 5 EDA figures to reports/figures/ successfully.")


if __name__ == "__main__":
    main()
