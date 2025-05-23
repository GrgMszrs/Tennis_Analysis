#!/usr/bin/env python3
"""
Date Visualization Script
Create comprehensive visualizations of match date distribution and temporal patterns.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import setup_logging


def load_standardized_data():
    """Load standardized match data."""
    logger = setup_logging()
    logger.info("Loading standardized match data for visualization")

    data_file = Path("data/cleaned_refactored/atp_matches_standardized.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Standardized data not found: {data_file}")

    df = pd.read_csv(data_file)
    logger.info(f"Loaded: {len(df):,} matches")

    return df


def prepare_data(df):
    """Prepare data for visualization."""
    # Convert dates
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # Remove invalid dates
    df = df.dropna(subset=["tourney_date"])

    # Create time periods
    df["year"] = df["tourney_date"].dt.year
    df["month"] = df["tourney_date"].dt.month
    df["month_period"] = df["tourney_date"].dt.to_period("M")

    print(
        f"Data prepared: {len(df):,} matches from {df['tourney_date'].min().strftime('%Y-%m-%d')} to {df['tourney_date'].max().strftime('%Y-%m-%d')}"
    )

    return df


def create_date_visualizations(df):
    """Create comprehensive date visualizations."""
    print("Creating date visualizations...")

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Tennis Match Date Analysis (2005-2024)", fontsize=16, fontweight="bold")

    # 1. Annual match counts (top left)
    ax1 = axes[0, 0]
    yearly_counts = df.groupby("year").size()
    yearly_counts.plot(kind="bar", ax=ax1, color="steelblue", alpha=0.8)
    ax1.set_title("Annual Match Counts", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Matches")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Monthly timeline (top right)
    ax2 = axes[0, 1]
    monthly_counts = df.groupby("month_period").size()
    monthly_counts.plot(kind="line", ax=ax2, color="darkgreen", linewidth=1.5)
    ax2.set_title("Monthly Match Distribution Over Time", fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of Matches")
    ax2.grid(True, alpha=0.3)

    # 3. Seasonal patterns (bottom left)
    ax3 = axes[1, 0]
    monthly_avg = df.groupby("month").size()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_avg.plot(kind="bar", ax=ax3, color="orange", alpha=0.8)
    ax3.set_title("Seasonal Distribution (Total Matches by Month)", fontweight="bold")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Total Matches")
    ax3.set_xticklabels(month_names, rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Surface distribution over time (bottom right)
    ax4 = axes[1, 1]
    surface_yearly = df.groupby(["year", "surface"]).size().unstack(fill_value=0)
    surface_yearly.plot(kind="area", ax=ax4, alpha=0.7)
    ax4.set_title("Surface Distribution Over Time", fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Number of Matches")
    ax4.legend(title="Surface", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "tennis_date_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to: {plot_file}")

    plt.show()

    return yearly_counts, monthly_avg


def analyze_temporal_patterns(df, yearly_counts, monthly_avg):
    """Analyze and report temporal patterns."""
    print("\n" + "=" * 60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 60)

    # Overall statistics
    print("\n📊 Overall Statistics:")
    print(f"  Total matches: {len(df):,}")
    print(f"  Date range: {df['tourney_date'].min().strftime('%Y-%m-%d')} to {df['tourney_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Years covered: {df['year'].nunique()} years ({df['year'].min()}-{df['year'].max()})")
    print(f"  Average matches per year: {len(df)/df['year'].nunique():.0f}")
    print(f"  Average matches per month: {len(df)/(df['year'].nunique()*12):.0f}")

    # Annual patterns
    print("\n📅 Annual Patterns:")
    print(f"  Busiest year: {yearly_counts.idxmax()} ({yearly_counts.max():,} matches)")
    print(f"  Quietest year: {yearly_counts.idxmin()} ({yearly_counts.min():,} matches)")
    print(f"  Annual growth: {((yearly_counts.iloc[-1] / yearly_counts.iloc[0]) - 1) * 100:.1f}% over period")

    # Seasonal patterns
    print("\n🌱 Seasonal Patterns:")
    print(f"  Busiest month: {monthly_avg.idxmax()} ({monthly_avg.max():,} total matches)")
    print(f"  Quietest month: {monthly_avg.idxmin()} ({monthly_avg.min():,} total matches)")
    print(f"  Seasonal variation: {((monthly_avg.max() / monthly_avg.min()) - 1) * 100:.0f}% difference")

    # Surface patterns
    surface_counts = df["surface"].value_counts()
    print("\n🏟️ Surface Distribution:")
    for surface, count in surface_counts.items():
        pct = count / len(df) * 100
        print(f"  {surface}: {count:,} matches ({pct:.1f}%)")

    # Data coverage analysis
    print("\n🔍 Data Coverage:")
    all_months = pd.period_range(df["month_period"].min(), df["month_period"].max(), freq="M")
    actual_months = set(df["month_period"])
    missing_months = set(all_months) - actual_months

    coverage_pct = len(actual_months) / len(all_months) * 100
    print(f"  Expected months in range: {len(all_months)}")
    print(f"  Months with data: {len(actual_months)}")
    print(f"  Coverage: {coverage_pct:.1f}%")

    if missing_months:
        missing_list = sorted([str(m) for m in missing_months])
        if len(missing_list) <= 15:
            print(f"  Missing months: {missing_list}")
        else:
            print(f"  Missing months: {missing_list[:10]} ... and {len(missing_list)-10} more")


def create_date_analysis():
    """Main function to create date analysis and visualizations."""
    print("🎾 TENNIS MATCH DATE VISUALIZATION")
    print("=" * 50)

    try:
        # Load and prepare data
        df = load_standardized_data()
        df = prepare_data(df)

        # Create visualizations
        yearly_counts, monthly_avg = create_date_visualizations(df)

        # Analyze patterns
        analyze_temporal_patterns(df, yearly_counts, monthly_avg)

        print("\n✅ Date analysis completed successfully!")
        print("📊 Visualization saved to data/output/tennis_date_analysis.png")

    except Exception as e:
        print(f"❌ Error in date analysis: {e}")
        raise


if __name__ == "__main__":
    create_date_analysis()
