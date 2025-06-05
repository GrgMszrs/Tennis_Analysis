#!/usr/bin/env python3
"""
Optimize Fuzzy Date Matching
Tests different date windows to maximize PBP-ATP matching.
"""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_fuzzy_date_windows(pbp_data: pd.DataFrame, atp_data: pd.DataFrame, max_days: int = 30):
    """Test different fuzzy date windows to find optimal matching rate."""
    logging.info("🔧 TESTING FUZZY DATE WINDOWS (1 to %d days)", max_days)
    logging.info("=" * 70)

    # Normalize data
    atp_data["winner_norm"] = atp_data["winner_name"].str.lower().str.strip()
    atp_data["loser_norm"] = atp_data["loser_name"].str.lower().str.strip()
    atp_data["date_parsed"] = pd.to_datetime(atp_data["tourney_date"], errors="coerce")

    pbp_data["player1_norm"] = pbp_data["server1"].str.lower().str.strip()
    pbp_data["player2_norm"] = pbp_data["server2"].str.lower().str.strip()
    pbp_data["date_parsed"] = pd.to_datetime(pbp_data["date_standardized"], errors="coerce")

    # Build ATP lookup by player combination
    atp_combo_dates = defaultdict(set)
    for _, row in atp_data.iterrows():
        if row["winner_norm"] and row["loser_norm"] and pd.notna(row["date_parsed"]):
            combo = frozenset([row["winner_norm"], row["loser_norm"]])
            atp_combo_dates[combo].add(row["date_parsed"].date())

    results = []

    # Test different date windows
    date_windows = [1, 2, 3, 5, 7, 10, 14, 21, 30]

    for window_days in date_windows:
        exact_matches = 0
        fuzzy_matches = 0
        no_matches = 0

        for _, row in pbp_data.iterrows():
            p1 = row["player1_norm"]
            p2 = row["player2_norm"]
            pbp_date = row["date_parsed"]

            if p1 and p2 and pd.notna(pbp_date):
                combo = frozenset([p1, p2])
                if combo in atp_combo_dates:
                    atp_dates = atp_combo_dates[combo]
                    pbp_date_only = pbp_date.date()

                    # Check exact match
                    if pbp_date_only in atp_dates:
                        exact_matches += 1
                    else:
                        # Check fuzzy match within window
                        found_fuzzy = False
                        for atp_date in atp_dates:
                            if abs((pbp_date_only - atp_date).days) <= window_days:
                                fuzzy_matches += 1
                                found_fuzzy = True
                                break
                        if not found_fuzzy:
                            no_matches += 1
                else:
                    no_matches += 1
            else:
                no_matches += 1

        total = len(pbp_data)
        total_matches = exact_matches + fuzzy_matches
        match_rate = total_matches / total * 100

        results.append(
            {
                "window_days": window_days,
                "exact_matches": exact_matches,
                "fuzzy_matches": fuzzy_matches,
                "total_matches": total_matches,
                "no_matches": no_matches,
                "match_rate": match_rate,
            }
        )

        logging.info(
            "   %2d days: %,d matches (%0.1f%%) - exact: %,d, fuzzy: %,d",
            window_days,
            total_matches,
            match_rate,
            exact_matches,
            fuzzy_matches,
        )

    return results


def plot_fuzzy_optimization_results(results):
    """Plot fuzzy date window optimization results."""
    logging.info("\n📈 CREATING OPTIMIZATION PLOT")
    logging.info("=" * 50)

    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Match rate vs window size
    ax1.plot(df["window_days"], df["match_rate"], "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Fuzzy Date Window (days)")
    ax1.set_ylabel("Match Rate (%)")
    ax1.set_title("PBP-ATP Match Rate vs Date Window Size")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add annotations for key points
    for _, row in df.iterrows():
        if row["window_days"] in [3, 7, 14]:
            ax1.annotate(
                f"{row['match_rate']:.1f}%",
                (row["window_days"], row["match_rate"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    # Plot 2: Diminishing returns analysis
    df["marginal_gain"] = df["match_rate"].diff().fillna(0)
    ax2.bar(df["window_days"], df["marginal_gain"], alpha=0.7, color="green")
    ax2.set_xlabel("Fuzzy Date Window (days)")
    ax2.set_ylabel("Marginal Gain (%)")
    ax2.set_title("Marginal Gain per Additional Day")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path("data/output/plots/fuzzy_date_optimization.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info("   Plot saved to: %s", output_path)

    plt.show()


def recommend_optimal_window(results):
    """Recommend optimal fuzzy date window based on diminishing returns."""
    logging.info("\n💡 OPTIMIZATION RECOMMENDATIONS")
    logging.info("=" * 50)

    df = pd.DataFrame(results)
    df["marginal_gain"] = df["match_rate"].diff().fillna(0)
    df["efficiency"] = df["marginal_gain"] / df["window_days"]

    # Find inflection points
    max_efficiency_idx = df["efficiency"].idxmax()
    optimal_window = df.iloc[max_efficiency_idx]["window_days"]

    logging.info("📊 Performance Analysis:")
    logging.info("   Current 3-day window:  %0.1f%% match rate", df[df["window_days"] == 3]["match_rate"].iloc[0])
    logging.info("   7-day window:          %0.1f%% match rate", df[df["window_days"] == 7]["match_rate"].iloc[0])
    logging.info("   14-day window:         %0.1f%% match rate", df[df["window_days"] == 14]["match_rate"].iloc[0])

    logging.info("\n🎯 Recommended optimal window: %d days", optimal_window)
    logging.info("   Match rate: %0.1f%%", df.iloc[max_efficiency_idx]["match_rate"])
    logging.info("   Total matches: %,d", df.iloc[max_efficiency_idx]["total_matches"])

    # Calculate potential improvement
    current_3day = df[df["window_days"] == 3]["match_rate"].iloc[0]
    optimal_rate = df.iloc[max_efficiency_idx]["match_rate"]
    improvement = optimal_rate - current_3day

    logging.info("\n📈 Improvement vs current 3-day window:")
    logging.info("   Additional match rate: +%0.1f percentage points", improvement)
    logging.info("   Additional matches: +%0.0f records", improvement / 100 * 11859)

    # Show diminishing returns
    logging.info("\n📉 Diminishing Returns Analysis:")
    for _, row in df.iterrows():
        if row["window_days"] <= 14:
            logging.info(f"   {int(row['window_days']):2d} days: +{row['marginal_gain']:.1f}% gain (total: {row['match_rate']:.1f}%)")


def main():
    """Main optimization function."""
    logging.info("🚀 FUZZY DATE MATCHING OPTIMIZATION")
    logging.info("=" * 70)

    # Load data
    logging.info("📂 Loading datasets...")
    try:
        atp_data = pd.read_csv("data/cleaned_refactored/atp_matches_standardized.csv")
        pbp_data = pd.read_csv("data/cleaned_refactored/atp_pbp_standardized.csv")
        logging.info("   ATP Data: %,d records", len(atp_data))
        logging.info("   PBP Data: %,d records", len(pbp_data))
    except FileNotFoundError as e:
        logging.error("❌ Error loading data: %s", e)
        return

    # Test different fuzzy windows
    results = test_fuzzy_date_windows(pbp_data, atp_data, max_days=30)

    # Plot results
    plot_fuzzy_optimization_results(results)

    # Recommend optimal window
    recommend_optimal_window(results)

    logging.info("\n✅ Fuzzy date optimization completed!")


if __name__ == "__main__":
    main()
