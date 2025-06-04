#!/usr/bin/env python3
"""
Tennis Era Analysis - Joinability Heatmap Analysis
Investigates unmatched PBP records by analyzing joinability patterns.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def load_datasets():
    """Load standardized datasets for joinability analysis."""
    print("📂 Loading standardized datasets...")

    try:
        # Load standardized datasets
        atp_matches = pd.read_csv("data/cleaned_refactored/atp_matches_standardized.csv")
        pbp_data = pd.read_csv("data/cleaned_refactored/atp_pbp_standardized.csv")

        print(f"   ATP Matches: {len(atp_matches):,} records")
        print(f"   PBP Data: {len(pbp_data):,} records")

        return atp_matches, pbp_data

    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Run the standardization phase first: python main.py --phase 1")
        sys.exit(1)


def create_normalized_keys(df: pd.DataFrame, player_cols: List[str], date_col: str, round_col: str = None) -> pd.DataFrame:
    """Create normalized keys for matching analysis."""
    result = df.copy()

    # Normalize player names
    for col in player_cols:
        if col in result.columns:
            result[f"{col}_norm"] = (
                result[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
            )

    # Normalize dates
    result[f"{date_col}_norm"] = pd.to_datetime(result[date_col], errors="coerce").dt.date

    # Normalize rounds if present
    if round_col and round_col in result.columns:
        result[f"{round_col}_norm"] = result[round_col].fillna("").astype(str).str.strip().str.upper()

    return result


def analyze_player_presence_by_grouping(atp_matches: pd.DataFrame, pbp_data: pd.DataFrame) -> Dict:
    """Analyze whether PBP players appear in ATP data by different grouping strategies."""
    print("\n🎾 PLAYER PRESENCE ANALYSIS BY GROUPINGS")
    print("=" * 60)

    # Normalize data
    atp_norm = create_normalized_keys(atp_matches, ["winner_name", "loser_name"], "tourney_date", "round")

    pbp_norm = create_normalized_keys(pbp_data, ["server1", "server2"], "date_standardized")

    # Create ATP player sets by different groupings
    atp_players_global = set()
    for _, row in atp_norm.iterrows():
        if row["winner_name_norm"]:
            atp_players_global.add(row["winner_name_norm"])
        if row["loser_name_norm"]:
            atp_players_global.add(row["loser_name_norm"])

    # Group ATP players by date
    atp_players_by_date = defaultdict(set)
    for _, row in atp_norm.iterrows():
        date = row["tourney_date_norm"]
        if pd.notna(date):
            if row["winner_name_norm"]:
                atp_players_by_date[date].add(row["winner_name_norm"])
            if row["loser_name_norm"]:
                atp_players_by_date[date].add(row["loser_name_norm"])

    # Group ATP players by date + round
    atp_players_by_date_round = defaultdict(set)
    for _, row in atp_norm.iterrows():
        date = row["tourney_date_norm"]
        round_val = row["round_norm"]
        if pd.notna(date) and round_val:
            key = (date, round_val)
            if row["winner_name_norm"]:
                atp_players_by_date_round[key].add(row["winner_name_norm"])
            if row["loser_name_norm"]:
                atp_players_by_date_round[key].add(row["loser_name_norm"])

    # Analyze PBP records
    results = {"global_matches": 0, "date_matches": 0, "date_round_matches": 0, "no_matches": 0, "total_pbp": 0, "unmatched_details": []}

    for _, row in pbp_norm.iterrows():
        results["total_pbp"] += 1

        player1 = row["server1_norm"]
        player2 = row["server2_norm"]
        date = row["date_standardized_norm"]

        # Check global presence
        p1_global = player1 in atp_players_global if player1 else False
        p2_global = player2 in atp_players_global if player2 else False

        # Check date-specific presence
        p1_date = player1 in atp_players_by_date[date] if player1 and pd.notna(date) else False
        p2_date = player2 in atp_players_by_date[date] if player2 and pd.notna(date) else False

        # Check date+round presence (use a reasonable round approximation)
        p1_date_round = False
        p2_date_round = False
        if pd.notna(date):
            for round_key in ["F", "SF", "QF", "R16", "R32", "R64", "R128"]:
                key = (date, round_key)
                if key in atp_players_by_date_round:
                    if player1 in atp_players_by_date_round[key]:
                        p1_date_round = True
                    if player2 in atp_players_by_date_round[key]:
                        p2_date_round = True

        # Categorize match
        if p1_global and p2_global:
            results["global_matches"] += 1
            if p1_date and p2_date:
                results["date_matches"] += 1
                if p1_date_round and p2_date_round:
                    results["date_round_matches"] += 1
        else:
            results["no_matches"] += 1
            results["unmatched_details"].append(
                {
                    "player1": player1,
                    "player2": player2,
                    "date": date,
                    "p1_global": p1_global,
                    "p2_global": p2_global,
                    "p1_date": p1_date,
                    "p2_date": p2_date,
                    "tournament": row.get("tny_name", ""),
                }
            )

    # Print results
    total = results["total_pbp"]
    print(f"📊 PBP Joinability Analysis (Total: {total:,} records)")
    print(f"   Global player presence:     {results['global_matches']:,} ({results['global_matches'] / total * 100:.1f}%)")
    print(f"   Date-specific presence:     {results['date_matches']:,} ({results['date_matches'] / total * 100:.1f}%)")
    print(f"   Date+Round presence:        {results['date_round_matches']:,} ({results['date_round_matches'] / total * 100:.1f}%)")
    print(f"   No ATP presence:            {results['no_matches']:,} ({results['no_matches'] / total * 100:.1f}%)")

    return results


def create_joinability_heatmap(results: Dict, atp_matches: pd.DataFrame, pbp_data: pd.DataFrame):
    """Create visual heatmap of joinability patterns."""
    print("\n📈 CREATING JOINABILITY HEATMAP")
    print("=" * 50)

    # Create monthly aggregations
    pbp_data_with_dates = pbp_data.copy()
    pbp_data_with_dates["date_standardized"] = pd.to_datetime(pbp_data_with_dates["date_standardized"])
    pbp_data_with_dates["year_month"] = pbp_data_with_dates["date_standardized"].dt.to_period("M")

    atp_data_with_dates = atp_matches.copy()
    atp_data_with_dates["tourney_date"] = pd.to_datetime(atp_data_with_dates["tourney_date"])
    atp_data_with_dates["year_month"] = atp_data_with_dates["tourney_date"].dt.to_period("M")

    # Count records by month
    pbp_monthly = pbp_data_with_dates.groupby("year_month").size()
    atp_monthly = atp_data_with_dates.groupby("year_month").size()

    # Analyze unmatched records by month
    unmatched_details = pd.DataFrame(results["unmatched_details"])
    if not unmatched_details.empty:
        unmatched_details["date"] = pd.to_datetime(unmatched_details["date"], errors="coerce")
        unmatched_details["year_month"] = unmatched_details["date"].dt.to_period("M")
        unmatched_monthly = unmatched_details.groupby("year_month").size()
    else:
        unmatched_monthly = pd.Series(dtype=int)

    # Create comprehensive date range
    all_months = pd.period_range(
        start=min(pbp_monthly.index.min(), atp_monthly.index.min()), end=max(pbp_monthly.index.max(), atp_monthly.index.max()), freq="M"
    )

    # Prepare heatmap data
    heatmap_data = pd.DataFrame(index=all_months)
    heatmap_data["PBP_Records"] = pbp_monthly.reindex(all_months, fill_value=0)
    heatmap_data["ATP_Records"] = atp_monthly.reindex(all_months, fill_value=0)
    heatmap_data["Unmatched_PBP"] = unmatched_monthly.reindex(all_months, fill_value=0)
    heatmap_data["Matchable_PBP"] = heatmap_data["PBP_Records"] - heatmap_data["Unmatched_PBP"]
    heatmap_data["Joinability_Rate"] = np.where(
        heatmap_data["PBP_Records"] > 0, heatmap_data["Matchable_PBP"] / heatmap_data["PBP_Records"] * 100, 0
    )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Tennis Data Joinability Heatmap Analysis", fontsize=16, fontweight="bold")

    # 1. PBP vs ATP Record Counts by Month
    ax1 = axes[0, 0]
    months_str = [str(m) for m in heatmap_data.index]
    x_pos = np.arange(len(months_str))

    ax1.bar(x_pos, heatmap_data["ATP_Records"], alpha=0.7, label="ATP Records", color="blue")
    ax1.bar(x_pos, heatmap_data["PBP_Records"], alpha=0.7, label="PBP Records", color="orange")
    ax1.set_title("Record Counts by Month")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Record Count")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Sample every 6 months for readability
    step = max(1, len(months_str) // 20)
    ax1.set_xticks(x_pos[::step])
    ax1.set_xticklabels([months_str[i] for i in x_pos[::step]], rotation=45)

    # 2. Joinability Rate Heatmap
    ax2 = axes[0, 1]
    joinability_matrix = heatmap_data["Joinability_Rate"].values.reshape(1, -1)
    im = ax2.imshow(joinability_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax2.set_title("Joinability Rate by Month (%)")
    ax2.set_xlabel("Month")
    ax2.set_yticks([])
    ax2.set_xticks(x_pos[::step])
    ax2.set_xticklabels([months_str[i] for i in x_pos[::step]], rotation=45)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Joinability Rate (%)")

    # 3. Temporal Overlap Analysis
    ax3 = axes[1, 0]
    overlap_data = heatmap_data[(heatmap_data["PBP_Records"] > 0) & (heatmap_data["ATP_Records"] > 0)]
    ax3.scatter(
        overlap_data["ATP_Records"], overlap_data["PBP_Records"], c=overlap_data["Joinability_Rate"], cmap="RdYlGn", s=50, alpha=0.7
    )
    ax3.set_xlabel("ATP Records")
    ax3.set_ylabel("PBP Records")
    ax3.set_title("Data Volume vs Joinability")

    # Add colorbar
    scatter = ax3.scatter(
        overlap_data["ATP_Records"], overlap_data["PBP_Records"], c=overlap_data["Joinability_Rate"], cmap="RdYlGn", s=50, alpha=0.7
    )
    cbar2 = plt.colorbar(scatter, ax=ax3)
    cbar2.set_label("Joinability Rate (%)")

    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate summary stats
    total_pbp = results["total_pbp"]
    global_match_rate = results["global_matches"] / total_pbp * 100
    date_match_rate = results["date_matches"] / total_pbp * 100
    date_round_match_rate = results["date_round_matches"] / total_pbp * 100
    no_match_rate = results["no_matches"] / total_pbp * 100

    summary_text = f"""
JOINABILITY SUMMARY

Total PBP Records: {total_pbp:,}

Player Presence Analysis:
• Global Presence: {results["global_matches"]:,} ({global_match_rate:.1f}%)
• Date-Specific: {results["date_matches"]:,} ({date_match_rate:.1f}%)  
• Date+Round: {results["date_round_matches"]:,} ({date_round_match_rate:.1f}%)
• No ATP Presence: {results["no_matches"]:,} ({no_match_rate:.1f}%)

Data Overlap:
• PBP Date Range: {heatmap_data[heatmap_data["PBP_Records"] > 0].index.min()} to {heatmap_data[heatmap_data["PBP_Records"] > 0].index.max()}
• ATP Date Range: {heatmap_data[heatmap_data["ATP_Records"] > 0].index.min()} to {heatmap_data[heatmap_data["ATP_Records"] > 0].index.max()}
• Overlap Months: {len(overlap_data):,}

Best Joinability Rate: {heatmap_data["Joinability_Rate"].max():.1f}%
Average Joinability: {heatmap_data[heatmap_data["PBP_Records"] > 0]["Joinability_Rate"].mean():.1f}%
"""

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    output_path = Path("data/output/plots/joinability_heatmap_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Heatmap saved to: {output_path}")

    plt.show()


def analyze_tournament_mismatch_patterns(results: Dict):
    """Analyze tournament naming patterns in unmatched records."""
    print("\n🏆 TOURNAMENT MISMATCH PATTERN ANALYSIS")
    print("=" * 60)

    if not results["unmatched_details"]:
        print("   No unmatched records to analyze")
        return

    unmatched_df = pd.DataFrame(results["unmatched_details"])

    # Count by tournament
    tournament_counts = unmatched_df["tournament"].value_counts()

    print("📊 Unmatched records by tournament (Top 10):")
    for i, (tournament, count) in enumerate(tournament_counts.head(10).items()):
        print(f"   {i + 1:2d}. {tournament[:50]:<50} ({count:,} records)")

    # Analyze player presence patterns
    print("\n👥 Player presence patterns in unmatched records:")

    both_missing = unmatched_df[~unmatched_df["p1_global"] & ~unmatched_df["p2_global"]]
    p1_only = unmatched_df[unmatched_df["p1_global"] & ~unmatched_df["p2_global"]]
    p2_only = unmatched_df[~unmatched_df["p1_global"] & unmatched_df["p2_global"]]

    total_unmatched = len(unmatched_df)
    print(f"   Both players missing:    {len(both_missing):,} ({len(both_missing) / total_unmatched * 100:.1f}%)")
    print(f"   Only player1 in ATP:     {len(p1_only):,} ({len(p1_only) / total_unmatched * 100:.1f}%)")
    print(f"   Only player2 in ATP:     {len(p2_only):,} ({len(p2_only) / total_unmatched * 100:.1f}%)")


def main():
    """Main analysis function."""
    print("🔍 TENNIS DATA JOINABILITY HEATMAP ANALYSIS")
    print("=" * 70)

    # Load data
    atp_matches, pbp_data = load_datasets()

    # Run joinability analysis
    results = analyze_player_presence_by_grouping(atp_matches, pbp_data)

    # Create heatmap visualization
    create_joinability_heatmap(results, atp_matches, pbp_data)

    # Analyze tournament patterns
    analyze_tournament_mismatch_patterns(results)

    # Final recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("1. 🎯 Focus embedding matching on the overlapping temporal period")
    print("2. 🏆 Implement tournament name normalization (URL → standard names)")
    print("3. 👥 Pre-filter PBP data to only include players present in ATP data")
    print("4. 📅 Implement date-fuzzy matching (±1-2 days) for scheduling differences")
    print("5. 🔍 Investigate why certain tournaments have zero joinability")

    print("\n✅ Joinability analysis completed!")
    print("   📊 Results saved to: data/output/plots/joinability_heatmap_analysis.png")


if __name__ == "__main__":
    main()
