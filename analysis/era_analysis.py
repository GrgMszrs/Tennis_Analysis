"""
Tennis Era Analysis - Era Performance Analysis
Statistical analysis of tennis performance metrics across different eras.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.constants import CLEANED_DATA_DIR, OUTPUT_DATA_DIR


def load_player_match_data() -> pd.DataFrame:
    """
    Load the transformed player-match data.

    Returns:
        Player-match DataFrame
    """
    print("=== LOADING PLAYER-MATCH DATA ===")

    # Try enhanced version first (primary per file naming strategy)
    enhanced_file = CLEANED_DATA_DIR / "atp_player_match_enhanced.csv"

    data_file = None
    dataset_type = None

    if enhanced_file.exists():
        data_file = enhanced_file
        dataset_type = "enhanced"
    else:
        raise FileNotFoundError(f"Player-match data not found. Expected files:\n" f"  Primary: {enhanced_file}")

    df = pd.read_csv(data_file)

    # Convert date column
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    print(f"✅ Loaded player-match data ({dataset_type}): {len(df):,} rows")
    print(f"   File: {data_file.name}")
    print(f"   Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    print(f"   Eras: {', '.join(df['era'].unique())}")

    # Show enhanced features if available
    if dataset_type == "enhanced":
        z_score_cols = [col for col in df.columns if "_z_" in col]
        ranking_cols = [col for col in df.columns if "historical" in col]
        if z_score_cols or ranking_cols:
            print(f"   Enhanced features: {len(z_score_cols)} z-score + {len(ranking_cols)} ranking columns")

    return df


def compute_era_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical summaries by era.

    Args:
        df: Player-match DataFrame

    Returns:
        Era statistics DataFrame
    """
    print("\n=== COMPUTING ERA STATISTICS ===")

    # Define metrics to analyze
    metrics = [
        "ace_rate",
        "df_rate",
        "first_serve_pct",
        "first_serve_win_pct",
        "second_serve_win_pct",
        "break_point_save_pct",
        "return_win_pct",
        "service_dominance",
    ]

    # Compute statistics by era
    era_stats = []

    for era in df["era"].unique():
        if era == "Unknown":
            continue

        era_data = df[df["era"] == era]

        era_summary = {
            "era": era,
            "matches": era_data["match_id"].nunique(),
            "players": era_data["player_name"].nunique(),
            "total_rows": len(era_data),
        }

        # Compute statistics for each metric
        for metric in metrics:
            if metric in era_data.columns:
                values = era_data[metric].dropna()
                if len(values) > 0:
                    era_summary.update(
                        {
                            f"{metric}_mean": values.mean(),
                            f"{metric}_std": values.std(),
                            f"{metric}_median": values.median(),
                            f"{metric}_q25": values.quantile(0.25),
                            f"{metric}_q75": values.quantile(0.75),
                        }
                    )

        era_stats.append(era_summary)

    era_stats_df = pd.DataFrame(era_stats)

    print(f"✅ Era statistics computed for {len(era_stats_df)} eras")

    return era_stats_df


def analyze_era_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trends across eras.

    Args:
        df: Player-match DataFrame

    Returns:
        Dictionary with trend analysis results
    """
    print("\n=== ANALYZING ERA TRENDS ===")

    # Define key metrics for trend analysis
    trend_metrics = ["ace_rate", "df_rate", "first_serve_pct", "first_serve_win_pct", "break_point_save_pct", "service_dominance"]

    # Compute era means
    era_means = df.groupby("era")[trend_metrics].mean()

    # Order eras chronologically
    era_order = ["Classic", "Transition", "Modern", "Current"]
    era_means = era_means.reindex([era for era in era_order if era in era_means.index])

    # Analyze trends
    trends = {}

    for metric in trend_metrics:
        if metric in era_means.columns:
            values = era_means[metric].values

            # Calculate trend direction
            if len(values) >= 2:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]

                # Classify trend
                if abs(trend_slope) < 0.001:
                    trend_direction = "stable"
                elif trend_slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"

                trends[metric] = {
                    "direction": trend_direction,
                    "slope": trend_slope,
                    "values": values.tolist(),
                    "eras": era_means.index.tolist(),
                }

    print(f"✅ Trend analysis complete for {len(trends)} metrics")

    # Print key findings
    print("\n📈 KEY TRENDS:")
    for metric, trend_data in trends.items():
        direction = trend_data["direction"]
        print(f"  {metric}: {direction}")

    return {"era_means": era_means, "trends": trends, "era_order": era_order}


def compare_surface_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance across surfaces by era.

    Args:
        df: Player-match DataFrame

    Returns:
        Surface comparison DataFrame
    """
    print("\n=== ANALYZING SURFACE PERFORMANCE ===")

    # Key metrics for surface analysis
    surface_metrics = ["ace_rate", "first_serve_win_pct", "break_point_save_pct"]

    # Compute surface statistics by era
    surface_stats = df.groupby(["era", "surface"])[surface_metrics].agg(["mean", "count"]).round(4)

    # Flatten column names
    surface_stats.columns = ["_".join(col).strip() for col in surface_stats.columns]
    surface_stats = surface_stats.reset_index()

    print("✅ Surface analysis complete")
    print(f"   Surfaces analyzed: {', '.join(df['surface'].unique())}")

    return surface_stats


def identify_era_champions(df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Identify top performers in each era.

    Args:
        df: Player-match DataFrame
        top_n: Number of top players to identify per era

    Returns:
        Dictionary with top performers by era
    """
    print(f"\n=== IDENTIFYING ERA CHAMPIONS (TOP {top_n}) ===")

    era_champions = {}

    for era in df["era"].unique():
        if era == "Unknown":
            continue

        era_data = df[df["era"] == era]

        # Calculate player performance metrics
        player_stats = (
            era_data.groupby("player_name")
            .agg(
                {
                    "won_match": ["sum", "count"],
                    "ace_rate": "mean",
                    "first_serve_win_pct": "mean",
                    "break_point_save_pct": "mean",
                    "service_dominance": "mean",
                    "match_id": "nunique",
                }
            )
            .round(4)
        )

        # Flatten column names
        player_stats.columns = ["_".join(col).strip() for col in player_stats.columns]
        player_stats = player_stats.reset_index()

        # Calculate win percentage
        player_stats["win_percentage"] = (player_stats["won_match_sum"] / player_stats["won_match_count"] * 100).round(2)

        # Filter players with minimum matches (at least 10)
        min_matches = 10
        qualified_players = player_stats[player_stats["won_match_count"] >= min_matches]

        # Sort by win percentage
        top_players = qualified_players.nlargest(top_n, "win_percentage")

        era_champions[era] = top_players

        print(f"  {era}: {len(qualified_players)} qualified players, top win rate: {top_players['win_percentage'].iloc[0]:.1f}%")

    return era_champions


def create_era_comparison_plots(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Create visualization plots for era comparison.

    Args:
        df: Player-match DataFrame
        output_dir: Directory to save plots (optional)
    """
    print("\n=== CREATING ERA COMPARISON PLOTS ===")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Key metrics to plot
    plot_metrics = [
        ("ace_rate", "Ace Rate", "Aces per Service Point"),
        ("first_serve_win_pct", "First Serve Win %", "First Serve Win Percentage"),
        ("break_point_save_pct", "Break Point Save %", "Break Point Save Percentage"),
        ("service_dominance", "Service Dominance", "Service Points Won / Total Points"),
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (metric, title, ylabel) in enumerate(plot_metrics):
        if metric in df.columns:
            # Create box plot
            era_order = ["Classic", "Transition", "Modern", "Current"]
            plot_data = df[df["era"].isin(era_order)]

            sns.boxplot(data=plot_data, x="era", y=metric, ax=axes[i], order=era_order)
            axes[i].set_title(f"{title} by Era", fontsize=14, fontweight="bold")
            axes[i].set_xlabel("Era", fontsize=12)
            axes[i].set_ylabel(ylabel, fontsize=12)
            axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if output_dir:
        plot_file = output_dir / "era_comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"  📊 Saved plots to: {plot_file}")

    plt.show()


def generate_era_analysis_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive era analysis report.

    Args:
        df: Player-match DataFrame

    Returns:
        Dictionary with complete analysis results
    """
    print("🎾 TENNIS ERA ANALYSIS - COMPREHENSIVE REPORT")
    print("=" * 60)

    # Run all analyses
    era_stats = compute_era_statistics(df)
    trend_analysis = analyze_era_trends(df)
    surface_analysis = compare_surface_performance(df)
    era_champions = identify_era_champions(df)

    # Create visualizations with output directory per file naming strategy
    plots_dir = OUTPUT_DATA_DIR / "plots"
    create_era_comparison_plots(df, output_dir=plots_dir)

    # Compile report
    report = {
        "dataset_summary": {
            "total_rows": len(df),
            "unique_players": df["player_name"].nunique(),
            "unique_matches": df["match_id"].nunique(),
            "date_range": (df["tourney_date"].min(), df["tourney_date"].max()),
            "eras": df["era"].value_counts().to_dict(),
        },
        "era_statistics": era_stats,
        "trend_analysis": trend_analysis,
        "surface_analysis": surface_analysis,
        "era_champions": era_champions,
    }

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"✅ Dataset: {len(df):,} player-match records")
    print(f"✅ Era statistics computed for {len(era_stats)} eras")
    print(f"✅ Trend analysis for {len(trend_analysis['trends'])} metrics")
    print(f"✅ Surface analysis across {len(surface_analysis['surface'].unique())} surfaces")
    print(f"✅ Era champions identified for {len(era_champions)} eras")
    print(f"📊 Plots saved to: {plots_dir}")

    return report


if __name__ == "__main__":
    # Load data and run analysis
    player_data = load_player_match_data()
    report = generate_era_analysis_report(player_data)
