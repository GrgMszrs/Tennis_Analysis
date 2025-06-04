"""
Tennis Era Analysis - Career Age Curve Analysis
Mixed-effects models for analyzing career trajectories and peak age evolution across eras.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from config.constants import OUTPUT_DATA_DIR


def prepare_career_data(df: pd.DataFrame, min_matches: int = 20) -> pd.DataFrame:
    """
    Prepare player-centric longitudinal data for age curve analysis.

    Args:
        df: Player-match DataFrame
        min_matches: Minimum matches required per player for inclusion

    Returns:
        Filtered DataFrame suitable for longitudinal analysis
    """
    print("=== PREPARING CAREER DATA FOR AGE CURVE ANALYSIS ===")

    # Filter for players with sufficient career data
    player_match_counts = df.groupby("player_id").size()
    sufficient_players = player_match_counts[player_match_counts >= min_matches].index

    career_data = df[df["player_id"].isin(sufficient_players)].copy()

    # Create age groupings for visualization stability
    career_data["age_group"] = pd.cut(career_data["age"], bins=range(15, 45, 2), labels=[f"{i}-{i + 1}" for i in range(15, 43, 2)])

    # Focus on prime career ages (18-38)
    career_data = career_data[(career_data["age"] >= 18) & (career_data["age"] <= 38)].copy()

    print("✅ Career data prepared:")
    print(f"   Players with ≥{min_matches} matches: {len(sufficient_players):,}")
    print(f"   Total career records: {len(career_data):,}")
    print(f"   Age range: {career_data['age'].min()}-{career_data['age'].max()}")

    return career_data


def calculate_ranking_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate peak ATP ranking for each player by era.

    Args:
        df: Career data DataFrame

    Returns:
        DataFrame with peak ranking analysis by player and era
    """
    print("\n=== CALCULATING RANKING PEAKS ===")

    # Use historical ranking where available, fallback to current ranking
    df["effective_rank"] = df["historical_rank"].fillna(df["rank"])

    # Calculate peak ranking (lowest number = best ranking) for each player
    player_peaks = []

    for player_id in df["player_id"].unique():
        player_data = df[df["player_id"] == player_id].copy()

        # Skip players without ranking data
        ranked_matches = player_data.dropna(subset=["effective_rank"])
        if len(ranked_matches) == 0:
            continue

        # Find peak ranking (minimum rank number)
        peak_rank = ranked_matches["effective_rank"].min()
        peak_match = ranked_matches[ranked_matches["effective_rank"] == peak_rank].iloc[0]

        # Calculate career span
        career_start = player_data["age"].min()
        career_end = player_data["age"].max()

        player_peaks.append(
            {
                "player_id": player_id,
                "player_name": peak_match["player_name"],
                "peak_rank": peak_rank,
                "peak_age": peak_match["age"],
                "peak_era": peak_match["era"],
                "career_start_age": career_start,
                "career_end_age": career_end,
                "career_span": career_end - career_start,
                "total_matches": len(player_data),
                "peak_year": peak_match["year"],
            }
        )

    peaks_df = pd.DataFrame(player_peaks)

    print("✅ Peak analysis complete:")
    print(f"   Players analyzed: {len(peaks_df):,}")
    print(f"   Mean peak age: {peaks_df['peak_age'].mean():.1f}")
    print(f"   Peak age range: {peaks_df['peak_age'].min()}-{peaks_df['peak_age'].max()}")

    return peaks_df


def analyze_peak_age_evolution(peaks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how peak ages have evolved across eras.

    Args:
        peaks_df: DataFrame with player peak information

    Returns:
        Dictionary with peak age evolution analysis
    """
    print("\n=== ANALYZING PEAK AGE EVOLUTION ===")

    era_order = ["Classic", "Transition", "Modern", "Current"]
    era_peaks = {}

    for era in era_order:
        era_data = peaks_df[peaks_df["peak_era"] == era]
        if len(era_data) > 0:
            era_peaks[era] = {
                "mean_peak_age": era_data["peak_age"].mean(),
                "median_peak_age": era_data["peak_age"].median(),
                "std_peak_age": era_data["peak_age"].std(),
                "count": len(era_data),
                "peak_ages": era_data["peak_age"].tolist(),
            }

    # Calculate trends
    era_means = [era_peaks[era]["mean_peak_age"] for era in era_order if era in era_peaks]
    if len(era_means) >= 2:
        trend_slope = np.polyfit(range(len(era_means)), era_means, 1)[0]
        trend_direction = "increasing" if trend_slope > 0.1 else "decreasing" if trend_slope < -0.1 else "stable"
    else:
        trend_slope = 0
        trend_direction = "insufficient_data"

    # Statistical test for era differences
    era_groups = [era_peaks[era]["peak_ages"] for era in era_order if era in era_peaks and len(era_peaks[era]["peak_ages"]) > 5]
    if len(era_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*era_groups)
    else:
        f_stat, p_value = np.nan, np.nan

    analysis_results = {
        "era_peaks": era_peaks,
        "trend_slope": trend_slope,
        "trend_direction": trend_direction,
        "statistical_test": {"f_statistic": f_stat, "p_value": p_value, "significant": p_value < 0.05 if not np.isnan(p_value) else False},
        "era_order": era_order,
    }

    print("✅ Peak age evolution analysis:")
    print(f"   Trend: {trend_direction} (slope: {trend_slope:.3f})")
    if not np.isnan(p_value):
        print(f"   Era differences: {'significant' if p_value < 0.05 else 'not significant'} (p={p_value:.3f})")

    return analysis_results


def fit_age_performance_curves(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
    """
    Fit age-performance curves using polynomial regression for each era/surface combination.

    Args:
        df: Career data DataFrame
        metrics: List of z-score metrics to analyze

    Returns:
        Dictionary with fitted curves and optimal ages
    """
    print(f"\n=== FITTING AGE-PERFORMANCE CURVES FOR {len(metrics)} METRICS ===")

    curve_results = {}
    era_order = ["Classic", "Transition", "Modern", "Current"]
    surfaces = ["Hard", "Clay", "Grass"]  # Focus on main surfaces

    for metric in metrics:
        if metric not in df.columns:
            continue

        curve_results[metric] = {}

        for era in era_order:
            era_data = df[df["era"] == era]
            if len(era_data) < 50:  # Minimum data requirement
                continue

            curve_results[metric][era] = {}

            for surface in surfaces:
                surface_data = era_data[era_data["surface"] == surface]

                # Calculate mean performance by age
                age_performance = surface_data.groupby("age").agg({metric: ["mean", "count"]}).reset_index()
                age_performance.columns = ["age", "mean_performance", "count"]

                # Filter for ages with sufficient data
                reliable_data = age_performance[age_performance["count"] >= 10]

                if len(reliable_data) < 5:  # Need minimum points for curve fitting
                    continue

                # Fit polynomial curve (degree 2 for parabolic peak)
                try:
                    ages = reliable_data["age"].values
                    performance = reliable_data["mean_performance"].values

                    # Polynomial features
                    poly_features = PolynomialFeatures(degree=2)
                    ages_poly = poly_features.fit_transform(ages.reshape(-1, 1))

                    # Fit model
                    model = LinearRegression()
                    model.fit(ages_poly, performance)

                    # Find optimal age (vertex of parabola)
                    # For y = ax² + bx + c, optimal x = -b/(2a)
                    coef = model.coef_
                    if len(coef) >= 3 and coef[2] != 0:  # Quadratic term exists and non-zero
                        optimal_age = -coef[1] / (2 * coef[2])

                        # Ensure optimal age is within reasonable bounds
                        if 18 <= optimal_age <= 38:
                            optimal_performance = model.predict(poly_features.transform([[optimal_age]]))[0]
                        else:
                            optimal_age = ages[np.argmax(performance)]
                            optimal_performance = np.max(performance)
                    else:
                        # Fallback to empirical maximum
                        optimal_age = ages[np.argmax(performance)]
                        optimal_performance = np.max(performance)

                    curve_results[metric][era][surface] = {
                        "optimal_age": optimal_age,
                        "optimal_performance": optimal_performance,
                        "model_r2": model.score(ages_poly, performance),
                        "data_points": len(reliable_data),
                        "age_range": (ages.min(), ages.max()),
                        "model": model,
                        "poly_features": poly_features,
                    }

                except Exception as e:
                    print(f"   Warning: Failed to fit curve for {era}/{surface}/{metric}: {e}")
                    continue

    print("✅ Age-performance curves fitted")

    return curve_results


def create_age_curve_visualizations(
    career_data: pd.DataFrame, peaks_df: pd.DataFrame, curve_results: Dict[str, Any], output_dir: Path = None
) -> None:
    """
    Create comprehensive age curve visualizations.

    Args:
        career_data: Career trajectory data
        peaks_df: Peak ranking analysis
        curve_results: Fitted curve results
        output_dir: Directory to save plots
    """
    print("\n=== CREATING AGE CURVE VISUALIZATIONS ===")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Peak age evolution by era
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot of peak ages by era
    era_order = ["Classic", "Transition", "Modern", "Current"]
    plot_data = peaks_df[peaks_df["peak_era"].isin(era_order)]

    sns.boxplot(data=plot_data, x="peak_era", y="peak_age", ax=ax1, order=era_order)
    ax1.set_title("Peak Age Distribution by Era", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Era", fontsize=12)
    ax1.set_ylabel("Peak Age", fontsize=12)

    # Mean peak age trend
    era_means = plot_data.groupby("peak_era")["peak_age"].mean().reindex(era_order)
    ax2.plot(era_order, era_means, "o-", linewidth=3, markersize=10)
    ax2.set_title("Mean Peak Age Trend Across Eras", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Era", fontsize=12)
    ax2.set_ylabel("Mean Peak Age", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add trend line
    x_numeric = range(len(era_means.dropna()))
    y_values = era_means.dropna().values
    if len(y_values) >= 2:
        z = np.polyfit(x_numeric, y_values, 1)
        p = np.poly1d(z)
        ax2.plot(era_order[: len(y_values)], p(x_numeric), "--", alpha=0.7, color="red")

        # Add slope annotation
        slope = z[0]
        ax2.text(
            0.7,
            0.95,
            f"Trend: {slope:+.2f} years/era",
            transform=ax2.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "peak_age_evolution.png", dpi=300, bbox_inches="tight")
        print("  📊 Saved peak age evolution plot")

    plt.show()

    # 2. Performance curves by key metric
    key_metrics = ["service_dominance_z_year_surface", "ace_rate_z_year_surface", "return_win_pct_z_year_surface"]
    available_metrics = [m for m in key_metrics if m in curve_results]

    if available_metrics:
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(14, 5 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            # Calculate age-performance curves for visualization
            for era in era_order:
                era_data = career_data[career_data["era"] == era]
                if len(era_data) < 30:
                    continue

                # Aggregate across all surfaces for cleaner visualization
                age_performance = era_data.groupby("age")[metric].mean().reset_index()

                if len(age_performance) >= 5:
                    axes[i].plot(age_performance["age"], age_performance[metric], "o-", label=f"{era} Era", linewidth=2, markersize=6)

            axes[i].set_title(f"{metric.replace('_', ' ').title()} by Age and Era", fontsize=14, fontweight="bold")
            axes[i].set_xlabel("Age", fontsize=12)
            axes[i].set_ylabel("Z-Score Performance", fontsize=12)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / "age_performance_curves.png", dpi=300, bbox_inches="tight")
            print("  📊 Saved age-performance curves plot")

        plt.show()


def analyze_career_trajectories(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete career trajectory analysis pipeline.

    Args:
        df: Player-match DataFrame

    Returns:
        Dictionary with complete career analysis results
    """
    print("🎾 CAREER AGE CURVE ANALYSIS")
    print("=" * 50)

    # Prepare career data
    career_data = prepare_career_data(df)

    # Calculate ranking peaks
    peaks_df = calculate_ranking_peaks(career_data)

    # Analyze peak age evolution
    peak_evolution = analyze_peak_age_evolution(peaks_df)

    # Define key z-score metrics for curve analysis
    z_score_metrics = [
        "service_dominance_z_year_surface",
        "ace_rate_z_year_surface",
        "first_serve_win_pct_z_year_surface",
        "return_win_pct_z_year_surface",
        "break_point_save_pct_z_year_surface",
    ]

    # Fit age-performance curves
    curve_results = fit_age_performance_curves(career_data, z_score_metrics)

    # Create visualizations
    plots_dir = OUTPUT_DATA_DIR / "plots" / "age_curves"
    create_age_curve_visualizations(career_data, peaks_df, curve_results, plots_dir)

    # Compile results
    analysis_results = {
        "career_data_summary": {
            "total_players": career_data["player_id"].nunique(),
            "total_records": len(career_data),
            "age_range": (career_data["age"].min(), career_data["age"].max()),
            "era_distribution": career_data["era"].value_counts().to_dict(),
        },
        "peak_analysis": {"peaks_data": peaks_df, "evolution_results": peak_evolution},
        "curve_analysis": curve_results,
        "output_location": plots_dir,
    }

    # Print summary
    print("\n=== CAREER ANALYSIS SUMMARY ===")
    print(f"✅ Players analyzed: {career_data['player_id'].nunique():,}")
    print(f"✅ Peak age analysis: {len(peaks_df)} players")
    print(f"✅ Performance curves fitted for {len([m for m in z_score_metrics if m in curve_results])} metrics")
    print(f"📊 Visualizations saved to: {plots_dir}")

    # Key findings
    if peak_evolution["statistical_test"]["significant"]:
        print(f"🔍 KEY FINDING: Peak age evolution is statistically significant (p={peak_evolution['statistical_test']['p_value']:.3f})")
        print(f"    Trend: {peak_evolution['trend_direction']} ({peak_evolution['trend_slope']:+.2f} years/era)")
    else:
        print("🔍 KEY FINDING: No significant peak age evolution detected across eras")

    return analysis_results
