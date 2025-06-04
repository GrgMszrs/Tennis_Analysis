"""
Tennis Era Analysis - Yearly Trends Analysis
Granular year-by-year analysis of tennis evolution and performance trends.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

from config.constants import OUTPUT_DATA_DIR


def prepare_yearly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for yearly trend analysis.

    Args:
        df: Player-match DataFrame

    Returns:
        DataFrame aggregated by year with trend metrics
    """
    print("=== PREPARING YEARLY TREND DATA ===")

    # Core performance metrics to analyze
    core_metrics = [
        "ace_rate",
        "df_rate",
        "first_serve_pct",
        "first_serve_win_pct",
        "second_serve_win_pct",
        "break_point_save_pct",
        "return_win_pct",
        "service_dominance",
    ]

    # Z-score metrics for normalized trends
    z_score_metrics = [f"{metric}_z_year" for metric in core_metrics if f"{metric}_z_year" in df.columns]

    # Age and ranking metrics
    demographic_metrics = ["age", "rank", "historical_rank"]

    # Calculate yearly statistics
    yearly_stats = []

    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year]

        year_summary = {
            "year": year,
            "total_matches": year_data["match_id"].nunique(),
            "total_players": year_data["player_id"].nunique(),
            "total_records": len(year_data),
        }

        # Core performance metrics - means and stds
        for metric in core_metrics:
            if metric in year_data.columns:
                values = year_data[metric].dropna()
                if len(values) > 10:  # Minimum sample size
                    year_summary.update(
                        {
                            f"{metric}_mean": values.mean(),
                            f"{metric}_std": values.std(),
                            f"{metric}_median": values.median(),
                            f"{metric}_q75": values.quantile(0.75),
                            f"{metric}_q25": values.quantile(0.25),
                            f"{metric}_count": len(values),
                        }
                    )

        # Z-score metrics (should center around 0 by design)
        for metric in z_score_metrics:
            if metric in year_data.columns:
                values = year_data[metric].dropna()
                if len(values) > 10:
                    year_summary.update({f"{metric}_mean": values.mean(), f"{metric}_std": values.std()})

        # Demographic trends
        for metric in demographic_metrics:
            if metric in year_data.columns:
                values = year_data[metric].dropna()
                if len(values) > 10:
                    year_summary.update({f"{metric}_mean": values.mean(), f"{metric}_median": values.median()})

        # Game style diversity metrics
        year_summary.update(
            {
                "ace_rate_iqr": year_data["ace_rate"].quantile(0.75) - year_data["ace_rate"].quantile(0.25)
                if "ace_rate" in year_data.columns
                else None,
                "service_diversity": year_data["service_dominance"].std() if "service_dominance" in year_data.columns else None,
                "return_diversity": year_data["return_win_pct"].std() if "return_win_pct" in year_data.columns else None,
            }
        )

        yearly_stats.append(year_summary)

    yearly_df = pd.DataFrame(yearly_stats)

    print("✅ Yearly data prepared:")
    print(f"   Years analyzed: {yearly_df['year'].min()}-{yearly_df['year'].max()}")
    print(f"   Total yearly records: {len(yearly_df)}")
    print(
        f"   Metrics per year: {len([col for col in yearly_df.columns if col not in ['year', 'total_matches', 'total_players', 'total_records']])}"
    )

    return yearly_df


def detect_trend_changes(years: np.ndarray, values: np.ndarray, min_segment_length: int = 3) -> List[Dict[str, Any]]:
    """
    Detect significant trend changes using piecewise linear regression.

    Args:
        years: Array of years
        values: Array of metric values
        min_segment_length: Minimum length of trend segments

    Returns:
        List of detected change points with statistics
    """
    if len(years) < 6:  # Need minimum data for change point detection
        return []

    change_points = []

    # Try different potential change points
    for i in range(min_segment_length, len(years) - min_segment_length):
        # Split data at potential change point
        years1, values1 = years[:i], values[:i]
        years2, values2 = years[i:], values[i:]

        # Fit linear models to each segment
        try:
            slope1 = np.polyfit(years1, values1, 1)[0]
            slope2 = np.polyfit(years2, values2, 1)[0]

            # Calculate R² for combined vs separate models
            # Combined model
            combined_slope = np.polyfit(years, values, 1)[0]
            combined_pred = combined_slope * (years - years[0]) + values[0]
            combined_r2 = r2_score(values, combined_pred)

            # Separate models
            pred1 = slope1 * (years1 - years1[0]) + values1[0]
            pred2 = slope2 * (years2 - years2[0]) + values2[0]
            separate_pred = np.concatenate([pred1, pred2])
            separate_r2 = r2_score(values, separate_pred)

            # Check if split improves model significantly
            improvement = separate_r2 - combined_r2
            slope_diff = abs(slope2 - slope1)

            if improvement > 0.1 and slope_diff > 0.001:  # Thresholds for significance
                change_points.append(
                    {
                        "year": years[i],
                        "index": i,
                        "slope_before": slope1,
                        "slope_after": slope2,
                        "slope_change": slope2 - slope1,
                        "r2_improvement": improvement,
                    }
                )

        except np.RankWarning:
            continue

    # Sort by R² improvement and return top changes
    change_points.sort(key=lambda x: x["r2_improvement"], reverse=True)
    return change_points[:3]  # Return top 3 change points


def analyze_yearly_trends(yearly_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze year-over-year trends in tennis performance metrics.

    Args:
        yearly_df: DataFrame with yearly aggregated data

    Returns:
        Dictionary with comprehensive trend analysis
    """
    print("\n=== ANALYZING YEARLY TRENDS ===")

    # Key metrics for trend analysis
    trend_metrics = [
        "ace_rate_mean",
        "df_rate_mean",
        "first_serve_pct_mean",
        "first_serve_win_pct_mean",
        "second_serve_win_pct_mean",
        "break_point_save_pct_mean",
        "return_win_pct_mean",
        "service_dominance_mean",
        "age_mean",
    ]

    trends_analysis = {}

    for metric in trend_metrics:
        if metric not in yearly_df.columns:
            continue

        # Remove NaN values
        valid_data = yearly_df[["year", metric]].dropna()
        if len(valid_data) < 5:
            continue

        years_clean = valid_data["year"].values
        values = valid_data[metric].values

        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years_clean, values)

        # Trend classification
        if p_value < 0.05:
            if slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # Calculate trend strength
        trend_strength = abs(r_value)

        # Detect change points
        change_points = detect_trend_changes(years_clean, values)

        # Calculate acceleration (second derivative)
        if len(values) >= 4:
            # Fit polynomial to detect acceleration
            poly_coef = np.polyfit(years_clean, values, 2)
            acceleration = 2 * poly_coef[0]  # Second derivative
        else:
            acceleration = 0

        # Recent trend (last 5 years)
        if len(years_clean) >= 5:
            recent_years = years_clean[-5:]
            recent_values = values[-5:]
            recent_slope, _, recent_r, recent_p, _ = stats.linregress(recent_years, recent_values)
        else:
            recent_slope, recent_r, recent_p = slope, r_value, p_value

        trends_analysis[metric] = {
            "overall_trend": {
                "direction": trend_direction,
                "slope": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "strength": trend_strength,
            },
            "recent_trend": {"slope": recent_slope, "r_squared": recent_r**2, "p_value": recent_p},
            "acceleration": acceleration,
            "change_points": change_points,
            "data_points": len(values),
            "year_range": (years_clean.min(), years_clean.max()),
            "value_range": (values.min(), values.max()),
            "total_change": values[-1] - values[0] if len(values) > 0 else 0,
        }

    print(f"✅ Yearly trends analyzed for {len(trends_analysis)} metrics")

    # Print key findings
    print("\n📈 KEY YEARLY TRENDS:")
    for metric, trend_data in trends_analysis.items():
        direction = trend_data["overall_trend"]["direction"]
        r2 = trend_data["overall_trend"]["r_squared"]
        print(f"  {metric.replace('_mean', '')}: {direction} (R²={r2:.3f})")

    return trends_analysis


def analyze_game_evolution_phases(yearly_df: pd.DataFrame, trends_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify distinct phases in tennis evolution based on multiple metrics.

    Args:
        yearly_df: Yearly aggregated data
        trends_analysis: Results from yearly trends analysis

    Returns:
        Dictionary with phase analysis results
    """
    print("\n=== ANALYZING GAME EVOLUTION PHASES ===")

    # Collect all significant change points
    all_change_points = []

    for metric, trend_data in trends_analysis.items():
        for cp in trend_data["change_points"]:
            all_change_points.append(
                {"year": cp["year"], "metric": metric, "slope_change": cp["slope_change"], "r2_improvement": cp["r2_improvement"]}
            )

    # Group change points by year (within 2-year windows)
    change_point_years = {}
    for cp in all_change_points:
        year = cp["year"]
        # Find nearest grouped year
        grouped_year = None
        for existing_year in change_point_years.keys():
            if abs(year - existing_year) <= 2:
                grouped_year = existing_year
                break

        if grouped_year is None:
            grouped_year = year
            change_point_years[grouped_year] = []

        change_point_years[grouped_year].append(cp)

    # Identify major transition years (multiple metrics changing)
    major_transitions = {}
    for year, changes in change_point_years.items():
        if len(changes) >= 2:  # Multiple metrics changing
            avg_impact = np.mean([cp["r2_improvement"] for cp in changes])
            major_transitions[year] = {
                "year": year,
                "metrics_affected": [cp["metric"] for cp in changes],
                "average_impact": avg_impact,
                "changes": changes,
            }

    # Define evolution phases based on major transitions
    transition_years = sorted(major_transitions.keys())
    phases = []

    start_year = yearly_df["year"].min()
    for i, transition_year in enumerate(transition_years):
        phase_end = transition_year
        phases.append(
            {
                "phase_id": i + 1,
                "start_year": start_year,
                "end_year": phase_end,
                "duration": phase_end - start_year,
                "transition_year": transition_year,
                "characteristics": major_transitions[transition_year]["metrics_affected"],
            }
        )
        start_year = transition_year + 1

    # Add final phase
    if start_year <= yearly_df["year"].max():
        phases.append(
            {
                "phase_id": len(phases) + 1,
                "start_year": start_year,
                "end_year": yearly_df["year"].max(),
                "duration": yearly_df["year"].max() - start_year,
                "transition_year": None,
                "characteristics": ["current_state"],
            }
        )

    phase_analysis = {"major_transitions": major_transitions, "evolution_phases": phases, "total_phases": len(phases)}

    print("✅ Game evolution analysis complete:")
    print(f"   Major transitions detected: {len(major_transitions)}")
    print(f"   Evolution phases identified: {len(phases)}")

    if major_transitions:
        print("   Key transition years:")
        for year, transition in major_transitions.items():
            metrics = ", ".join([m.replace("_mean", "") for m in transition["metrics_affected"]])
            print(f"     {year}: {metrics}")

    return phase_analysis


def create_yearly_trend_visualizations(
    yearly_df: pd.DataFrame, trends_analysis: Dict[str, Any], phase_analysis: Dict[str, Any], output_dir: Path = None
) -> None:
    """
    Create comprehensive yearly trend visualizations.

    Args:
        yearly_df: Yearly aggregated data
        trends_analysis: Trend analysis results
        phase_analysis: Phase analysis results
        output_dir: Directory to save plots
    """
    print("\n=== CREATING YEARLY TREND VISUALIZATIONS ===")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Multi-metric trend overview
    key_metrics = [
        ("ace_rate_mean", "Ace Rate"),
        ("first_serve_win_pct_mean", "First Serve Win %"),
        ("return_win_pct_mean", "Return Win %"),
        ("service_dominance_mean", "Service Dominance"),
    ]

    available_metrics = [(col, name) for col, name in key_metrics if col in yearly_df.columns]

    if available_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, (metric, name) in enumerate(available_metrics[:4]):
            ax = axes[i]

            # Plot data
            # Remove NaN values for plotting
            valid_data = yearly_df[["year", metric]].dropna()
            if len(valid_data) == 0:
                continue

            years_clean = valid_data["year"]
            values_clean = valid_data[metric]

            # Scatter plot with trend line
            ax.scatter(years_clean, values_clean, alpha=0.7, s=50)

            # Add trend line
            if metric in trends_analysis:
                trend = trends_analysis[metric]["overall_trend"]
                slope, r2 = trend["slope"], trend["r_squared"]

                # Linear trend line
                x_trend = np.array([years_clean.min(), years_clean.max()])
                y_trend = slope * (x_trend - years_clean.iloc[0]) + values_clean.iloc[0]
                ax.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2)

                # Add R² annotation
                ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

            # Mark change points
            if metric in trends_analysis:
                for cp in trends_analysis[metric]["change_points"][:2]:  # Show top 2
                    ax.axvline(cp["year"], color="orange", linestyle=":", alpha=0.7)

            ax.set_title(f"{name} Evolution (2005-2024)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / "yearly_trends_overview.png", dpi=300, bbox_inches="tight")
            print("  📊 Saved yearly trends overview")

        plt.show()

    # 2. Game evolution phases
    if phase_analysis["major_transitions"]:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Plot service dominance as main trend
        if "service_dominance_mean" in yearly_df.columns:
            valid_data = yearly_df[["year", "service_dominance_mean"]].dropna()
            ax.plot(valid_data["year"], valid_data["service_dominance_mean"], "o-", linewidth=3, markersize=8, label="Service Dominance")

        # Mark transition years
        transition_years = list(phase_analysis["major_transitions"].keys())
        for year in transition_years:
            ax.axvline(year, color="red", linestyle="--", alpha=0.7, linewidth=2)
            ax.text(year, ax.get_ylim()[1] * 0.95, f"{year}", rotation=90, ha="right", va="top", fontweight="bold")

        # Add phase backgrounds
        phases = phase_analysis["evolution_phases"]
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))

        for i, phase in enumerate(phases):
            ax.axvspan(
                phase["start_year"],
                phase["end_year"],
                alpha=0.2,
                color=colors[i],
                label=f"Phase {phase['phase_id']} ({phase['start_year']}-{phase['end_year']})",
            )

        ax.set_title("Tennis Game Evolution Phases", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Service Dominance")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / "game_evolution_phases.png", dpi=300, bbox_inches="tight")
            print("  📊 Saved game evolution phases")

        plt.show()


def analyze_yearly_evolution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete yearly evolution analysis pipeline.

    Args:
        df: Player-match DataFrame

    Returns:
        Dictionary with complete yearly analysis results
    """
    print("🎾 YEARLY TENNIS EVOLUTION ANALYSIS")
    print("=" * 50)

    # Prepare yearly data
    yearly_df = prepare_yearly_data(df)

    # Analyze yearly trends
    trends_analysis = analyze_yearly_trends(yearly_df)

    # Analyze game evolution phases
    phase_analysis = analyze_game_evolution_phases(yearly_df, trends_analysis)

    # Create visualizations
    plots_dir = OUTPUT_DATA_DIR / "plots" / "yearly_trends"
    create_yearly_trend_visualizations(yearly_df, trends_analysis, phase_analysis, plots_dir)

    # Compile results
    analysis_results = {
        "yearly_data": yearly_df,
        "trends_analysis": trends_analysis,
        "phase_analysis": phase_analysis,
        "summary": {
            "years_analyzed": len(yearly_df),
            "year_range": (yearly_df["year"].min(), yearly_df["year"].max()),
            "metrics_analyzed": len(trends_analysis),
            "major_transitions": len(phase_analysis["major_transitions"]),
            "evolution_phases": len(phase_analysis["evolution_phases"]),
        },
        "output_location": plots_dir,
    }

    # Print summary
    print("\n=== YEARLY ANALYSIS SUMMARY ===")
    print(f"✅ Years analyzed: {yearly_df['year'].min()}-{yearly_df['year'].max()}")
    print(f"✅ Metrics analyzed: {len(trends_analysis)}")
    print(f"✅ Major transitions detected: {len(phase_analysis['major_transitions'])}")
    print(f"✅ Evolution phases identified: {len(phase_analysis['evolution_phases'])}")
    print(f"📊 Visualizations saved to: {plots_dir}")

    # Key findings
    significant_trends = [
        metric
        for metric, data in trends_analysis.items()
        if data["overall_trend"]["p_value"] < 0.05 and data["overall_trend"]["strength"] > 0.5
    ]

    if significant_trends:
        print(f"🔍 KEY FINDINGS: {len(significant_trends)} metrics show strong significant trends")
        for metric in significant_trends[:3]:  # Show top 3
            trend = trends_analysis[metric]["overall_trend"]
            print(f"    {metric.replace('_mean', '')}: {trend['direction']} (R²={trend['r_squared']:.3f})")

    return analysis_results
