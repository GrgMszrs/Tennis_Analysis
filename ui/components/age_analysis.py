"""
Age Analysis Components for Streamlit UI
UI-friendly wrappers for age curves analysis functions.
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from ui.components.chart_utils import create_plotly_box_plot, create_plotly_chart
from ui.components.data_loader import get_player_match_data


@st.cache_data(ttl=1800, show_spinner=False)
def get_career_data(min_matches: int = 20) -> pd.DataFrame:
    """
    UI-friendly wrapper for career data preparation.

    Args:
        min_matches: Minimum matches required per player for inclusion

    Returns:
        Filtered DataFrame suitable for longitudinal analysis
    """
    df = get_player_match_data()

    # Filter for players with sufficient career data
    player_match_counts = df.groupby("player_id").size()
    sufficient_players = player_match_counts[player_match_counts >= min_matches].index

    career_data = df[df["player_id"].isin(sufficient_players)].copy()

    # Create age groupings for visualization stability
    career_data["age_group"] = pd.cut(career_data["age"], bins=range(15, 45, 2), labels=[f"{i}-{i + 1}" for i in range(15, 43, 2)])

    # Focus on prime career ages (18-38)
    career_data = career_data[(career_data["age"] >= 18) & (career_data["age"] <= 38)].copy()

    return career_data


@st.cache_data(ttl=1800, show_spinner=False)
def get_ranking_peaks() -> pd.DataFrame:
    """
    UI-friendly wrapper for peak ranking calculation.

    Returns:
        DataFrame with peak ranking analysis by player and era
    """
    df = get_career_data()

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

    return pd.DataFrame(player_peaks)


def create_peak_age_by_era_plot(peaks_df: pd.DataFrame):
    """
    Create an interactive box plot showing peak ages by era using Plotly.

    Args:
        peaks_df: DataFrame with peak ranking analysis

    Returns:
        None (displays chart directly)
    """
    # Filter out unknown era and ensure we have data
    plot_data = peaks_df[peaks_df["peak_era"] != "Unknown"].copy()

    if len(plot_data) == 0:
        st.error("No data available for peak age analysis")
        return

    # Define era order and filter available eras
    era_order = ["Classic", "Transition", "Modern", "Current"]
    available_eras = [era for era in era_order if era in plot_data["peak_era"].values]

    # Filter and reorder data
    plot_data = plot_data[plot_data["peak_era"].isin(available_eras)].copy()

    # Create interactive box plot using Plotly
    fig = create_plotly_box_plot(
        data=plot_data, x_col="peak_era", y_col="peak_age", title="Peak Ages by Tennis Era", x_title="Era", y_title="Peak Age"
    )

    # Display the interactive chart with unique key
    create_plotly_chart(
        fig,
        chart_key="peak_age_by_era_boxplot",
    )


@st.cache_data(ttl=1800, show_spinner=False)
def get_peak_age_summary(peaks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for peak ages by era.

    Args:
        peaks_df: DataFrame with peak ranking analysis

    Returns:
        Dictionary with summary statistics
    """
    summary = {}

    era_order = ["Classic", "Transition", "Modern", "Current"]

    for era in era_order:
        era_data = peaks_df[peaks_df["peak_era"] == era]
        if len(era_data) > 0:
            summary[era] = {
                "count": len(era_data),
                "mean_age": era_data["peak_age"].mean(),
                "median_age": era_data["peak_age"].median(),
                "std_age": era_data["peak_age"].std(),
                "min_age": era_data["peak_age"].min(),
                "max_age": era_data["peak_age"].max(),
            }

    return summary
