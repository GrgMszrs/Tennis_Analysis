"""
Data Loading Components for Streamlit UI
Cached data loading functions that wrap the existing analysis modules.
"""

import pandas as pd
import streamlit as st

# Import existing data loading functions
from analysis.era_analysis import load_player_match_data


@st.cache_data(ttl=3600, show_spinner=False)
def get_player_match_data() -> pd.DataFrame:
    """
    Load player-match data with caching.

    Returns:
        Player-match DataFrame
    """
    return load_player_match_data()


@st.cache_data(ttl=3600, show_spinner=False)
def get_data_summary() -> dict:
    """
    Get basic summary statistics about the dataset.

    Returns:
        Dictionary with summary statistics
    """
    try:
        df = get_player_match_data()

        summary = {
            "total_rows": len(df),
            "unique_players": df["player_name"].nunique(),
            "unique_matches": df["match_id"].nunique(),
            "date_range": (df["tourney_date"].min(), df["tourney_date"].max()),
            "eras": sorted(df["era"].unique()),
            "surfaces": sorted(df["surface"].unique()),
            "years": (int(df["year"].min()), int(df["year"].max())),
        }

        # Check for enhanced features
        z_score_cols = [col for col in df.columns if "_z_" in col]
        ranking_cols = [col for col in df.columns if "historical" in col]

        summary["enhanced_features"] = {"z_score_metrics": len(z_score_cols), "ranking_metrics": len(ranking_cols)}

        return summary

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}
