"""
Age Curves Analysis Page
Interactive visualization of player career trajectories and peak age analysis.
"""

import pandas as pd
import streamlit as st

from ui.components.age_analysis import create_peak_age_by_era_plot, get_peak_age_summary, get_ranking_peaks

# Import data loading and analysis


def render_age_curves_page():
    """Render the age curves analysis page."""

    st.title("Age Curves Analysis")
    st.markdown("### Player Career Trajectories and Peak Performance Analysis")

    # Load data with progress indicator
    with st.spinner("Loading career and peak age data..."):
        try:
            peaks_df = get_ranking_peaks()

            if len(peaks_df) == 0:
                st.error("No peak ranking data available. Check data pipeline configuration.")
                return

        except Exception as e:
            st.error(f"Error loading peak age data: {str(e)}")
            return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Players Analyzed", f"{len(peaks_df):,}")

    with col2:
        overall_mean = peaks_df["peak_age"].mean()
        st.metric("Mean Peak Age", f"{overall_mean:.1f}")

    with col3:
        overall_median = peaks_df["peak_age"].median()
        st.metric("Median Peak Age", f"{overall_median:.1f}")

    with col4:
        age_range = peaks_df["peak_age"].max() - peaks_df["peak_age"].min()
        st.metric("Peak Age Range", f"{age_range:.1f} years")

    # Era comparison visualization
    st.subheader("Peak Ages by Era")

    # Create and display the interactive Plotly chart
    create_peak_age_by_era_plot(peaks_df)

    # Era statistics table
    st.subheader("Statistical Summary by Era")

    summary_stats = get_peak_age_summary(peaks_df)

    if summary_stats:
        # Create a summary table
        summary_data = []
        for era, stats in summary_stats.items():
            summary_data.append(
                {
                    "Era": era,
                    "Sample Size": stats["count"],
                    "Mean Age": f"{stats['mean_age']:.1f}",
                    "Median Age": f"{stats['median_age']:.1f}",
                    "Standard Deviation": f"{stats['std_age']:.2f}",
                    "Age Range": f"{stats['min_age']:.0f}-{stats['max_age']:.0f}",
                }
            )

        summary_table = pd.DataFrame(summary_data)
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    # Interactive exploration section
    st.subheader("Data Exploration")

    # Player filter
    era_filter = st.selectbox("Filter by Era:", ["All Eras"] + sorted(peaks_df["peak_era"].unique()))

    # Apply filter
    if era_filter != "All Eras":
        filtered_peaks = peaks_df[peaks_df["peak_era"] == era_filter]
    else:
        filtered_peaks = peaks_df

    # Show filtered results
    with st.expander(f"Player Details ({len(filtered_peaks)} players)"):
        if len(filtered_peaks) > 0:
            # Sort by peak rank (best first)
            display_df = filtered_peaks.sort_values("peak_rank").copy()

            # Format for display
            display_df["Peak Rank"] = display_df["peak_rank"].astype(int)
            display_df["Peak Age"] = display_df["peak_age"].round(1)
            display_df["Career Span"] = display_df["career_span"].round(1)

            # Select columns for display
            display_columns = ["player_name", "Peak Rank", "Peak Age", "peak_era", "peak_year", "Career Span", "total_matches"]

            # Rename columns for display
            column_mapping = {"player_name": "Player", "peak_era": "Era", "peak_year": "Peak Year", "total_matches": "Total Matches"}

            display_table = display_df[display_columns].rename(columns=column_mapping)

            st.dataframe(display_table, use_container_width=True, hide_index=True)
        else:
            st.info("No players found for selected criteria.")

    # Methodological notes
    st.markdown("---")
    st.markdown(
        """
    <div class="highlight-section">
        <h6>Methodology</h6>
        <p><strong>Peak Definition:</strong> Lowest (best) ranking achieved during career<br>
        <strong>Age Calculation:</strong> Player age at time of peak ranking achievement<br>
        <strong>Minimum Threshold:</strong> 20+ matches required for inclusion<br>
        <strong>Era Classification:</strong> Based on temporal and stylistic tennis evolution patterns</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_age_curves_page()
