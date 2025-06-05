"""
Tennis Era Analysis - Era Analysis Components
UI components for era-based analysis and visualization.
"""

import pandas as pd
import streamlit as st

from analysis.era_analysis import (
    analyze_era_trends,
    compare_surface_performance,
    compute_era_statistics,
    identify_era_champions,
    load_player_match_data,
)
from ui.components.chart_utils import (
    create_plotly_chart,
    create_plotly_heatmap,
    create_plotly_line_plot,
)


@st.cache_data(ttl=3600)
def get_era_data():
    """Load and cache era analysis data."""
    try:
        df = load_player_match_data()
        return df
    except Exception as e:
        st.error(f"Error loading era data: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def get_era_statistics():
    """Get and cache era statistics."""
    df = get_era_data()
    if df is not None:
        return compute_era_statistics(df)
    return None


@st.cache_data(ttl=3600)
def get_era_trends():
    """Get and cache era trend analysis."""
    df = get_era_data()
    if df is not None:
        return analyze_era_trends(df)
    return None


@st.cache_data(ttl=3600)
def get_surface_comparison():
    """Get and cache surface comparison data."""
    df = get_era_data()
    if df is not None:
        return compare_surface_performance(df)
    return None


@st.cache_data(ttl=3600)
def get_era_champions(top_n: int = 10):
    """Get and cache era champions data."""
    df = get_era_data()
    if df is not None:
        return identify_era_champions(df, top_n)
    return None


def display_era_overview():
    """Display era overview section."""
    st.markdown("## 🏟️ Era Overview")

    era_stats = get_era_statistics()
    if era_stats is None:
        st.error("Unable to load era statistics")
        return

    # Display era comparison table
    st.markdown("### 📊 Era Statistics Comparison")

    # Select key metrics to display
    display_cols = [
        "era",
        "matches",
        "players",
        "ace_rate_mean",
        "first_serve_win_pct_mean",
        "break_point_save_pct_mean",
        "service_dominance_mean",
    ]

    if all(col in era_stats.columns for col in display_cols):
        display_df = era_stats[display_cols].copy()

        # Round numeric columns
        numeric_cols = display_df.select_dtypes(include=["float64"]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(3)

        # Rename columns for better display
        display_df.columns = ["Era", "Matches", "Players", "Ace Rate", "First Serve Win%", "Break Point Save%", "Service Dominance"]

        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(era_stats, use_container_width=True)

    # Key insights
    st.markdown("### 🔍 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_matches = era_stats["matches"].sum()
        st.metric("Total Matches Analyzed", f"{total_matches:,}")

    with col2:
        total_players = era_stats["players"].sum()
        st.metric("Total Players", f"{total_players:,}")

    with col3:
        era_span = len(era_stats)
        st.metric("Tennis Eras", f"{era_span}")


def display_era_trends():
    """Display era trends analysis."""
    st.markdown("## 📈 Era Trends Analysis")

    trends_data = get_era_trends()
    if trends_data is None:
        st.error("Unable to load era trends data")
        return

    era_means = trends_data["era_means"]
    trends = trends_data["trends"]

    # Create trend visualization
    st.markdown("### 📊 Performance Metrics Evolution")

    # Select metrics to visualize
    metrics_to_plot = ["ace_rate", "first_serve_win_pct", "break_point_save_pct", "service_dominance"]
    available_metrics = [m for m in metrics_to_plot if m in era_means.columns]

    if available_metrics:
        # Prepare data for plotting
        plot_data = []
        era_order = trends_data["era_order"]

        for metric in available_metrics:
            for i, era in enumerate(era_order):
                if era in era_means.index:
                    plot_data.append(
                        {
                            "Era": era,
                            "Era_Order": i,
                            "Metric": metric.replace("_", " ").title(),
                            "Value": era_means.loc[era, metric],
                            "Metric_Type": metric,
                        }
                    )

        plot_df = pd.DataFrame(plot_data)

        # Create subplot for each metric
        metrics_display = ["Ace Rate", "First Serve Win Pct", "Break Point Save Pct", "Service Dominance"]
        available_display = [m for m in metrics_display if m in plot_df["Metric"].values]

        # Create individual charts for each metric
        for i, metric_display in enumerate(available_display):
            metric_data = plot_df[plot_df["Metric"] == metric_display].copy()

            if len(metric_data) > 0:
                # Create line plot
                fig = create_plotly_line_plot(
                    data=metric_data,
                    x_col="Era",
                    y_col="Value",
                    title=f"{metric_display} Evolution Across Eras",
                    x_title="Era",
                    y_title=metric_display,
                )

                # Add trend information if available
                metric_key = metric_display.lower().replace(" ", "_")
                if metric_key in trends:
                    trend_info = trends[metric_key]
                    direction = trend_info["direction"]
                    color = "green" if direction == "increasing" else "red" if direction == "decreasing" else "blue"

                    fig.add_annotation(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"Trend: {direction.title()}",
                        showarrow=False,
                        bgcolor=color,
                        bordercolor=color,
                        font=dict(color="white"),
                        opacity=0.8,
                    )

                # Display chart with unique key
                create_plotly_chart(
                    fig,
                    chart_key=f"era_trend_{metric_key}_{i}",
                    caption=f"Interactive chart showing {metric_display.lower()} evolution. Hover for exact values, zoom to focus on specific eras.",
                )

    # Trend summary
    st.markdown("### 🎯 Trend Summary")

    trend_summary = []
    for metric, trend_data in trends.items():
        trend_summary.append(
            {
                "Metric": metric.replace("_", " ").title(),
                "Trend Direction": trend_data["direction"].title(),
                "Trend Strength": abs(trend_data["slope"]),
            }
        )

    if trend_summary:
        trend_df = pd.DataFrame(trend_summary)
        st.dataframe(trend_df, use_container_width=True)


def display_surface_analysis():
    """Display surface-specific analysis."""
    st.markdown("## 🏟️ Surface-Specific Analysis")

    surface_data = get_surface_comparison()
    if surface_data is None:
        st.error("Unable to load surface comparison data")
        return

    st.markdown("### 🎾 Performance by Surface and Era")

    # Filter out invalid surface values (NAN, None, etc.)
    surface_data = surface_data[surface_data["surface"].notna()]
    surface_data = surface_data[surface_data["surface"] != "NAN"]

    # The surface_data is already a flattened DataFrame with columns like 'ace_rate_mean', 'ace_rate_count', etc.
    # Let's identify available metrics
    available_metrics = []
    for col in surface_data.columns:
        if col.endswith("_mean") and not col.startswith(("era", "surface")):
            # Extract the base metric name (e.g., 'ace_rate' from 'ace_rate_mean')
            base_metric = col.replace("_mean", "")
            available_metrics.append(base_metric)

    if available_metrics:
        # Select metric for visualization
        selected_metric = st.selectbox("Select Performance Metric:", available_metrics, format_func=lambda x: x.replace("_", " ").title())

        # Prepare data for heatmap using the correct column name
        mean_col = f"{selected_metric}_mean"

        if mean_col in surface_data.columns:
            # Create the plot data directly from the surface_data DataFrame
            plot_data = surface_data[["era", "surface", mean_col]].copy()
            plot_data = plot_data.rename(columns={mean_col: "Value"})

            # Remove any rows with missing values
            plot_data = plot_data.dropna()

            if len(plot_data) > 0:
                # Create interactive heatmap
                fig = create_plotly_heatmap(
                    data=plot_data,
                    x_col="surface",
                    y_col="era",
                    z_col="Value",
                    title=f"{selected_metric.replace('_', ' ').title()} by Era and Surface",
                    x_title="Court Surface",
                    y_title="Tennis Era",
                )

                # Display chart with unique key
                create_plotly_chart(
                    fig,
                    chart_key=f"surface_heatmap_{selected_metric}",
                    caption="Interactive heatmap showing performance variations. Hover for exact values, darker colors indicate higher performance.",
                )
            else:
                st.warning(f"No data available for {selected_metric} after removing missing values")
        else:
            st.error(f"Column {mean_col} not found in surface data")
    else:
        st.warning("No metrics with '_mean' suffix found in the surface data")

    # Surface statistics table
    st.markdown("### 📋 Detailed Surface Statistics")

    # Display the surface data table
    if not surface_data.empty:
        st.dataframe(surface_data, use_container_width=True)


def display_era_champions():
    """Display era champions analysis."""
    st.markdown("## 🏆 Era Champions")

    champions_data = get_era_champions()
    if champions_data is None:
        st.error("Unable to load era champions data")
        return

    # Select era to display
    era_options = list(champions_data.keys())
    selected_era = st.selectbox("Select Era:", era_options)

    if selected_era in champions_data:
        champions_df = champions_data[selected_era]

        st.markdown(f"### 🥇 Top Performers - {selected_era} Era")

        # Display top performers
        if not champions_df.empty:
            # Select key columns for display
            display_cols = [
                "player_name",
                "matches",
                "ace_rate_mean",
                "first_serve_win_pct_mean",
                "break_point_save_pct_mean",
                "service_dominance_mean",
            ]

            available_cols = [col for col in display_cols if col in champions_df.columns]

            if available_cols:
                display_df = champions_df[available_cols].head(10).copy()

                # Round numeric columns
                numeric_cols = display_df.select_dtypes(include=["float64"]).columns
                display_df[numeric_cols] = display_df[numeric_cols].round(3)

                # Rename columns for better display
                rename_dict = {
                    "player_name": "Player",
                    "matches": "Matches",
                    "ace_rate_mean": "Ace Rate",
                    "first_serve_win_pct_mean": "First Serve Win%",
                    "break_point_save_pct_mean": "Break Point Save%",
                    "service_dominance_mean": "Service Dominance",
                }

                display_df = display_df.rename(columns=rename_dict)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(champions_df.head(10), use_container_width=True)
        else:
            st.info(f"No champion data available for {selected_era} era")
