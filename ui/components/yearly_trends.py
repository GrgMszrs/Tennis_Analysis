"""
Tennis Analysis - Yearly Trends Components
UI components for yearly trend analysis and visualization.
"""

import pandas as pd
import streamlit as st

from analysis.yearly_trends import (
    analyze_game_evolution_phases,
    analyze_yearly_evolution,
    analyze_yearly_trends,
    prepare_yearly_data,
)
from ui.components.chart_utils import (
    TENNIS_COLORS,
    create_plotly_chart,
    create_plotly_line_plot,
    get_plotly_tennis_theme,
)
from ui.components.data_loader import get_player_match_data


@st.cache_data(ttl=3600)
def get_yearly_evolution_data():
    """Load and cache yearly evolution data."""
    try:
        df = get_player_match_data()
        return analyze_yearly_evolution(df)
    except Exception as e:
        st.error(f"Error loading yearly evolution data: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def get_yearly_trends_data():
    """Get and cache yearly trends analysis."""
    df = get_player_match_data()
    if df is not None:
        yearly_df = prepare_yearly_data(df)
        trends_analysis = analyze_yearly_trends(yearly_df)
        phase_analysis = analyze_game_evolution_phases(yearly_df, trends_analysis)
        return {"yearly_data": yearly_df, "trends_analysis": trends_analysis, "phase_analysis": phase_analysis}
    return None


def display_yearly_overview():
    """Display yearly trends overview section."""
    st.markdown("## Yearly Trends Overview")

    evolution_data = get_yearly_evolution_data()
    if evolution_data is None:
        st.error("Unable to load yearly evolution data")
        return

    # Display summary metrics
    summary = evolution_data["summary"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        year_range = summary["year_range"]
        st.metric("Years Analyzed", f"{year_range[0]}-{year_range[1]}")

    with col2:
        st.metric("Metrics Tracked", f"{summary['metrics_analyzed']}")

    with col3:
        st.metric("Major Transitions", f"{summary['major_transitions']}")

    with col4:
        st.metric("Evolution Phases", f"{summary['evolution_phases']}")


def display_performance_trends():
    """Display year-over-year performance trends."""
    st.markdown("## Performance Evolution")

    trends_data = get_yearly_trends_data()
    if trends_data is None:
        st.error("Unable to load trends data")
        return

    yearly_df = trends_data["yearly_data"]
    trends_analysis = trends_data["trends_analysis"]

    # Key metrics for visualization
    key_metrics = [
        ("ace_rate_mean", "Ace Rate"),
        ("first_serve_win_pct_mean", "First Serve Win %"),
        ("return_win_pct_mean", "Return Win %"),
        ("service_dominance_mean", "Service Dominance"),
    ]

    # Create tabs for different views
    trend_tab1, trend_tab2 = st.tabs(["Individual Metrics", "Multi-Metric Comparison"])

    with trend_tab1:
        # Select metric to display
        available_metrics = [(col, name) for col, name in key_metrics if col in yearly_df.columns]

        if available_metrics:
            metric_names = [name for _, name in available_metrics]
            selected_metric_name = st.selectbox("Select Performance Metric:", metric_names)

            # Find the corresponding column name
            selected_metric = next(col for col, name in available_metrics if name == selected_metric_name)

            # Prepare data for plotting
            valid_data = yearly_df[["year", selected_metric]].dropna()

            if len(valid_data) > 0:
                # Create line plot
                fig = create_plotly_line_plot(
                    data=valid_data,
                    x_col="year",
                    y_col=selected_metric,
                    title=f"{selected_metric_name} Evolution Over Time",
                    x_title="Year",
                    y_title=selected_metric_name,
                )

                # Add trend information if available
                if selected_metric in trends_analysis:
                    trend_info = trends_analysis[selected_metric]["overall_trend"]
                    direction = trend_info["direction"]
                    r2 = trend_info["r_squared"]

                    # Add trend annotation
                    color = "green" if direction == "increasing" else "red" if direction == "decreasing" else "blue"

                    fig.add_annotation(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"Trend: {direction.title()} (R²={r2:.3f})",
                        showarrow=False,
                        bgcolor=color,
                        bordercolor=color,
                        font=dict(color="white"),
                        opacity=0.8,
                    )

                    # Add change points as vertical lines
                    change_points = trends_analysis[selected_metric]["change_points"]
                    for cp in change_points[:2]:  # Show top 2 change points
                        fig.add_vline(
                            x=cp["year"], line_dash="dot", line_color="orange", opacity=0.7, annotation_text=f"Change {cp['year']}"
                        )

                # Display chart
                create_plotly_chart(
                    fig,
                    chart_key=f"yearly_trend_{selected_metric}",
                )

    with trend_tab2:
        # Multi-metric comparison
        st.markdown("### Normalized Multi-Metric Comparison")

        available_metrics = [(col, name) for col, name in key_metrics if col in yearly_df.columns]

        if len(available_metrics) >= 2:
            # Create normalized comparison chart
            fig = create_multi_metric_comparison_chart(yearly_df, available_metrics, trends_analysis)

            create_plotly_chart(
                fig,
                chart_key="multi_metric_comparison",
            )
        else:
            st.info("Need at least 2 metrics for comparison view.")


def create_multi_metric_comparison_chart(yearly_df, available_metrics, trends_analysis):
    """
    Create a multi-metric comparison chart with normalized values.

    Args:
        yearly_df: DataFrame with yearly data
        available_metrics: List of (column, name) tuples for available metrics
        trends_analysis: Dictionary with trend analysis results

    Returns:
        Plotly figure with multi-metric comparison
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Color cycle for different metrics
    colors = TENNIS_COLORS

    for i, (metric_col, metric_name) in enumerate(available_metrics):
        # Get valid data
        valid_data = yearly_df[["year", metric_col]].dropna()

        if len(valid_data) > 0:
            # Normalize the values (z-score normalization)
            values = valid_data[metric_col]
            normalized_values = (values - values.mean()) / values.std()

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=valid_data["year"],
                    y=normalized_values,
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{metric_name}</b><br>"
                    + "Year: %{x}<br>"
                    + "Normalized Value: %{y:.2f}<br>"
                    + "Original Value: %{customdata:.3f}<extra></extra>",
                    customdata=values,
                )
            )

    # Update layout with tennis theme
    layout_config = get_plotly_tennis_theme()
    layout_config.update(
        {
            "title": {"text": "Multi-Metric Performance Evolution (Normalized)"},
            "xaxis_title": "Year",
            "yaxis_title": "Normalized Performance (Z-Score)",
            "hovermode": "x unified",
            "showlegend": True,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        }
    )

    fig.update_layout(layout_config)

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def display_trend_summary():
    """Display trends summary table."""
    st.markdown("## Trend Summary")

    trends_data = get_yearly_trends_data()
    if trends_data is None:
        st.error("Unable to load trends data")
        return

    trends_analysis = trends_data["trends_analysis"]

    # Create trends summary table
    trend_summary = []
    for metric, trend_data in trends_analysis.items():
        overall_trend = trend_data["overall_trend"]
        recent_trend = trend_data["recent_trend"]

        trend_summary.append(
            {
                "Metric": metric.replace("_mean", "").replace("_", " ").title(),
                "Overall Trend": overall_trend["direction"].title(),
                "Trend Strength (R²)": f"{overall_trend['r_squared']:.3f}",
                "P-value": f"{overall_trend['p_value']:.3f}",
                "Recent Trend (R²)": f"{recent_trend['r_squared']:.3f}",
                "Total Change": f"{trend_data['total_change']:.3f}",
            }
        )

    if trend_summary:
        trend_df = pd.DataFrame(trend_summary)
        st.dataframe(trend_df, use_container_width=True)


def display_evolution_phases():
    """Display game evolution phases analysis."""
    st.markdown("## Game Evolution Phases")

    trends_data = get_yearly_trends_data()
    if trends_data is None:
        st.error("Unable to load phase data")
        return

    phase_analysis = trends_data["phase_analysis"]

    # Show major transitions
    if phase_analysis["major_transitions"]:
        st.markdown("### Major Transition Years")

        for year, transition in phase_analysis["major_transitions"].items():
            with st.expander(f"{year} - {len(transition['metrics_affected'])} metrics changed"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Affected Metrics:**")
                    for metric in transition["metrics_affected"]:
                        st.markdown(f"• {metric.replace('_mean', '').replace('_', ' ').title()}")

                with col2:
                    st.metric("Impact Score", f"{transition['average_impact']:.3f}")

    # Show evolution phases
    st.markdown("### Evolution Phases")

    phases = phase_analysis["evolution_phases"]
    if phases:
        phase_data = []
        for phase in phases:
            characteristics = ", ".join([c.replace("_", " ").title() for c in phase["characteristics"]])
            phase_data.append(
                {
                    "Phase": f"Phase {phase['phase_id']}",
                    "Period": f"{phase['start_year']}-{phase['end_year']}",
                    "Duration": f"{phase['duration']} years",
                    "Key Changes": characteristics[:50] + "..." if len(characteristics) > 50 else characteristics,
                }
            )

        phase_df = pd.DataFrame(phase_data)
        st.dataframe(phase_df, use_container_width=True)
    else:
        st.info("No distinct evolution phases detected in the data.")
