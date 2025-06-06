"""
Tennis Analysis - Date Analysis Components
UI components for date-based analysis and temporal pattern visualization.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components.chart_utils import (
    SURFACE_COLORS,
    TENNIS_COLORS,
    get_plotly_tennis_theme,
)
from ui.components.data_loader import get_player_match_data


@st.cache_data(ttl=3600)
def get_date_analysis_data():
    """Load and prepare data for date analysis."""
    try:
        df = get_player_match_data()

        # Convert dates with better error handling
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

        # Remove invalid dates
        initial_count = len(df)
        df = df.dropna(subset=["tourney_date"])
        if len(df) < initial_count:
            st.warning(f"Removed {initial_count - len(df)} records with invalid dates")

        # Create time periods efficiently
        dates = df["tourney_date"]
        df["year"] = dates.dt.year
        df["month"] = dates.dt.month
        df["month_name"] = dates.dt.month_name()
        df["quarter"] = dates.dt.quarter
        df["day_of_year"] = dates.dt.dayofyear

        # Use isocalendar() more safely
        try:
            df["week_of_year"] = dates.dt.isocalendar().week
        except AttributeError:
            df["week_of_year"] = dates.dt.week

        # Create month period more safely
        try:
            df["month_period"] = dates.dt.to_period("M")
        except (AttributeError, ValueError):
            # Fallback to year-month string
            df["month_period"] = dates.dt.strftime("%Y-%m")

        return df

    except Exception as e:
        st.error(f"Error loading date analysis data: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return None


@st.cache_data(ttl=3600)
def get_temporal_statistics(df):
    """Calculate temporal pattern statistics."""
    try:
        if df is None or len(df) == 0:
            return {}

        stats = {
            "total_matches": len(df),
            "date_range": (df["tourney_date"].min(), df["tourney_date"].max()),
            "years_covered": df["year"].nunique(),
            "avg_matches_per_year": len(df) / max(df["year"].nunique(), 1),
            "avg_matches_per_month": len(df) / max(df["year"].nunique() * 12, 1),
        }

        # Calculate patterns with error handling
        try:
            yearly_counts = df.groupby("year").size()
            stats["busiest_year"] = yearly_counts.idxmax()
            stats["busiest_year_count"] = yearly_counts.max()
            stats["quietest_year"] = yearly_counts.idxmin()
            stats["quietest_year_count"] = yearly_counts.min()
        except (ValueError, KeyError):
            stats.update({"busiest_year": "Unknown", "busiest_year_count": 0, "quietest_year": "Unknown", "quietest_year_count": 0})

        try:
            monthly_counts = df.groupby("month_name").size()
            stats["busiest_month"] = monthly_counts.idxmax()
            stats["busiest_month_count"] = monthly_counts.max()
            stats["quietest_month"] = monthly_counts.idxmin()
            stats["quietest_month_count"] = monthly_counts.min()
            stats["seasonal_variation"] = ((monthly_counts.max() / max(monthly_counts.min(), 1)) - 1) * 100
        except (ValueError, KeyError):
            stats.update(
                {
                    "busiest_month": "Unknown",
                    "busiest_month_count": 0,
                    "quietest_month": "Unknown",
                    "quietest_month_count": 0,
                    "seasonal_variation": 0,
                }
            )

        try:
            stats["surface_distribution"] = df["surface"].value_counts().to_dict()
        except (KeyError, ValueError):
            stats["surface_distribution"] = {}

        return stats

    except Exception as e:
        st.error(f"Error calculating temporal statistics: {str(e)}")
        return {}


def create_annual_matches_chart(df):
    """Create annual match counts chart."""
    yearly_counts = df.groupby("year").size().reset_index()
    yearly_counts.columns = ["Year", "Matches"]

    fig = px.bar(
        yearly_counts,
        x="Year",
        y="Matches",
        title="Annual Match Counts",
        color_discrete_sequence=[TENNIS_COLORS[0]],
    )

    fig.update_layout(
        **get_plotly_tennis_theme(),
        xaxis_title="Year",
        yaxis_title="Number of Matches",
        showlegend=False,
    )

    # Add trend line
    fig.add_scatter(
        x=yearly_counts["Year"],
        y=yearly_counts["Matches"],
        mode="lines",
        name="Trend",
        line=dict(color="red", width=2, dash="dot"),
        opacity=0.7,
    )

    return fig


def create_monthly_timeline_chart(df):
    """Create monthly timeline chart."""
    try:
        # Smart sampling for chart performance while maintaining data integrity
        if len(df) > 200000:
            df_chart = df.sample(n=100000, random_state=42)
        else:
            df_chart = df.copy()

        # Create proper datetime for better x-axis labeling
        df_chart["year_month_date"] = pd.to_datetime(
            df_chart["year"].astype(str) + "-" + df_chart["month"].astype(str).str.zfill(2) + "-01"
        )

        monthly_counts = df_chart.groupby("year_month_date").size().reset_index()
        monthly_counts.columns = ["Month_Period", "Matches"]

        # Sort by date
        monthly_counts = monthly_counts.sort_values("Month_Period")

        fig = px.line(
            monthly_counts,
            x="Month_Period",
            y="Matches",
            title="Monthly Match Distribution Over Time",
            markers=True,
            color_discrete_sequence=[TENNIS_COLORS[1]],
        )

        # Get theme and update layout safely
        theme_config = get_plotly_tennis_theme()
        theme_config.update(
            {
                "xaxis_title": "Date",
                "yaxis_title": "Number of Matches",
                "showlegend": False,
            }
        )

        fig.update_layout(theme_config)

        # Better x-axis formatting
        fig.update_xaxes(
            tickformat="%Y-%m",
            tickmode="auto",
            nticks=10,  # Limit number of ticks for readability
        )

        return fig

    except Exception as e:
        # Return a simple error chart
        fig = px.scatter(x=[1], y=[1], title="Error creating monthly timeline")
        fig.add_annotation(text=f"Error: {str(e)}", x=1, y=1)
        return fig


def create_seasonal_patterns_chart(df):
    """Create seasonal patterns chart."""
    try:
        # Define proper month order
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        monthly_counts = df.groupby("month_name").size().reindex(month_order).reset_index()
        monthly_counts.columns = ["Month", "Matches"]

        # Create a gradient of colors for the months
        seasonal_colors = [
            "#4169E1",
            "#6495ED",
            "#87CEEB",  # Winter blues
            "#228B22",
            "#32CD32",
            "#90EE90",  # Spring greens
            "#FF6347",
            "#FF7F50",
            "#FFA500",  # Summer oranges/reds
            "#D2691E",
            "#CD853F",
            "#DEB887",  # Autumn browns
        ]

        fig = px.bar(
            monthly_counts,
            x="Month",
            y="Matches",
            title="Seasonal Distribution (Total Matches by Month)",
            color="Month",
            color_discrete_sequence=seasonal_colors,
        )

        # Get theme and update layout safely
        theme_config = get_plotly_tennis_theme()
        theme_config.update(
            {
                "xaxis_title": "Month",
                "yaxis_title": "Total Matches",
                "showlegend": False,
            }
        )

        fig.update_layout(theme_config)
        fig.update_xaxes(tickangle=45)

        return fig

    except Exception as e:
        # Return error chart
        fig = px.scatter(x=[1], y=[1], title="Error creating seasonal chart")
        fig.add_annotation(text=f"Error: {str(e)}", x=1, y=1)
        return fig


def create_surface_timeline_chart(df):
    """Create surface distribution over time chart."""
    try:
        surface_yearly = df.groupby(["year", "surface"]).size().unstack(fill_value=0)

        if surface_yearly.empty:
            fig = go.Figure()
            fig.add_annotation(text="No surface data available", x=0.5, y=0.5)
            return fig

        # Create stacked area chart
        fig = go.Figure()

        # Use more varied colors for better visualization
        chart_colors = [
            "#4682B4",  # Steel blue for Hard (updated to be more blueish)
            "#D2691E",  # Orange for Clay
            "#228B22",  # Forest green for Grass
            "#800080",  # Purple for Carpet
            "#FF6347",  # Tomato
            "#4169E1",  # Royal blue
            "#32CD32",  # Lime green
            "#FF1493",  # Deep pink
        ]

        for i, surface in enumerate(surface_yearly.columns):
            if surface in SURFACE_COLORS:
                color = SURFACE_COLORS[surface]
            else:
                color = chart_colors[i % len(chart_colors)]

            fig.add_trace(
                go.Scatter(
                    x=surface_yearly.index,
                    y=surface_yearly[surface],
                    mode="lines",
                    name=surface,
                    stackgroup="one",
                    line=dict(width=0.5),
                    fillcolor=color,
                )
            )

        # Get theme and update layout safely
        theme_config = get_plotly_tennis_theme()
        theme_config.update(
            {
                "title": {"text": "Surface Distribution Over Time"},
                "xaxis_title": "Year",
                "yaxis_title": "Number of Matches",
                "hovermode": "x unified",
            }
        )

        fig.update_layout(theme_config)

        return fig

    except Exception as e:
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating surface timeline: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title="Error in Surface Timeline")
        return fig


def create_quarterly_heatmap(df):
    """Create quarterly distribution heatmap."""
    try:
        # Smart sampling for chart performance
        if len(df) > 150000:
            df_chart = df.sample(n=75000, random_state=42)
        else:
            df_chart = df.copy()

        quarterly_data = df_chart.groupby(["year", "quarter"]).size().unstack(fill_value=0)

        # Ensure we have valid data
        if quarterly_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No quarterly data available", x=0.5, y=0.5)
            return fig

        fig = go.Figure(
            data=go.Heatmap(
                z=quarterly_data.values,
                x=[f"Q{i}" for i in quarterly_data.columns],
                y=quarterly_data.index,
                colorscale="Viridis",
                showscale=True,
                text=quarterly_data.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        # Get theme and update layout safely
        theme_config = get_plotly_tennis_theme()
        theme_config.update(
            {
                "title": {"text": "Quarterly Match Distribution Heatmap"},
                "xaxis_title": "Quarter",
                "yaxis_title": "Year",
            }
        )

        fig.update_layout(theme_config)

        return fig

    except Exception as e:
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating quarterly heatmap: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title="Error in Quarterly Heatmap")
        return fig


def create_weekly_patterns_chart(df):
    """Create weekly patterns throughout the year."""
    weekly_counts = df.groupby("week_of_year").size().reset_index()
    weekly_counts.columns = ["Week", "Matches"]

    fig = px.line(
        weekly_counts,
        x="Week",
        y="Matches",
        title="Weekly Distribution Throughout the Year",
        markers=True,
        color_discrete_sequence=[TENNIS_COLORS[3]],
    )

    fig.update_layout(
        **get_plotly_tennis_theme(),
        xaxis_title="Week of Year",
        yaxis_title="Average Matches per Week",
        showlegend=False,
    )

    # Add season markers
    season_markers = [(1, "Winter"), (13, "Spring"), (26, "Summer"), (39, "Autumn")]

    for week, season in season_markers:
        fig.add_vline(
            x=week,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=season,
            annotation_position="top",
        )

    return fig


def create_match_intensity_chart(df):
    """Create match intensity chart showing matches per day distribution."""
    # Smart sampling for chart performance
    if len(df) > 100000:
        df_chart = df.sample(n=50000, random_state=42)
    else:
        df_chart = df.copy()

    df_chart["date_only"] = df_chart["tourney_date"].dt.date
    daily_counts = df_chart.groupby("date_only").size()

    # Create histogram of daily match counts
    fig = px.histogram(
        x=daily_counts.values,
        title="Match Intensity Distribution (Matches per Day)",
        labels={"x": "Matches per Day", "y": "Number of Days"},
        color_discrete_sequence=[TENNIS_COLORS[4]],
        nbins=30,
    )

    fig.update_layout(
        **get_plotly_tennis_theme(),
        showlegend=False,
    )

    # Add statistics annotation
    mean_matches = daily_counts.mean()
    max_matches = daily_counts.max()

    fig.add_annotation(
        x=0.7,
        y=0.9,
        xref="paper",
        yref="paper",
        text=f"Avg: {mean_matches:.1f} matches/day<br>Max: {max_matches} matches/day",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    return fig


def create_era_timeline_chart(df):
    """Create era distribution timeline."""
    try:
        if "era" not in df.columns:
            return None

        era_yearly = df.groupby(["year", "era"]).size().unstack(fill_value=0)

        if era_yearly.empty:
            fig = go.Figure()
            fig.add_annotation(text="No era data available", x=0.5, y=0.5)
            return fig

        # Create stacked bar chart
        fig = go.Figure()

        # More distinct era colors
        era_colors = {
            "Classic": "#8B4513",  # Saddle brown
            "Transition": "#FF6347",  # Tomato
            "Modern": "#4169E1",  # Royal blue
            "Current": "#228B22",  # Forest green
            "Unknown": "#708090",  # Slate gray
        }

        for era in era_yearly.columns:
            color = era_colors.get(era, TENNIS_COLORS[0])
            fig.add_trace(
                go.Bar(
                    x=era_yearly.index,
                    y=era_yearly[era],
                    name=era,
                    marker_color=color,
                )
            )

        # Get theme and update layout safely
        theme_config = get_plotly_tennis_theme()
        theme_config.update(
            {
                "title": {"text": "Era Distribution Over Time"},
                "xaxis_title": "Year",
                "yaxis_title": "Number of Matches",
                "barmode": "stack",
            }
        )

        fig.update_layout(theme_config)

        return fig

    except Exception as e:
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating era timeline: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title="Error in Era Timeline")
        return fig
