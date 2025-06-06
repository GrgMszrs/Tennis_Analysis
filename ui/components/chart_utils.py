"""
Tennis Analysis - Chart Utilities
Plotly-based chart utilities for interactive Streamlit visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Tennis color schemes for Plotly
TENNIS_COLORS = ["#228B22", "#32CD32", "#006400", "#0066CC", "#D2691E", "#E74C3C"]
TENNIS_COLOR_SCALE = [
    [0.0, "#006400"],  # Dark green
    [0.2, "#228B22"],  # Tennis green
    [0.4, "#32CD32"],  # Light green
    [0.6, "#0066CC"],  # Court blue
    [0.8, "#D2691E"],  # Clay orange
    [1.0, "#E74C3C"],  # Accent red
]

# Era-specific colors
ERA_COLORS = {
    "Classic": "#8B4513",  # Brown
    "Transition": "#FF6347",  # Tomato
    "Modern": "#4169E1",  # Royal blue
    "Current": "#228B22",  # Tennis green
}

# Surface-specific colors
SURFACE_COLORS = {
    "Hard": "#2E8B57",  # Sea green
    "Clay": "#D2691E",  # Saddle brown
    "Grass": "#228B22",  # Forest green
    "Carpet": "#800080",  # Purple
}


def get_plotly_tennis_theme():
    """Get tennis-themed layout configuration for Plotly charts."""
    return {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#2C3E50"},
        "title": {"font": {"size": 14, "color": "#006400"}, "x": 0.5, "xanchor": "center"},
        "xaxis": {"gridcolor": "rgba(0,0,0,0.1)", "linecolor": "#2C3E50", "tickfont": {"size": 10}},
        "yaxis": {"gridcolor": "rgba(0,0,0,0.1)", "linecolor": "#2C3E50", "tickfont": {"size": 10}},
        "legend": {"bgcolor": "rgba(255,255,255,0.8)", "bordercolor": "#2C3E50", "borderwidth": 1},
    }


def create_plotly_chart(fig, title=None, caption=None, chart_key=None):
    """
    Display a Plotly figure with consistent tennis styling.

    Args:
        fig: Plotly figure
        title: Optional title for the chart
        caption: Optional caption for the chart
        chart_key: Unique key for the chart to prevent duplicate IDs
    """
    if title:
        st.markdown(f"### {title}")

    # Apply tennis theme
    fig.update_layout(get_plotly_tennis_theme())

    # Display chart natively in Streamlit with unique key
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", key=chart_key)

    if caption:
        st.caption(caption)


def create_plotly_box_plot(data, x_col, y_col, title=None, x_title=None, y_title=None, color_col=None):
    """
    Create a tennis-themed box plot using Plotly.

    Args:
        data: DataFrame
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis (values)
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color_col: Optional column for color grouping

    Returns:
        Plotly figure
    """
    if color_col:
        fig = px.box(data, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=TENNIS_COLORS)
    else:
        fig = px.box(data, x=x_col, y=y_col, title=title, color_discrete_sequence=TENNIS_COLORS)

    # Update layout
    fig.update_layout(xaxis_title=x_title or x_col, yaxis_title=y_title or y_col, **get_plotly_tennis_theme())

    # Add sample size annotations using data max instead of layout range
    if x_col in data.columns:
        y_max = data[y_col].max()
        y_offset = (data[y_col].max() - data[y_col].min()) * 0.05  # 5% offset from top

        for i, category in enumerate(data[x_col].unique()):
            count = len(data[data[x_col] == category])
            fig.add_annotation(x=category, y=y_max + y_offset, text=f"n={count}", showarrow=False, font=dict(size=11, color="#666"))

    return fig


def create_plotly_line_plot(data, x_col, y_col, color_col=None, title=None, x_title=None, y_title=None):
    """
    Create a tennis-themed line plot using Plotly.

    Args:
        data: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Optional column for line colors
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title

    Returns:
        Plotly figure
    """
    if color_col:
        fig = px.line(data, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=TENNIS_COLORS, markers=True)
    else:
        fig = px.line(data, x=x_col, y=y_col, title=title, color_discrete_sequence=TENNIS_COLORS, markers=True)

    # Update layout
    fig.update_layout(xaxis_title=x_title or x_col, yaxis_title=y_title or y_col, **get_plotly_tennis_theme())

    # Enhance line styling
    fig.update_traces(line=dict(width=3), marker=dict(size=8))

    return fig


def create_plotly_heatmap(data, x_col, y_col, z_col, title=None, x_title=None, y_title=None):
    """
    Create a tennis-themed heatmap using Plotly.

    Args:
        data: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        z_col: Column for color values
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title

    Returns:
        Plotly figure
    """
    # Pivot data for heatmap - using pivot_table to handle duplicate combinations
    pivot_data = data.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=TENNIS_COLOR_SCALE,
            showscale=True,
            text=pivot_data.values,
            texttemplate="%{text:.3f}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    # Get base theme and update with specific settings
    layout_config = get_plotly_tennis_theme()
    layout_config.update(
        {
            "xaxis_title": x_title or x_col,
            "yaxis_title": y_title or y_col,
        }
    )

    # Add title if provided
    if title:
        layout_config["title"]["text"] = title

    fig.update_layout(layout_config)

    return fig


def create_plotly_subplot_grid(subplot_specs, titles=None, shared_xaxes=False, shared_yaxes=False):
    """
    Create a grid of subplots using Plotly.

    Args:
        subplot_specs: List of subplot specifications
        titles: List of subplot titles
        shared_xaxes: Whether to share x-axes
        shared_yaxes: Whether to share y-axes

    Returns:
        Plotly figure with subplots
    """
    rows = len(subplot_specs)
    cols = max(len(row) for row in subplot_specs) if subplot_specs else 1

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes)

    # Apply tennis theme
    fig.update_layout(get_plotly_tennis_theme())

    return fig
