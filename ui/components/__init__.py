"""
UI Components for Tennis Analysis
Reusable components for data loading, filtering, and visualization.
Plotly-based interactive chart utilities for native Streamlit rendering.
"""

from .age_analysis import (
    create_peak_age_by_era_plot,
    get_career_data,
    get_peak_age_summary,
    get_ranking_peaks,
)
from .chart_utils import (
    ERA_COLORS,
    SURFACE_COLORS,
    TENNIS_COLOR_SCALE,
    # Color schemes
    TENNIS_COLORS,
    create_plotly_box_plot,
    # Plotly chart utilities
    create_plotly_chart,
    create_plotly_heatmap,
    create_plotly_line_plot,
    create_plotly_subplot_grid,
    get_plotly_tennis_theme,
)
from .data_loader import get_data_summary
from .date_analysis import (
    create_annual_matches_chart,
    create_era_timeline_chart,
    create_match_intensity_chart,
    create_monthly_timeline_chart,
    create_quarterly_heatmap,
    create_seasonal_patterns_chart,
    create_surface_timeline_chart,
    create_weekly_patterns_chart,
    get_date_analysis_data,
    get_temporal_statistics,
)
from .era_analysis import (
    display_era_champions,
    display_era_overview,
    display_era_trends,
    display_surface_analysis,
    get_era_champions,
    get_era_data,
    get_era_statistics,
    get_era_trends,
    get_surface_comparison,
)
from .yearly_trends import (
    display_evolution_phases,
    display_performance_trends,
    display_trend_summary,
    display_yearly_overview,
    get_yearly_evolution_data,
    get_yearly_trends_data,
)

__all__ = [
    # Age analysis
    "get_career_data",
    "get_ranking_peaks",
    "create_peak_age_by_era_plot",
    "get_peak_age_summary",
    # Chart utilities
    "ERA_COLORS",
    "SURFACE_COLORS",
    "TENNIS_COLOR_SCALE",
    "TENNIS_COLORS",
    "create_plotly_box_plot",
    "create_plotly_chart",
    "create_plotly_heatmap",
    "create_plotly_line_plot",
    "create_plotly_subplot_grid",
    "get_plotly_tennis_theme",
    # Data loader
    "get_data_summary",
    # Era analysis
    "display_era_champions",
    "display_era_overview",
    "display_era_trends",
    "display_surface_analysis",
    "get_era_champions",
    "get_era_data",
    "get_era_statistics",
    "get_era_trends",
    "get_surface_comparison",
    # Yearly trends
    "display_evolution_phases",
    "display_performance_trends",
    "display_trend_summary",
    "display_yearly_overview",
    "get_yearly_evolution_data",
    "get_yearly_trends_data",
    # Date analysis
    "create_annual_matches_chart",
    "create_era_timeline_chart",
    "create_match_intensity_chart",
    "create_monthly_timeline_chart",
    "create_quarterly_heatmap",
    "create_seasonal_patterns_chart",
    "create_surface_timeline_chart",
    "create_weekly_patterns_chart",
    "get_date_analysis_data",
    "get_temporal_statistics",
]
