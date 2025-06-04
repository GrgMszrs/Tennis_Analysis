"""
Tennis Era Analysis - Analysis Module
Comprehensive statistical analysis of tennis performance across eras.
"""

from .age_curves import (
    analyze_career_trajectories,
    analyze_peak_age_evolution,
    calculate_ranking_peaks,
    create_age_curve_visualizations,
    fit_age_performance_curves,
    prepare_career_data,
)
from .era_analysis import (
    analyze_era_trends,
    compare_surface_performance,
    compute_era_statistics,
    create_era_comparison_plots,
    generate_era_analysis_report,
    identify_era_champions,
    load_player_match_data,
)
from .yearly_trends import (
    analyze_game_evolution_phases,
    analyze_yearly_evolution,
    analyze_yearly_trends,
    create_yearly_trend_visualizations,
    detect_trend_changes,
    prepare_yearly_data,
)

__all__ = [
    # Era analysis functions
    "load_player_match_data",
    "generate_era_analysis_report",
    "compute_era_statistics",
    "analyze_era_trends",
    "compare_surface_performance",
    "identify_era_champions",
    "create_era_comparison_plots",
    # Age curve analysis functions
    "analyze_career_trajectories",
    "prepare_career_data",
    "calculate_ranking_peaks",
    "analyze_peak_age_evolution",
    "fit_age_performance_curves",
    "create_age_curve_visualizations",
    # Yearly trends analysis functions
    "analyze_yearly_evolution",
    "prepare_yearly_data",
    "analyze_yearly_trends",
    "analyze_game_evolution_phases",
    "detect_trend_changes",
    "create_yearly_trend_visualizations",
]
