"""
Tennis Era Analysis - Utility Functions
Common helper functions used across the project.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("tennis_era_analysis")
    return logger


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame") -> bool:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages

    Returns:
        True if valid, raises ValueError if not
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{name} missing required columns: {missing_columns}")

    return True


def clean_player_name(name: str) -> str:
    """
    Clean and standardize player names.

    Args:
        name: Raw player name

    Returns:
        Cleaned player name
    """
    if pd.isna(name) or name == "":
        return ""

    # Convert to string and strip whitespace
    name = str(name).strip()

    # Remove extra spaces
    name = " ".join(name.split())

    # Standardize case (Title Case)
    name = name.title()

    return name


def calculate_age_from_birthdate(birthdate: Union[str, datetime], reference_date: Union[str, datetime]) -> Optional[float]:
    """
    Calculate age from birthdate and reference date.

    Args:
        birthdate: Birth date
        reference_date: Reference date (e.g., match date)

    Returns:
        Age in years, or None if calculation fails
    """
    try:
        if pd.isna(birthdate) or pd.isna(reference_date):
            return None

        birth_dt = pd.to_datetime(birthdate)
        ref_dt = pd.to_datetime(reference_date)

        age = (ref_dt - birth_dt).days / 365.25
        return round(age, 2)

    except Exception:
        return None


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default value
    """
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return default

    return numerator / denominator


def calculate_win_percentage(wins: int, total_matches: int) -> float:
    """
    Calculate win percentage.

    Args:
        wins: Number of wins
        total_matches: Total number of matches

    Returns:
        Win percentage (0-100)
    """
    if total_matches == 0:
        return 0.0

    return round((wins / total_matches) * 100, 2)


def standardize_surface_name(surface: str) -> str:
    """
    Standardize surface names.

    Args:
        surface: Raw surface name

    Returns:
        Standardized surface name
    """
    if pd.isna(surface):
        return "Unknown"

    surface = str(surface).strip().lower()

    surface_mapping = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet", "indoor": "Indoor", "outdoor": "Outdoor"}

    return surface_mapping.get(surface, surface.title())


def format_match_score(score: str) -> str:
    """
    Format and clean match scores.

    Args:
        score: Raw match score

    Returns:
        Formatted match score
    """
    if pd.isna(score) or score == "":
        return ""

    # Remove extra whitespace
    score = str(score).strip()

    # Remove common artifacts
    score = score.replace("  ", " ")

    return score


def calculate_service_metrics(row: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive service metrics from match statistics.

    Args:
        row: DataFrame row with service statistics

    Returns:
        Dictionary of calculated service metrics
    """
    metrics = {}

    # Basic service metrics
    serve_points = row.get("serve_points", 0)
    if serve_points > 0:
        metrics["ace_rate"] = safe_divide(row.get("aces", 0), serve_points)
        metrics["df_rate"] = safe_divide(row.get("double_faults", 0), serve_points)
        metrics["first_serve_pct"] = safe_divide(row.get("first_serves_in", 0), serve_points)

    # First serve effectiveness
    first_serves_in = row.get("first_serves_in", 0)
    if first_serves_in > 0:
        metrics["first_serve_win_pct"] = safe_divide(row.get("first_serves_won", 0), first_serves_in)

    # Second serve effectiveness
    second_serve_attempts = serve_points - first_serves_in
    if second_serve_attempts > 0:
        metrics["second_serve_win_pct"] = safe_divide(row.get("second_serves_won", 0), second_serve_attempts)

    # Break point performance
    break_points_faced = row.get("break_points_faced", 0)
    if break_points_faced > 0:
        metrics["break_point_save_pct"] = safe_divide(row.get("break_points_saved", 0), break_points_faced)

    return metrics


def detect_outliers(series: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a pandas Series.

    Args:
        series: Data series
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def create_summary_statistics(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Create summary statistics for numeric columns.

    Args:
        df: DataFrame
        numeric_columns: List of numeric column names

    Returns:
        Summary statistics DataFrame
    """
    summary_stats = []

    for col in numeric_columns:
        if col in df.columns:
            series = df[col].dropna()

            if len(series) > 0:
                stats = {
                    "column": col,
                    "count": len(series),
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "q25": series.quantile(0.25),
                    "median": series.median(),
                    "q75": series.quantile(0.75),
                    "max": series.max(),
                    "missing": df[col].isna().sum(),
                    "missing_pct": (df[col].isna().sum() / len(df)) * 100,
                }
                summary_stats.append(stats)

    return pd.DataFrame(summary_stats)


def export_to_multiple_formats(df: pd.DataFrame, base_filename: str, output_dir: Path, formats: List[str] = None) -> Dict[str, Path]:
    """
    Export DataFrame to multiple file formats.

    Args:
        df: DataFrame to export
        base_filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats ('csv', 'parquet', 'excel')

    Returns:
        Dictionary mapping format to file path
    """
    if formats is None:
        formats = ["csv"]

    output_dir = ensure_directory_exists(output_dir)
    exported_files = {}

    for fmt in formats:
        if fmt == "csv":
            file_path = output_dir / f"{base_filename}.csv"
            df.to_csv(file_path, index=False)

        elif fmt == "parquet":
            file_path = output_dir / f"{base_filename}.parquet"
            df.to_parquet(file_path, index=False)

        elif fmt == "excel":
            file_path = output_dir / f"{base_filename}.xlsx"
            df.to_excel(file_path, index=False)

        else:
            continue

        exported_files[fmt] = file_path

    return exported_files


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive information about a DataFrame.

    Args:
        df: DataFrame to analyze
        name: Name for display
    """
    print(f"\n=== {name.upper()} INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn types:")
    for dtype in df.dtypes.value_counts().index:
        count = df.dtypes.value_counts()[dtype]
        print(f"  {dtype}: {count} columns")

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")

    if len(missing[missing > 0]) == 0:
        print("  No missing values")


if __name__ == "__main__":
    print("🎾 Tennis Era Analysis - Utility Functions")
    print("This module provides helper functions for data processing and analysis.")
