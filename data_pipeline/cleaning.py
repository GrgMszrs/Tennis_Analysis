#!/usr/bin/env python3
"""
Tennis Analysis - Data Cleaning
Phase 1a: Clean raw data before standardization.
Reconstructed from comparison analysis between raw and cleaned datasets.
"""

import logging
from typing import Tuple

import pandas as pd

from config.constants import CLEANED_DATA_DIR, RAW_DATA_DIR

# Set up logging
logger = logging.getLogger(__name__)


def load_truly_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the truly raw datasets (aggregated files) for cleaning.

    Returns:
        Tuple of (atp_matches, atp_pbp) DataFrames from raw aggregated files
    """
    print("=== LOADING TRULY RAW DATASETS FOR CLEANING ===")

    # Load the raw aggregated files
    raw_matches_path = RAW_DATA_DIR / "atp_matches" / "aggregated_atp_matches.csv"
    raw_pbp_path = RAW_DATA_DIR / "atp_point_by_point" / "aggregated_pbp_matches.csv"

    if not raw_matches_path.exists():
        raise FileNotFoundError(f"Raw ATP matches file not found: {raw_matches_path}")
    if not raw_pbp_path.exists():
        raise FileNotFoundError(f"Raw ATP PBP file not found: {raw_pbp_path}")

    atp_matches = pd.read_csv(raw_matches_path)
    atp_pbp = pd.read_csv(raw_pbp_path)

    print(f"✅ Loaded ATP Matches (truly raw): {len(atp_matches):,} rows, {len(atp_matches.columns)} columns")
    print(f"✅ Loaded ATP PBP (truly raw): {len(atp_pbp):,} rows, {len(atp_pbp.columns)} columns")

    return atp_matches, atp_pbp


def clean_atp_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ATP Matches dataset.
    Removes rows with missing critical statistics and invalid data.

    Expected: 58,502 → 58,081 rows (-421 rows, -0.72%)

    Args:
        df: Raw ATP matches DataFrame

    Returns:
        Cleaned ATP matches DataFrame
    """
    print("\n=== CLEANING ATP MATCHES ===")
    print(f"Starting ATP Matches cleaning: {len(df):,} rows")
    original_count = len(df)

    # Step 1: Remove rows with missing critical match statistics
    # Based on analysis: 29 columns had reduced missing values
    # Strategy: Remove matches with missing critical match statistics
    critical_stats = ["w_svpt", "l_svpt", "winner_rank", "loser_rank"]
    before_missing = len(df)
    df_clean = df.dropna(subset=critical_stats, how="any")
    after_missing = len(df_clean)
    print(f"  Removed {before_missing - after_missing:,} rows with missing critical stats")

    # Step 2: Remove invalid match data
    # Filter out matches with impossible statistics
    before_invalid = len(df_clean)
    df_clean = df_clean[
        (df_clean["best_of"].isin([3, 5]))  # Valid match formats
        & (df_clean["minutes"] >= 20)  # Minimum realistic match duration
        & (df_clean["winner_age"] >= 14)  # Minimum professional age
        & (df_clean["loser_age"] >= 14)
    ]
    after_invalid = len(df_clean)
    print(f"  Removed {before_invalid - after_invalid:,} rows with invalid match data")

    final_count = len(df_clean)
    total_removed = original_count - final_count
    print(f"✅ ATP Matches cleaning complete: {final_count:,} rows ({total_removed:,} removed, {(total_removed/original_count)*100:.2f}%)")

    return df_clean


def clean_atp_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ATP Point-by-Point dataset.
    Removes duplicates, invalid durations, adds date parsing, validates PbP data.

    Expected: 13,050 → 11,859 rows (-1,191 rows, -9.13%)

    Args:
        df: Raw ATP PBP DataFrame

    Returns:
        Cleaned ATP PBP DataFrame
    """
    print("\n=== CLEANING ATP POINT-BY-POINT ===")
    print(f"Starting ATP PbP cleaning: {len(df):,} rows")
    original_count = len(df)

    # Step 1: Remove exact duplicates (38 found in analysis)
    before_dupes = len(df)
    df_clean = df.drop_duplicates()
    after_dupes = len(df_clean)
    print(f"  Removed {before_dupes - after_dupes:,} duplicate rows")

    # Step 2: Filter invalid match durations
    # Key finding: wh_minutes min changed from -1398 to 20
    before_duration = len(df_clean)
    df_clean = df_clean[df_clean["wh_minutes"] >= 20]  # Remove negative/invalid durations
    after_duration = len(df_clean)
    print(f"  Removed {before_duration - after_duration:,} rows with invalid match durations (<20 min)")

    # Step 3: Add parsed_date column (found in cleaned data)
    df_clean["parsed_date"] = pd.to_datetime(df_clean["date"], format="%d %b %y", errors="coerce")
    valid_dates = df_clean["parsed_date"].notna()
    before_dates = len(df_clean)
    df_clean = df_clean[valid_dates]  # Remove rows with unparseable dates
    after_dates = len(df_clean)
    print(f"  Added parsed_date column, removed {before_dates - after_dates:,} rows with invalid dates")

    # Step 4: Remove matches with invalid point-by-point data
    before_pbp = len(df_clean)
    df_clean = df_clean[
        (df_clean["pbp"].str.len() > 10)  # Minimum realistic point sequence
        & (df_clean["winner"].isin([1, 2]))  # Valid winner values
        & (df_clean["adf_flag"].isin([0, 1]))  # Valid flag values
    ]
    after_pbp = len(df_clean)
    print(f"  Removed {before_pbp - after_pbp:,} rows with invalid PbP data")

    final_count = len(df_clean)
    total_removed = original_count - final_count
    print(f"✅ ATP PbP cleaning complete: {final_count:,} rows ({total_removed:,} removed, {(total_removed/original_count)*100:.2f}%)")

    return df_clean


def save_cleaned_datasets(atp_matches: pd.DataFrame, atp_pbp: pd.DataFrame) -> None:
    """
    Save cleaned datasets to disk for standardization phase.

    Args:
        atp_matches: Cleaned ATP matches DataFrame
        atp_pbp: Cleaned ATP PBP DataFrame
    """
    print("\n=== SAVING CLEANED DATASETS ===")

    # Ensure output directory exists
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    matches_output = CLEANED_DATA_DIR / "atp_matches_cleaned.csv"
    pbp_output = CLEANED_DATA_DIR / "atp_pbp_cleaned.csv"

    atp_matches.to_csv(matches_output, index=False)
    atp_pbp.to_csv(pbp_output, index=False)

    print(f"✅ Saved cleaned datasets to {CLEANED_DATA_DIR}/")
    print("   Files created:")
    print("   - atp_matches_cleaned.csv")
    print("   - atp_pbp_cleaned.csv")


def clean_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to clean all raw datasets.
    This is Phase 1a - executed before standardization.

    Returns:
        Tuple of (cleaned_atp_matches, cleaned_atp_pbp) DataFrames
    """
    print("🧹 TENNIS ANALYSIS - PHASE 1A: DATA CLEANING")
    print("=" * 60)
    print("Reconstructed cleaning process based on raw vs cleaned comparison analysis")

    # Load truly raw datasets
    atp_matches, atp_pbp = load_truly_raw_datasets()

    # Clean datasets
    cleaned_matches = clean_atp_matches(atp_matches)
    cleaned_pbp = clean_atp_pbp(atp_pbp)

    # Save cleaned datasets for standardization phase
    save_cleaned_datasets(cleaned_matches, cleaned_pbp)

    # Summary
    print("\n=== PHASE 1A SUMMARY ===")
    print("\n✅ DATA CLEANING COMPLETE:")
    print("🔍 Duplicate removal: ATP PBP duplicates eliminated")
    print("⏱️  Invalid durations: Negative/unrealistic match durations filtered")
    print("📅 Date validation: Unparseable dates removed, parsed_date column added")
    print("🎾 Match validation: Invalid match formats and statistics filtered")
    print("👥 Player validation: Unrealistic player ages filtered")
    print("📊 Data quality: Point-by-point sequences and flags validated")

    print("\n📊 CLEANING RESULTS:")
    print(f"  ATP Matches: {len(atp_matches):,} → {len(cleaned_matches):,} rows")
    print(
        f"    Removed: {len(atp_matches) - len(cleaned_matches):,} rows ({((len(atp_matches) - len(cleaned_matches))/len(atp_matches)*100):.2f}%)"
    )
    print(f"  ATP PBP: {len(atp_pbp):,} → {len(cleaned_pbp):,} rows")
    print(f"    Removed: {len(atp_pbp) - len(cleaned_pbp):,} rows ({((len(atp_pbp) - len(cleaned_pbp))/len(atp_pbp)*100):.2f}%)")
    print("  Enhanced: parsed_date column added to PBP data")

    return cleaned_matches, cleaned_pbp


if __name__ == "__main__":
    clean_datasets()
