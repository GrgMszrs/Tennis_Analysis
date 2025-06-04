#!/usr/bin/env python3
"""
Tennis Era Analysis - Data Standardization
Phase 1: Convert data types, standardize categorical values, and create universal identifiers.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config.constants import CLEANED_DATA_DIR
from data_pipeline.tournament_normalization import apply_tournament_normalization, normalize_tournament_name


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the cleaned raw datasets for standardization processing.

    Returns:
        Tuple of (atp_matches, atp_pbp) DataFrames from cleaned files
    """
    print("=== LOADING RAW DATASETS FOR STANDARDIZATION ===")

    # Load the cleaned files for standardization
    atp_matches = pd.read_csv(CLEANED_DATA_DIR / "atp_matches_cleaned.csv")
    atp_pbp = pd.read_csv(CLEANED_DATA_DIR / "atp_pbp_cleaned.csv")

    print(f"✅ Loaded ATP Matches (raw/cleaned): {len(atp_matches):,} rows")
    print(f"✅ Loaded ATP PBP (raw/cleaned): {len(atp_pbp):,} rows")

    return atp_matches, atp_pbp


def load_standardized_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the standardized datasets for analysis and matching.

    Returns:
        Tuple of (atp_matches, atp_pbp) DataFrames from standardized files
    """
    print("=== LOADING STANDARDIZED DATASETS ===")

    # Load the standardized files for analysis
    atp_matches = pd.read_csv(CLEANED_DATA_DIR / "atp_matches_standardized.csv")
    atp_pbp = pd.read_csv(CLEANED_DATA_DIR / "atp_pbp_standardized.csv")

    print(f"✅ Loaded ATP Matches (standardized): {len(atp_matches):,} rows")
    print(f"✅ Loaded ATP PBP (standardized): {len(atp_pbp):,} rows")

    return atp_matches, atp_pbp


def standardize_dates(df: pd.DataFrame, date_column: str = "tourney_date") -> pd.DataFrame:
    """
    Standardize date columns to proper datetime format.

    Args:
        df: DataFrame to process
        date_column: Name of the date column to standardize

    Returns:
        DataFrame with standardized dates
    """
    print(f"\n=== STANDARDIZING DATES: {date_column} ===")

    # Keep raw integer as backup
    df[f"{date_column}_int"] = df[date_column]

    # Convert to proper DATE format (YYYYMMDD -> DATE)
    df[date_column] = pd.to_datetime(df[date_column].astype(str), format="%Y%m%d", errors="coerce")

    print("✅ Date conversion complete:")
    print(f"  New {date_column} type: {df[date_column].dtype}")
    print(f"  Date range: {df[date_column].min()} to {df[date_column].max()}")

    # Check for conversion errors
    null_dates = df[date_column].isnull().sum()
    print(f"  Date conversion errors: {null_dates}")

    return df


def standardize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric columns, handling sentinel values.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with standardized numeric columns
    """
    print("\n=== STANDARDIZING NUMERICS ===")

    # Identify numeric columns
    numeric_columns = [col for col in df.columns if col.startswith("w_") or col.startswith("l_") or "rank" in col.lower()]

    print(f"Identified {len(numeric_columns)} numeric columns")

    # Convert numeric columns, handling sentinels
    for col in numeric_columns:
        # Replace common sentinel values with NaN
        df[col] = df[col].replace(["", "-", "--", "-1"], np.nan)

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

        null_count = df[col].isnull().sum()
        print(f"  {col}: standardized ({null_count} nulls)")

    return df


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical columns with consistent codes.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with standardized categorical columns
    """
    print("\n=== STANDARDIZING CATEGORICALS ===")

    categorical_columns = ["winner_hand", "loser_hand", "surface", "round"]

    for col in categorical_columns:
        if col in df.columns:
            print(f"\n📊 Standardizing {col}:")

            # Show current values
            current_values = df[col].value_counts()
            print(f"  Current values: {len(current_values)} unique")

            # Standardize: upper case, trim whitespace
            df[col] = df[col].astype(str).str.upper().str.strip()

            # Apply specific mappings
            if col in ["winner_hand", "loser_hand"]:
                hand_mapping = {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R", "U": "U", "UNKNOWN": "U", "NAN": np.nan, "NONE": np.nan}
                df[col] = df[col].map(hand_mapping).fillna(df[col])

            elif col == "surface":
                surface_mapping = {"HARD": "Hard", "CLAY": "Clay", "GRASS": "Grass", "CARPET": "Carpet", "NAN": np.nan, "NONE": np.nan}
                df[col] = df[col].map(surface_mapping).fillna(df[col])

            # Show results
            new_values = df[col].value_counts()
            print(f"  ✅ After standardization: {len(new_values)} unique")

    return df


def create_universal_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create universal match ID for consistent linking.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with match_id column
    """
    print("\n=== CREATING UNIVERSAL MATCH ID ===")

    # Create match_id from tourney_id + match_num
    df["match_id"] = df["tourney_id"].astype(str) + "-" + df["match_num"].astype(str)

    print("✅ Created match_id column:")
    print("  Format: tourney_id + '-' + match_num")
    print(f"  Sample match_ids: {df['match_id'].head().tolist()}")
    print(f"  Unique match_ids: {df['match_id'].nunique():,}")
    print(f"  Total rows: {len(df):,}")

    # Check for duplicates
    duplicates = df["match_id"].duplicated().sum()
    print(f"  Duplicate match_ids: {duplicates}")

    return df


def standardize_atp_tournament_names(df: pd.DataFrame, enable_tournament_normalization: bool = False) -> pd.DataFrame:
    """
    Standardize ATP tournament names.

    Args:
        df: ATP matches DataFrame to process
        enable_tournament_normalization: Whether to apply tournament normalization

    Returns:
        DataFrame with standardized tournament names
    """
    if not enable_tournament_normalization:
        print("\n   ⚠️  Tournament normalization disabled - using original tournament names")
        return df

    print("\n=== STANDARDIZING ATP TOURNAMENT NAMES ===")

    if "tourney_name" not in df.columns:
        print("   ⚠️  No 'tourney_name' column found, skipping tournament normalization")
        return df

    # Apply tournament normalization to ATP data
    df["tourney_name_normalized"] = df["tourney_name"].apply(normalize_tournament_name)

    # Create stats
    original_unique = df["tourney_name"].nunique()
    normalized_unique = df["tourney_name_normalized"].nunique()

    print("✅ ATP tournament normalization complete:")
    print(f"   Original unique tournaments: {original_unique:,}")
    print(f"   Normalized unique tournaments: {normalized_unique:,}")
    print(f"   Consolidation: {original_unique - normalized_unique:,} tournaments merged")

    # Show sample mappings
    print("   Sample normalizations:")
    sample_mappings = df[["tourney_name", "tourney_name_normalized"]].drop_duplicates().head(5)
    for _, row in sample_mappings.iterrows():
        if row["tourney_name"] != row["tourney_name_normalized"]:
            print(f"     {row['tourney_name'][:40]:<40} → {row['tourney_name_normalized']}")

    return df


def standardize_pbp_tournament_names(df: pd.DataFrame, enable_tournament_normalization: bool = False) -> pd.DataFrame:
    """
    Standardize PBP tournament names.

    Args:
        df: PBP DataFrame to process
        enable_tournament_normalization: Whether to apply tournament normalization

    Returns:
        DataFrame with standardized tournament names
    """
    if not enable_tournament_normalization:
        print("\n   ⚠️  Tournament normalization disabled - using original tournament names")
        return df

    print("\n=== STANDARDIZING PBP TOURNAMENT NAMES ===")

    # PBP data uses 'tny_name' for tournament names, not 'tournament_name'
    if "tny_name" not in df.columns:
        print("   ⚠️  No 'tny_name' column found, skipping tournament normalization")
        return df

    # Create a temporary column with the expected name for apply_tournament_normalization
    df_temp = df.copy()
    df_temp["tournament_name"] = df_temp["tny_name"]

    # Apply tournament normalization to PBP data
    result = apply_tournament_normalization(df_temp)

    # Copy the normalization result back to the original column name format
    df["tny_name_normalized"] = result["tournament_name_normalized"]

    # Create stats
    original_unique = df["tny_name"].nunique()
    normalized_unique = df["tny_name_normalized"].nunique()

    print("✅ PBP tournament normalization complete:")
    print(f"   Original unique tournaments: {original_unique:,}")
    print(f"   Normalized unique tournaments: {normalized_unique:,}")
    print(f"   Consolidation: {original_unique - normalized_unique:,} tournaments merged")

    # Show sample mappings
    print("   Sample normalizations:")
    sample_mappings = df[["tny_name", "tny_name_normalized"]].drop_duplicates().head(5)
    for _, row in sample_mappings.iterrows():
        if row["tny_name"] != row["tny_name_normalized"]:
            print(f"     {row['tny_name'][:40]:<40} → {row['tny_name_normalized']}")

    return df


def standardize_pbp_data(atp_pbp: pd.DataFrame, enable_tournament_normalization: bool = False) -> pd.DataFrame:
    """
    Standardize point-by-point data.

    Args:
        atp_pbp: PBP DataFrame to process
        enable_tournament_normalization: Whether to apply tournament normalization

    Returns:
        Standardized PBP DataFrame
    """
    print("\n=== STANDARDIZING ATP PBP DATA ===")

    # Convert PBP dates
    print("\n📅 Converting PBP dates:")
    atp_pbp["date_standardized"] = pd.to_datetime(atp_pbp["date"], errors="coerce")
    date_errors = atp_pbp["date_standardized"].isnull().sum()
    print(f"  Converted to datetime: {atp_pbp['date_standardized'].dtype}")
    print(f"  Date range: {atp_pbp['date_standardized'].min()} to {atp_pbp['date_standardized'].max()}")
    print(f"  Conversion errors: {date_errors}")

    # Standardize categoricals
    print("\n🏷️ Standardizing PBP categoricals:")
    if "tour" in atp_pbp.columns:
        atp_pbp["tour"] = atp_pbp["tour"].astype(str).str.upper().str.strip()
        print(f"  Tour values: {atp_pbp['tour'].value_counts().to_dict()}")

    if "draw" in atp_pbp.columns:
        atp_pbp["draw"] = atp_pbp["draw"].astype(str).str.upper().str.strip()
        print(f"  Draw values: {atp_pbp['draw'].value_counts().to_dict()}")

    # Apply tournament normalization if enabled
    atp_pbp = standardize_pbp_tournament_names(atp_pbp, enable_tournament_normalization)

    # Create PBP match_id
    print("\n🔑 Creating PBP match_id:")
    atp_pbp["match_id"] = "pbp_" + atp_pbp["pbp_id"].astype(str)
    print("  Created PBP match_id: pbp_<pbp_id>")
    print(f"  Sample match_ids: {atp_pbp['match_id'].head().tolist()}")

    return atp_pbp


def save_standardized_datasets(atp_matches: pd.DataFrame, atp_pbp: pd.DataFrame) -> None:
    """
    Save standardized datasets to disk.

    Args:
        atp_matches: Standardized ATP matches DataFrame
        atp_pbp: Standardized ATP PBP DataFrame
    """
    print("\n=== SAVING STANDARDIZED DATA ===")

    # Ensure output directory exists
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    atp_matches.to_csv(CLEANED_DATA_DIR / "atp_matches_standardized.csv", index=False)
    atp_pbp.to_csv(CLEANED_DATA_DIR / "atp_pbp_standardized.csv", index=False)

    print(f"✅ Saved standardized datasets to {CLEANED_DATA_DIR}/")
    print("   Files created:")
    print("   - atp_matches_standardized.csv")
    print("   - atp_pbp_standardized.csv")


def standardize_datasets(enable_tournament_normalization: bool = True) -> Dict[str, Any]:
    """
    Main function to standardize all datasets.

    Args:
        enable_tournament_normalization: Whether to apply tournament name normalization

    Returns:
        Dictionary with standardized datasets and summary statistics
    """
    print("🎾 TENNIS ERA ANALYSIS - PHASE 1: STANDARDIZATION")
    print("=" * 60)

    if enable_tournament_normalization:
        print("🏆 Tournament normalization: ENABLED (default)")
    else:
        print("🏆 Tournament normalization: DISABLED (use --no-tournament-normalization to disable)")

    # Load raw datasets
    atp_matches, atp_pbp = load_raw_datasets()

    # Standardize ATP matches
    atp_matches = standardize_dates(atp_matches, "tourney_date")
    atp_matches = standardize_numerics(atp_matches)
    atp_matches = standardize_categoricals(atp_matches)
    atp_matches = standardize_atp_tournament_names(atp_matches, enable_tournament_normalization)
    atp_matches = create_universal_match_id(atp_matches)

    # Standardize PBP data
    atp_pbp = standardize_pbp_data(atp_pbp, enable_tournament_normalization)

    # Save results
    save_standardized_datasets(atp_matches, atp_pbp)

    # Summary
    print("\n=== PHASE 1 SUMMARY ===")
    print("\n✅ STANDARDIZATION COMPLETE:")
    print("📅 Dates: ATP matches tourney_date + ATP PBP date converted to datetime")
    print(
        f"🔢 Numerics: {len([col for col in atp_matches.columns if col.startswith(('w_', 'l_')) or 'rank' in col.lower()])} ATP match columns + PBP validation"
    )
    print("🏷️  Categories: ATP match categories + PBP tour/draw standardized")
    if enable_tournament_normalization:
        print("🏆 Tournaments: ATP + PBP tournament names normalized for better matching")
    print("🔑 Match ID: Universal match_id created for both datasets")

    print("\n📊 FINAL DATASETS:")
    print(f"  ATP Matches: {len(atp_matches):,} rows, {len(atp_matches.columns)} columns")
    print(f"    Date range: {atp_matches['tourney_date'].dt.year.min()} - {atp_matches['tourney_date'].dt.year.max()}")
    print(f"    Unique matches: {atp_matches['match_id'].nunique():,}")
    if enable_tournament_normalization and "tourney_name_normalized" in atp_matches.columns:
        print(f"    Unique tournaments (normalized): {atp_matches['tourney_name_normalized'].nunique():,}")
    print(f"  ATP PBP: {len(atp_pbp):,} rows, {len(atp_pbp.columns)} columns")
    print(f"    Date range: {atp_pbp['date_standardized'].dt.year.min()} - {atp_pbp['date_standardized'].dt.year.max()}")
    print(f"    Unique matches: {atp_pbp['match_id'].nunique():,}")
    if enable_tournament_normalization and "tny_name_normalized" in atp_pbp.columns:
        print(f"    Unique tournaments (normalized): {atp_pbp['tny_name_normalized'].nunique():,}")

    return {
        "atp_matches": atp_matches,
        "atp_pbp": atp_pbp,
        "summary": {
            "atp_matches_count": len(atp_matches),
            "atp_pbp_count": len(atp_pbp),
            "tournament_normalization_enabled": enable_tournament_normalization,
            "date_range": (atp_matches["tourney_date"].min(), atp_matches["tourney_date"].max()),
        },
    }


if __name__ == "__main__":
    standardize_datasets()
