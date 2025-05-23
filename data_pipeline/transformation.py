"""
Tennis Era Analysis - Data Transformation
Phase 2: Transform match-level data to player-match format with era-focused derived metrics.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.constants import (
    CLEANED_DATA_DIR,
    CORE_COLUMNS,
    ERA_DEFINITIONS,
    LOSER_TO_PLAYER_MAPPING,
    WINNER_TO_PLAYER_MAPPING,
)


def load_standardized_data() -> pd.DataFrame:
    """
    Load standardized ATP matches data.

    Returns:
        Standardized ATP matches DataFrame
    """
    print("=== LOADING STANDARDIZED DATA ===")

    atp_matches = pd.read_csv(CLEANED_DATA_DIR / "atp_matches_standardized.csv")

    # Convert date back to datetime if needed
    if atp_matches["tourney_date"].dtype == "object":
        atp_matches["tourney_date"] = pd.to_datetime(atp_matches["tourney_date"])

    print(f"✅ Loaded ATP Matches: {len(atp_matches):,} rows")
    print(f"   Date range: {atp_matches['tourney_date'].min()} to {atp_matches['tourney_date'].max()}")

    return atp_matches


def add_era_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add era classification based on tournament date.

    Args:
        df: DataFrame with tourney_date column

    Returns:
        DataFrame with era column added
    """
    print("\n=== ADDING ERA CLASSIFICATION ===")

    # Extract year from tourney_date
    df["year"] = df["tourney_date"].dt.year

    # Map years to eras
    def classify_era(year):
        for era_name, (start_year, end_year) in ERA_DEFINITIONS.items():
            if start_year <= year <= end_year:
                return era_name
        return "Unknown"

    df["era"] = df["year"].apply(classify_era)

    # Show era distribution
    era_counts = df["era"].value_counts()
    print("✅ Era classification complete:")
    for era, count in era_counts.items():
        print(f"  {era}: {count:,} matches")

    return df


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tennis-specific derived metrics for era analysis.

    Args:
        df: DataFrame with match statistics

    Returns:
        DataFrame with derived metrics added
    """
    print("\n=== COMPUTING DERIVED METRICS ===")

    # Service metrics
    print("📊 Computing service metrics...")

    # Ace rates
    df["w_ace_rate"] = df["w_ace"] / df["w_svpt"].replace(0, np.nan)
    df["l_ace_rate"] = df["l_ace"] / df["l_svpt"].replace(0, np.nan)

    # Double fault rates
    df["w_df_rate"] = df["w_df"] / df["w_svpt"].replace(0, np.nan)
    df["l_df_rate"] = df["l_df"] / df["l_svpt"].replace(0, np.nan)

    # First serve percentages
    df["w_first_serve_pct"] = df["w_1stIn"] / df["w_svpt"].replace(0, np.nan)
    df["l_first_serve_pct"] = df["l_1stIn"] / df["l_svpt"].replace(0, np.nan)

    # First serve win percentages
    df["w_first_serve_win_pct"] = df["w_1stWon"] / df["w_1stIn"].replace(0, np.nan)
    df["l_first_serve_win_pct"] = df["l_1stWon"] / df["l_1stIn"].replace(0, np.nan)

    # Second serve win percentages
    second_serve_attempts_w = df["w_svpt"] - df["w_1stIn"]
    second_serve_attempts_l = df["l_svpt"] - df["l_1stIn"]
    df["w_second_serve_win_pct"] = df["w_2ndWon"] / second_serve_attempts_w.replace(0, np.nan)
    df["l_second_serve_win_pct"] = df["l_2ndWon"] / second_serve_attempts_l.replace(0, np.nan)

    # Break point metrics
    df["w_break_point_save_pct"] = df["w_bpSaved"] / df["w_bpFaced"].replace(0, np.nan)
    df["l_break_point_save_pct"] = df["l_bpSaved"] / df["l_bpFaced"].replace(0, np.nan)

    # Return metrics (opponent's serve stats)
    df["w_return_points_won"] = df["l_svpt"] - df["l_1stWon"] - df["l_2ndWon"]
    df["l_return_points_won"] = df["w_svpt"] - df["w_1stWon"] - df["w_2ndWon"]

    df["w_return_win_pct"] = df["w_return_points_won"] / df["l_svpt"].replace(0, np.nan)
    df["l_return_win_pct"] = df["l_return_points_won"] / df["w_svpt"].replace(0, np.nan)

    # Service dominance (serve points won / total points)
    total_points_w = df["w_svpt"] + df["l_svpt"]
    serve_points_won_w = df["w_1stWon"] + df["w_2ndWon"]
    serve_points_won_l = df["l_1stWon"] + df["l_2ndWon"]

    df["w_service_dominance"] = serve_points_won_w / total_points_w.replace(0, np.nan)
    df["l_service_dominance"] = serve_points_won_l / total_points_w.replace(0, np.nan)

    print("✅ Derived metrics computed:")
    print("  Service rates: ace_rate, df_rate, first_serve_pct")
    print("  Service effectiveness: first/second_serve_win_pct")
    print("  Pressure situations: break_point_save_pct")
    print("  Return game: return_win_pct")
    print("  Overall dominance: service_dominance")

    return df


def reshape_to_player_match_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from match-level (winner/loser) to player-match format.

    Args:
        df: Match-level DataFrame

    Returns:
        Player-match format DataFrame
    """
    print("\n=== RESHAPING TO PLAYER-MATCH FORMAT ===")

    # Create winner rows
    print("📊 Creating winner rows...")
    winner_df = df[CORE_COLUMNS].copy()

    # Add winner-specific columns
    for match_col, player_col in WINNER_TO_PLAYER_MAPPING.items():
        if match_col in df.columns:
            winner_df[player_col] = df[match_col]

    # Add derived metrics for winners
    derived_cols = [col for col in df.columns if col.startswith("w_") and "rate" in col or "pct" in col or "dominance" in col]
    for col in derived_cols:
        new_col = col.replace("w_", "")
        winner_df[new_col] = df[col]

    # Add match outcome
    winner_df["won_match"] = 1
    winner_df["player_type"] = "winner"

    # Create loser rows
    print("📊 Creating loser rows...")
    loser_df = df[CORE_COLUMNS].copy()

    # Add loser-specific columns
    for match_col, player_col in LOSER_TO_PLAYER_MAPPING.items():
        if match_col in df.columns:
            loser_df[player_col] = df[match_col]

    # Add derived metrics for losers
    derived_cols = [col for col in df.columns if col.startswith("l_") and "rate" in col or "pct" in col or "dominance" in col]
    for col in derived_cols:
        new_col = col.replace("l_", "")
        loser_df[new_col] = df[col]

    # Add match outcome
    loser_df["won_match"] = 0
    loser_df["player_type"] = "loser"

    # Combine winner and loser rows
    print("🔄 Combining winner and loser rows...")
    player_match_df = pd.concat([winner_df, loser_df], ignore_index=True)

    print("✅ Reshape complete:")
    print(f"  Original matches: {len(df):,}")
    print(f"  Player-match rows: {len(player_match_df):,}")
    print(f"  Unique players: {player_match_df['player_name'].nunique():,}")

    return player_match_df


def add_opponent_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent information for each player-match row.

    Args:
        df: Player-match format DataFrame

    Returns:
        DataFrame with opponent context added
    """
    print("\n=== ADDING OPPONENT CONTEXT ===")

    # Create opponent mapping
    match_players = df.groupby("match_id").agg({"player_name": list, "player_id": list, "rank": list, "rank_points": list}).reset_index()

    # Create opponent lookup
    opponent_map = {}
    for _, row in match_players.iterrows():
        match_id = row["match_id"]
        players = row["player_name"]
        player_ids = row["player_id"]
        ranks = row["rank"]
        rank_points = row["rank_points"]

        if len(players) == 2:
            # Player 1's opponent is Player 2
            opponent_map[(match_id, players[0])] = {
                "opponent_name": players[1],
                "opponent_id": player_ids[1],
                "opponent_rank": ranks[1],
                "opponent_rank_points": rank_points[1],
            }
            # Player 2's opponent is Player 1
            opponent_map[(match_id, players[1])] = {
                "opponent_name": players[0],
                "opponent_id": player_ids[0],
                "opponent_rank": ranks[0],
                "opponent_rank_points": rank_points[0],
            }

    # Add opponent information
    df["opponent_name"] = df.apply(lambda row: opponent_map.get((row["match_id"], row["player_name"]), {}).get("opponent_name"), axis=1)
    df["opponent_id"] = df.apply(lambda row: opponent_map.get((row["match_id"], row["player_name"]), {}).get("opponent_id"), axis=1)
    df["opponent_rank"] = df.apply(lambda row: opponent_map.get((row["match_id"], row["player_name"]), {}).get("opponent_rank"), axis=1)
    df["opponent_rank_points"] = df.apply(
        lambda row: opponent_map.get((row["match_id"], row["player_name"]), {}).get("opponent_rank_points"), axis=1
    )

    print("✅ Opponent context added:")
    print(f"  Matches with opponent data: {df['opponent_name'].notna().sum():,}")

    return df


def save_transformed_data(df: pd.DataFrame) -> None:
    """
    Save transformed player-match data.

    Args:
        df: Transformed DataFrame to save
    """
    print("\n=== SAVING TRANSFORMED DATA ===")

    # Ensure output directory exists
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save main dataset
    output_file = CLEANED_DATA_DIR / "player_match_data.csv"
    df.to_csv(output_file, index=False)

    print("✅ Saved transformed data:")
    print(f"  File: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Save era-specific subsets
    for era in df["era"].unique():
        if era != "Unknown":
            era_df = df[df["era"] == era]
            era_file = CLEANED_DATA_DIR / f"player_match_data_{era.lower()}.csv"
            era_df.to_csv(era_file, index=False)
            print(f"  Era subset - {era}: {len(era_df):,} rows -> {era_file.name}")


def transform_to_player_match() -> Dict[str, Any]:
    """
    Main function to transform data to player-match format.

    Returns:
        Dictionary with transformed data and summary statistics
    """
    print("🎾 TENNIS ERA ANALYSIS - PHASE 2: TRANSFORMATION")
    print("=" * 60)

    # Load standardized data
    atp_matches = load_standardized_data()

    # Add era classification
    atp_matches = add_era_classification(atp_matches)

    # Compute derived metrics
    atp_matches = compute_derived_metrics(atp_matches)

    # Reshape to player-match format
    player_match_df = reshape_to_player_match_format(atp_matches)

    # Add opponent context
    player_match_df = add_opponent_context(player_match_df)

    # Save results
    save_transformed_data(player_match_df)

    # Summary
    print("\n=== PHASE 2 SUMMARY ===")
    print("\n✅ TRANSFORMATION COMPLETE:")
    print("📊 Format: Match-level → Player-match format")
    print(f"🏷️  Eras: {', '.join(player_match_df['era'].unique())}")
    print("📈 Metrics: Service, return, pressure situation analytics")
    print("👥 Context: Opponent information added")

    print("\n📊 FINAL DATASET:")
    print(f"  Player-match rows: {len(player_match_df):,}")
    print(f"  Unique players: {player_match_df['player_name'].nunique():,}")
    print(f"  Unique matches: {player_match_df['match_id'].nunique():,}")
    print(f"  Date range: {player_match_df['tourney_date'].min()} to {player_match_df['tourney_date'].max()}")

    # Era breakdown
    print("\n📅 ERA BREAKDOWN:")
    era_summary = (
        player_match_df.groupby("era")
        .agg({"match_id": "nunique", "player_name": "nunique"})
        .rename(columns={"match_id": "matches", "player_name": "players"})
    )

    for era, row in era_summary.iterrows():
        print(f"  {era}: {row['matches']:,} matches, {row['players']:,} players")

    return {
        "player_match_data": player_match_df,
        "summary": {
            "total_rows": len(player_match_df),
            "unique_players": player_match_df["player_name"].nunique(),
            "unique_matches": player_match_df["match_id"].nunique(),
            "era_breakdown": era_summary.to_dict(),
        },
    }


if __name__ == "__main__":
    transform_to_player_match()
