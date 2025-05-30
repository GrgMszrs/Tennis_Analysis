"""
Tennis Era Analysis - Data Transformation
Phase 2: Transform match-level data to player-match format with era-focused derived metrics.
Enhanced with flexible z-scoring and historical ranking integration.
"""

from io import StringIO
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests

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
    Save transformed player-match data with enhanced features.

    Args:
        df: Transformed DataFrame to save
    """
    print("\n=== SAVING ENHANCED TRANSFORMED DATA ===")

    # Ensure output directory exists
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save main enhanced dataset
    output_file = CLEANED_DATA_DIR / "atp_player_match_enhanced.csv"
    df.to_csv(output_file, index=False)

    print("✅ Saved enhanced transformed data:")
    print(f"  File: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Count enhanced features
    z_score_cols = [col for col in df.columns if "_z_" in col]
    ranking_cols = [col for col in df.columns if "historical" in col]
    print(f"  Z-score features: {len(z_score_cols)}")
    print(f"  Historical ranking features: {len(ranking_cols)}")

    # Save summary file for quick reference
    summary_data = {
        "total_rows": len(df),
        "unique_players": df["player_name"].nunique(),
        "unique_matches": df["match_id"].nunique(),
        "date_range_start": str(df["tourney_date"].min()),
        "date_range_end": str(df["tourney_date"].max()),
        "z_score_features": len(z_score_cols),
        "ranking_features": len(ranking_cols),
        "eras": list(df["era"].unique()),
    }

    import json

    summary_file = CLEANED_DATA_DIR / "atp_player_match_enhanced_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"  Summary: {summary_file.name}")
    print("📈 Enhanced dataset ready for era analysis!")


def transform_to_player_match() -> Dict[str, Any]:
    """
    Main function to transform data to player-match format with enhanced features.

    Returns:
        Dictionary with transformed data and summary statistics
    """
    print("🎾 TENNIS ERA ANALYSIS - PHASE 2: ENHANCED TRANSFORMATION")
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

    # Add enhanced normalization (both year-only and year+surface z-scores)
    player_match_df = add_enhanced_normalization(player_match_df)

    # Integrate historical rankings from Sackmann repository
    player_match_df = integrate_historical_rankings(player_match_df)

    # Save results
    save_transformed_data(player_match_df)

    # Summary
    print("\n=== PHASE 2 ENHANCED SUMMARY ===")
    print("\n✅ TRANSFORMATION COMPLETE:")
    print("📊 Format: Match-level → Player-match format")
    print(f"🏷️  Eras: {', '.join(player_match_df['era'].unique())}")
    print("📈 Metrics: Service, return, pressure situation analytics")
    print("📊 Normalization: Year-only & Year+Surface z-scores")
    print("🏆 Rankings: Historical Sackmann ranking integration")
    print("👥 Context: Opponent information added")

    print("\n📊 FINAL ENHANCED DATASET:")
    print(f"  Player-match rows: {len(player_match_df):,}")
    print(f"  Unique players: {player_match_df['player_name'].nunique():,}")
    print(f"  Unique matches: {player_match_df['match_id'].nunique():,}")
    print(f"  Date range: {player_match_df['tourney_date'].min()} to {player_match_df['tourney_date'].max()}")

    # Feature summary
    z_score_cols = [col for col in player_match_df.columns if "_z_" in col]
    ranking_cols = [col for col in player_match_df.columns if "historical" in col]
    print(f"  Z-score columns: {len(z_score_cols)}")
    print(f"  Historical ranking columns: {len(ranking_cols)}")

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
            "z_score_features": len(z_score_cols),
            "ranking_features": len(ranking_cols),
        },
    }


# =============================================================================
# SACKMANN DATA INTEGRATION
# =============================================================================


def fetch_sackmann_players() -> pd.DataFrame:
    """
    Fetch ATP players data from Jeff Sackmann's repository.

    Returns:
        DataFrame with player information
    """
    print("\n=== FETCHING SACKMANN PLAYERS DATA ===")

    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()

        players_df = pd.read_csv(StringIO(response.text))
        print(f"✅ Fetched {len(players_df):,} players from Sackmann repo")
        print(f"   Columns: {list(players_df.columns)}")

        return players_df

    except requests.RequestException as e:
        print(f"❌ Error fetching players data: {e}")
        return pd.DataFrame()


def fetch_sackmann_rankings() -> pd.DataFrame:
    """
    Fetch historical ATP rankings from Jeff Sackmann's repository.

    Returns:
        Combined DataFrame with all rankings data
    """
    print("\n=== FETCHING SACKMANN RANKINGS DATA ===")

    ranking_urls = {
        "2000s": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_00s.csv",
        "2010s": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_10s.csv",
        "2020s": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_20s.csv",
        "current": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",  # 2024+ data
    }

    rankings_dfs = []

    for period, url in ranking_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))
            df["decade"] = period
            rankings_dfs.append(df)

            print(f"✅ Fetched {period} rankings: {len(df):,} records")

        except requests.RequestException as e:
            print(f"❌ Error fetching {period} rankings: {e}")

    if rankings_dfs:
        combined_rankings = pd.concat(rankings_dfs, ignore_index=True)
        print(f"📊 Combined rankings: {len(combined_rankings):,} total records")
        print(f"   Date range: {combined_rankings['ranking_date'].min()} to {combined_rankings['ranking_date'].max()}")
        return combined_rankings

    return pd.DataFrame()


def integrate_historical_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate historical ranking data with player-match data.

    Args:
        df: Player-match DataFrame

    Returns:
        DataFrame with historical ranking information added
    """
    print("\n=== INTEGRATING HISTORICAL RANKINGS ===")

    # Get our actual data constraints
    our_players = df["player_name"].unique()
    our_date_range = (df["tourney_date"].min(), df["tourney_date"].max())
    print(f"📅 Our data range: {our_date_range[0].strftime('%Y-%m-%d')} to {our_date_range[1].strftime('%Y-%m-%d')}")
    print(f"👥 Our players: {len(our_players):,} unique players")

    # Fetch Sackmann data
    players_ref = fetch_sackmann_players()
    rankings_ref = fetch_sackmann_rankings()

    if players_ref.empty or rankings_ref.empty:
        print("⚠️ Could not fetch reference data, skipping ranking integration")
        return df

    # Prepare ranking data - filter to relevant date range
    rankings_ref["ranking_date"] = pd.to_datetime(rankings_ref["ranking_date"], format="%Y%m%d", errors="coerce")

    # Filter rankings to our date range (with some buffer for earlier rankings)
    date_start = our_date_range[0] - pd.Timedelta(days=365)  # 1 year buffer for earlier rankings
    date_end = our_date_range[1]

    rankings_filtered = rankings_ref[(rankings_ref["ranking_date"] >= date_start) & (rankings_ref["ranking_date"] <= date_end)]

    print(f"📊 Filtered rankings: {len(rankings_filtered):,} records (from {len(rankings_ref):,})")

    # Create full name column in players_ref if it doesn't exist
    if "name_full" not in players_ref.columns:
        players_ref["name_full"] = (players_ref["name_first"].fillna("") + " " + players_ref["name_last"].fillna("")).str.strip()

    # Match only our players by name first, then by ID where possible
    print("🔍 Matching players...")

    # Create a mapping from our player names to Sackmann IDs
    player_mapping = {}

    for player_name in our_players:
        # Try exact name match first
        name_match = players_ref[players_ref["name_full"].str.lower() == player_name.lower()]
        if not name_match.empty:
            player_mapping[player_name] = name_match.iloc[0]["player_id"]
            continue

        # Try fuzzy matching for close names (could improve this)
        # For now, skip if no exact match

    print(f"✅ Mapped {len(player_mapping)} players to Sackmann IDs")

    # Filter rankings to only our mapped players for efficiency
    our_sackmann_ids = list(player_mapping.values())
    rankings_our_players = rankings_filtered[rankings_filtered["player"].isin(our_sackmann_ids)]
    print(f"📈 Relevant rankings: {len(rankings_our_players):,} records for our players")

    # Add historical ranking information efficiently
    print("🔄 Adding historical rankings...")

    # Create ranking lookup for faster processing
    ranking_lookup = {}
    for _, row in rankings_our_players.iterrows():
        player_id = row["player"]
        date = row["ranking_date"]
        if player_id not in ranking_lookup:
            ranking_lookup[player_id] = []
        ranking_lookup[player_id].append({"date": date, "rank": row["rank"], "points": row.get("points", np.nan)})

    # Sort rankings by date for each player
    for player_id in ranking_lookup:
        ranking_lookup[player_id].sort(key=lambda x: x["date"])

    # Add rankings to dataframe
    historical_ranks = []
    historical_points = []
    matches_with_rankings = 0

    for _, row in df.iterrows():
        player_name = row["player_name"]
        match_date = row["tourney_date"]

        # Default values
        historical_rank = np.nan
        historical_points_val = np.nan

        if player_name in player_mapping:
            sackmann_id = player_mapping[player_name]

            if sackmann_id in ranking_lookup:
                # Find the most recent ranking before or on match date
                player_rankings = ranking_lookup[sackmann_id]

                # Binary search would be more efficient, but this is clearer
                for ranking in reversed(player_rankings):  # Start from most recent
                    if ranking["date"] <= match_date:
                        historical_rank = ranking["rank"]
                        historical_points_val = ranking["points"]
                        matches_with_rankings += 1
                        break

        historical_ranks.append(historical_rank)
        historical_points.append(historical_points_val)

    # Add to DataFrame
    df["historical_rank"] = historical_ranks
    df["historical_points"] = historical_points

    coverage_pct = (matches_with_rankings / len(df)) * 100
    print("📈 Added historical ranking data:")
    print(f"   Matches with rankings: {matches_with_rankings:,} / {len(df):,} ({coverage_pct:.1f}%)")
    print("   Coverage includes 2024 data from current rankings file")

    return df


# =============================================================================
# ENHANCED Z-SCORE NORMALIZATION
# =============================================================================


def compute_z_score_normalization(df: pd.DataFrame, include_surface: bool = False) -> pd.DataFrame:
    """
    Compute z-score normalization for metrics.

    Args:
        df: DataFrame with metrics to normalize
        include_surface: If True, normalize within (year, surface) groups
                        If False, normalize within year groups only

    Returns:
        DataFrame with z-score columns added
    """
    normalization_type = "year+surface" if include_surface else "year-only"
    print(f"\n=== COMPUTING Z-SCORE NORMALIZATION ({normalization_type.upper()}) ===")

    # Metrics to normalize
    metrics_to_normalize = [
        "ace_rate",
        "df_rate",
        "first_serve_pct",
        "first_serve_win_pct",
        "second_serve_win_pct",
        "break_point_save_pct",
        "return_win_pct",
        "service_dominance",
    ]

    # Group by year (and surface if specified)
    groupby_cols = ["year"]
    suffix = "_z_year"

    if include_surface:
        groupby_cols.append("surface")
        suffix = "_z_year_surface"

    print(f"📊 Normalizing within groups: {', '.join(groupby_cols)}")

    # Compute z-scores
    for metric in metrics_to_normalize:
        if metric in df.columns:
            z_score_col = f"{metric}{suffix}"

            # Calculate group statistics
            group_stats = df.groupby(groupby_cols)[metric].agg(["mean", "std"]).reset_index()

            # Merge back to main dataframe
            df = df.merge(group_stats, on=groupby_cols, how="left", suffixes=("", f"_{metric}_stats"))

            # Calculate z-scores
            mean_col = "mean"
            std_col = "std"

            # Handle cases where std = 0 or NaN
            df[z_score_col] = np.where((df[std_col] > 0) & df[metric].notna(), (df[metric] - df[mean_col]) / df[std_col], np.nan)

            # Clean up temporary columns
            df.drop([mean_col, std_col], axis=1, inplace=True)

            # Report results
            valid_z_scores = df[z_score_col].notna().sum()
            print(f"  ✅ {metric} → {z_score_col}: {valid_z_scores:,} valid z-scores")

    return df


def add_enhanced_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add both year-only and year+surface z-score normalizations.

    Args:
        df: DataFrame with derived metrics

    Returns:
        DataFrame with both normalization approaches added
    """
    print("\n=== ADDING ENHANCED NORMALIZATION ===")

    # Option A: Year-only z-scores
    df = compute_z_score_normalization(df, include_surface=False)

    # Option B: Year+surface z-scores
    df = compute_z_score_normalization(df, include_surface=True)

    print("✅ Enhanced normalization complete:")
    print("  Option A: Year-only z-scores (_z_year suffix)")
    print("  Option B: Year+surface z-scores (_z_year_surface suffix)")

    return df


if __name__ == "__main__":
    transform_to_player_match()
