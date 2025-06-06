#!/usr/bin/env python3
"""
Tennis Analysis - Data Overlap Investigation
Diagnostic script to understand why PBP and ATP match data have limited overlap.
"""

from typing import Dict

import pandas as pd


def create_player_key(player1: str, player2: str) -> frozenset:
    """Create order-invariant key from two player names."""
    p1 = str(player1).strip().lower() if player1 and str(player1) != "nan" else ""
    p2 = str(player2).strip().lower() if player2 and str(player2) != "nan" else ""

    if p1 and p2:
        return frozenset([p1, p2])
    return frozenset()


def analyze_temporal_overlap(pbp_df: pd.DataFrame, match_df: pd.DataFrame):
    """Analyze temporal overlap between datasets."""
    print("\n📅 TEMPORAL OVERLAP ANALYSIS")
    print("=" * 50)

    # Convert dates
    pbp_df["parsed_date"] = pd.to_datetime(pbp_df["parsed_date"], errors="coerce")
    # tourney_date is in format YYYYMMDD, so we need to convert it to datetime
    match_df["parsed_tourney_date"] = pd.to_datetime(match_df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    # Get date ranges
    pbp_start = pbp_df["parsed_date"].min()
    pbp_end = pbp_df["parsed_date"].max()
    match_start = match_df["parsed_tourney_date"].min()
    match_end = match_df["parsed_tourney_date"].max()

    print(f"PBP Date Range:   {pbp_start.date()} to {pbp_end.date()}")
    print(f"Match Date Range: {match_start.date()} to {match_end.date()}")

    # Find overlap period
    overlap_start = max(pbp_start, match_start)
    overlap_end = min(pbp_end, match_end)

    if overlap_start <= overlap_end:
        print(f"Overlap Period:   {overlap_start.date()} to {overlap_end.date()}")

        # Count records in overlap period
        pbp_overlap = pbp_df[(pbp_df["parsed_date"] >= overlap_start) & (pbp_df["parsed_date"] <= overlap_end)]
        match_overlap = match_df[(match_df["parsed_tourney_date"] >= overlap_start) & (match_df["parsed_tourney_date"] <= overlap_end)]

        print(f"PBP in overlap:   {len(pbp_overlap):,} records")
        print(f"Matches in overlap: {len(match_overlap):,} records")

        return pbp_overlap, match_overlap
    else:
        print("❌ NO TEMPORAL OVERLAP FOUND!")
        return pd.DataFrame(), pd.DataFrame()


def analyze_tournament_scope(pbp_df: pd.DataFrame, match_df: pd.DataFrame):
    """Analyze tournament scope differences."""
    print("\n🏆 TOURNAMENT SCOPE ANALYSIS")
    print("=" * 50)

    # Get unique tournament names
    pbp_tournaments = set(pbp_df["tny_name"].dropna().str.lower())
    match_tournaments = set(match_df["tourney_name"].dropna().str.lower())

    print(f"PBP tournaments:   {len(pbp_tournaments):,}")
    print(f"Match tournaments: {len(match_tournaments):,}")

    # Find overlaps
    tournament_overlap = pbp_tournaments.intersection(match_tournaments)
    pbp_only = pbp_tournaments - match_tournaments
    match_only = match_tournaments - pbp_tournaments

    print(f"Shared tournaments: {len(tournament_overlap):,}")
    print(f"PBP-only tournaments: {len(pbp_only):,}")
    print(f"Match-only tournaments: {len(match_only):,}")

    # Sample tournaments
    if tournament_overlap:
        print("\n📋 Sample shared tournaments:")
        for t in list(tournament_overlap)[:5]:
            print(f"   {t}")

    if pbp_only:
        print("\n📋 Sample PBP-only tournaments:")
        for t in list(pbp_only)[:5]:
            print(f"   {t}")


def analyze_player_overlap(pbp_df: pd.DataFrame, match_df: pd.DataFrame, sample_size: int = 1000):
    """Analyze player name overlap between datasets."""
    print(f"\n👥 PLAYER OVERLAP ANALYSIS (Sample: {sample_size:,})")
    print("=" * 50)

    # Sample data for faster analysis
    pbp_sample = pbp_df.sample(n=min(sample_size, len(pbp_df)), random_state=42)
    match_sample = match_df.sample(n=min(sample_size, len(match_df)), random_state=42)

    # Extract player names
    pbp_players = set()
    for _, row in pbp_sample.iterrows():
        if row.get("server1") and str(row["server1"]) != "nan":
            pbp_players.add(str(row["server1"]).strip().lower())
        if row.get("server2") and str(row["server2"]) != "nan":
            pbp_players.add(str(row["server2"]).strip().lower())

    match_players = set()
    for _, row in match_sample.iterrows():
        if row.get("winner_name") and str(row["winner_name"]) != "nan":
            match_players.add(str(row["winner_name"]).strip().lower())
        if row.get("loser_name") and str(row["loser_name"]) != "nan":
            match_players.add(str(row["loser_name"]).strip().lower())

    print(f"PBP players:      {len(pbp_players):,}")
    print(f"Match players:    {len(match_players):,}")

    # Find overlaps
    player_overlap = pbp_players.intersection(match_players)
    pbp_only = pbp_players - match_players
    match_only = match_players - pbp_players

    print(f"Shared players:   {len(player_overlap):,}")
    print(f"PBP-only players: {len(pbp_only):,}")
    print(f"Match-only players: {len(match_only):,}")

    overlap_percentage = len(player_overlap) / len(pbp_players) * 100 if pbp_players else 0
    print(f"Player overlap:   {overlap_percentage:.1f}%")

    # Sample players
    if player_overlap:
        print("\n📋 Sample shared players:")
        for p in list(player_overlap)[:5]:
            print(f"   {p}")


def analyze_combination_overlap(pbp_df: pd.DataFrame, match_df: pd.DataFrame, sample_size: int = 1000):
    """Analyze player combination overlap."""
    print(f"\n🎾 PLAYER COMBINATION OVERLAP (Sample: {sample_size:,})")
    print("=" * 50)

    # Sample data
    pbp_sample = pbp_df.sample(n=min(sample_size, len(pbp_df)), random_state=42)
    match_sample = match_df.sample(n=min(sample_size, len(match_df)), random_state=42)

    # Extract player combinations
    pbp_combinations = set()
    for _, row in pbp_sample.iterrows():
        key = create_player_key(row.get("server1"), row.get("server2"))
        if key:
            pbp_combinations.add(key)

    match_combinations = set()
    for _, row in match_sample.iterrows():
        key = create_player_key(row.get("winner_name"), row.get("loser_name"))
        if key:
            match_combinations.add(key)

    print(f"PBP combinations:   {len(pbp_combinations):,}")
    print(f"Match combinations: {len(match_combinations):,}")

    # Find overlaps
    combination_overlap = pbp_combinations.intersection(match_combinations)
    pbp_only = pbp_combinations - match_combinations

    print(f"Shared combinations: {len(combination_overlap):,}")
    print(f"PBP-only combinations: {len(pbp_only):,}")

    overlap_percentage = len(combination_overlap) / len(pbp_combinations) * 100 if pbp_combinations else 0
    print(f"Combination overlap: {overlap_percentage:.1f}%")

    return combination_overlap, pbp_only


def recommend_solutions(overlap_analysis: Dict):
    """Recommend solutions based on analysis results."""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)

    print("Based on the data overlap analysis:")

    print("\n🎯 Immediate Actions:")
    print("1. ✅ Order-invariant matching is implemented")
    print("2. 🔍 Expand sample sizes to find more overlaps")
    print("3. 📅 Check if date parsing/formatting differs between datasets")

    print("\n📈 Data Engineering Improvements:")
    print("1. 🗓️  Standardize date formats across datasets")
    print("2. 🏆 Normalize tournament names (fuzzy matching)")
    print("3. 👥 Implement player name standardization")
    print("4. 🔄 Consider expanding ATP match data to include more tournament types")

    print("\n🚀 Matching Algorithm Enhancements:")
    print("1. 🎯 Add retrieval prompts for embedding model")
    print("2. 📝 Expand context for short player names")
    print("3. 🗓️  Implement date-range fuzzy matching")
    print("4. 🏆 Add tournament name similarity scoring")


def investigate_data_overlap(sample_size: int = 2000):
    """Main investigation function."""
    print("🔍 TENNIS DATA OVERLAP INVESTIGATION")
    print("=" * 60)

    try:
        # Load datasets
        print("📂 Loading datasets...")
        pbp_df = pd.read_csv("data/cleaned_refactored/atp_pbp_cleaned.csv", nrows=sample_size)
        match_df = pd.read_csv("data/cleaned_refactored/atp_matches_cleaned.csv", nrows=sample_size)

        print(f"   Loaded {len(pbp_df):,} PBP records")
        print(f"   Loaded {len(match_df):,} match records")

        # Run analyses
        pbp_overlap, match_overlap = analyze_temporal_overlap(pbp_df, match_df)
        analyze_tournament_scope(pbp_df, match_df)
        analyze_player_overlap(pbp_df, match_df, sample_size=sample_size // 2)
        combination_overlap, pbp_only = analyze_combination_overlap(pbp_df, match_df, sample_size=sample_size // 2)

        # Summary
        overlap_analysis = {
            "temporal_overlap": len(pbp_overlap) > 0 and len(match_overlap) > 0,
            "combination_overlap_count": len(combination_overlap),
            "pbp_only_combinations": len(pbp_only),
        }

        recommend_solutions(overlap_analysis)

        print("\n✅ Investigation completed!")

    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure you've run the data pipeline first:")
        print("   python main.py --phase 1")


if __name__ == "__main__":
    investigate_data_overlap(sample_size=60000)
