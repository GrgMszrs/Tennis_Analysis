#!/usr/bin/env python3
"""
Tournament Normalization Demo
Demonstrates tournament name normalization and its impact on data matching.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
try:
    from data_pipeline.matching import run_matching_experiment
    from data_pipeline.standardization import load_cleaned_datasets, standardize_datasets
    from data_pipeline.tournament_normalization import normalize_tournament_name
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def demonstrate_tournament_normalization():
    """Show examples of tournament name normalization."""
    print("🎾 TOURNAMENT NORMALIZATION DEMO")
    print("=" * 80)

    print("\n🔄 Tournament Name Normalization Examples:")
    print("-" * 50)

    # Sample problematic tournament names from PBP data
    sample_tournaments = [
        "Gentlemen'sWimbledonSingles",
        "Men'sAustralianOpen",
        "BNPParibasOpen-ATPIndianWells",
        "SonyOpenTennis-ATPMiami",
        "MutualMadridOpen-ATPMadrid",
        "ItalianOpen-ATPRome",
        "Winston-SalemOpen-ATPWinston-Salem",
        "SkiStarSwedishOpen-ATPBastad",
        "http://atpworldtour.com/en/tournaments/wimbledon/540/overview",
        "https://www.atptour.com/en/tournaments/indian-wells/404/overview",
    ]

    for tournament in sample_tournaments:
        normalized = normalize_tournament_name(tournament)
        status = "✅ NORMALIZED" if tournament != normalized else "➡️  UNCHANGED"
        print(f"{status}: {tournament[:45]:<45} → {normalized}")

    return True


def run_matching_comparison(sample_size: int = 1000):
    """Compare matching performance with and without tournament normalization."""
    print("\n🔬 MATCHING PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"📊 Sample size: {sample_size:,} PBP records")

    # Load and prepare data
    print("\n📂 Loading datasets...")
    atp_matches, atp_pbp = load_cleaned_datasets()

    # Sample data for demo (to keep it fast) - use index-based sampling for raw data
    if len(atp_pbp) > sample_size:
        atp_pbp_sample = atp_pbp.sample(n=sample_size, random_state=42)
        print(f"   Sampled {len(atp_pbp_sample):,} PBP records for demo")
    else:
        atp_pbp_sample = atp_pbp
        print(f"   Using all {len(atp_pbp_sample):,} PBP records")

    # Sample ATP matches to keep it manageable
    atp_matches_sample = atp_matches.sample(n=min(5000, len(atp_matches)), random_state=42)
    print(f"   Using {len(atp_matches_sample):,} ATP match records")

    # Test 1: WITHOUT tournament normalization
    print("\n" + "=" * 60)
    print("TEST 1: WITHOUT TOURNAMENT NORMALIZATION")
    print("=" * 60)

    # Standardize without tournament normalization
    start_time = time.time()
    result_without = standardize_datasets(enable_tournament_normalization=False)
    standardization_time_without = time.time() - start_time

    atp_matches_std = result_without["atp_matches"]
    atp_pbp_std = result_without["atp_pbp"]

    # Filter to our sample using pbp_id for PBP and index for ATP matches
    atp_pbp_std_sample = atp_pbp_std[atp_pbp_std["pbp_id"].isin(atp_pbp_sample["pbp_id"])]

    # For ATP matches, we need to use a different approach since match_id is created during standardization
    # Let's just take the first N matches that were standardized to keep it simple
    atp_matches_std_sample = atp_matches_std.head(len(atp_matches_sample))

    # Run matching without tournament normalization
    start_time = time.time()
    results_without = run_matching_experiment(
        atp_pbp_std_sample,
        atp_matches_std_sample,
        strategies=["fuzzy"],  # Use fuzzy for speed in demo
        use_cache=False,  # Disable cache for fair comparison
        date_window_days=7,
        use_tournament_normalization=False,
    )
    matching_time_without = time.time() - start_time

    # Test 2: WITH tournament normalization
    print("\n" + "=" * 60)
    print("TEST 2: WITH TOURNAMENT NORMALIZATION")
    print("=" * 60)

    # Standardize WITH tournament normalization
    start_time = time.time()
    result_with = standardize_datasets(enable_tournament_normalization=True)
    standardization_time_with = time.time() - start_time

    atp_matches_norm = result_with["atp_matches"]
    atp_pbp_norm = result_with["atp_pbp"]

    # Filter to our sample using pbp_id for PBP
    atp_pbp_norm_sample = atp_pbp_norm[atp_pbp_norm["pbp_id"].isin(atp_pbp_sample["pbp_id"])]

    # For ATP matches, use the same approach
    atp_matches_norm_sample = atp_matches_norm.head(len(atp_matches_sample))

    # Run matching WITH tournament normalization
    start_time = time.time()
    results_with = run_matching_experiment(
        atp_pbp_norm_sample,
        atp_matches_norm_sample,
        strategies=["fuzzy"],  # Use fuzzy for speed in demo
        use_cache=False,  # Disable cache for fair comparison
        date_window_days=7,
        use_tournament_normalization=True,
    )
    matching_time_with = time.time() - start_time

    # Compare results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)

    print("\n📊 MATCHING RESULTS:")
    print(f"{'Metric':<30} {'Without Norm':<15} {'With Norm':<15} {'Improvement':<15}")
    print("-" * 75)

    matches_without = len(results_without)
    matches_with = len(results_with)
    match_rate_without = matches_without / len(atp_pbp_std_sample) * 100 if len(atp_pbp_std_sample) > 0 else 0
    match_rate_with = matches_with / len(atp_pbp_norm_sample) * 100 if len(atp_pbp_norm_sample) > 0 else 0

    print(f"{'Matches Found':<30} {matches_without:<15,} {matches_with:<15,} {matches_with - matches_without:+,}")
    print(f"{'Match Rate':<30} {match_rate_without:<15.1f}% {match_rate_with:<15.1f}% {match_rate_with - match_rate_without:+.1f}%")

    if len(results_without) > 0:
        avg_score_without = results_without["score"].mean()
    else:
        avg_score_without = 0

    if len(results_with) > 0:
        avg_score_with = results_with["score"].mean()
    else:
        avg_score_with = 0

    print(f"{'Avg Match Score':<30} {avg_score_without:<15.1f} {avg_score_with:<15.1f} {avg_score_with - avg_score_without:+.1f}")

    print("\n⏱️  PROCESSING TIME:")
    print(
        f"{'Standardization':<30} {standardization_time_without:<15.1f}s {standardization_time_with:<15.1f}s {standardization_time_with - standardization_time_without:+.1f}s"
    )
    print(
        f"{'Matching':<30} {matching_time_without:<15.1f}s {matching_time_with:<15.1f}s {matching_time_with - matching_time_without:+.1f}s"
    )

    total_time_without = standardization_time_without + matching_time_without
    total_time_with = standardization_time_with + matching_time_with
    print(f"{'Total Pipeline':<30} {total_time_without:<15.1f}s {total_time_with:<15.1f}s {total_time_with - total_time_without:+.1f}s")

    # Quality analysis
    print("\n🎯 QUALITY ANALYSIS:")

    if len(results_with) > 0 and len(results_without) > 0:
        # High confidence matches (score >= 90)
        high_conf_without = (results_without["score"] >= 90).sum()
        high_conf_with = (results_with["score"] >= 90).sum()

        print(f"{'High Confidence (≥90)':<30} {high_conf_without:<15,} {high_conf_with:<15,} {high_conf_with - high_conf_without:+,}")

        # Show score distribution
        print("\n📈 Score Distribution:")
        print(f"Without Normalization: {results_without['score'].describe().round(1).to_dict()}")
        print(f"With Normalization:    {results_with['score'].describe().round(1).to_dict()}")

    # Tournament consolidation analysis
    if "tourney_name_normalized" in atp_matches_norm.columns and "tny_name_normalized" in atp_pbp_norm.columns:
        print("\n🏆 TOURNAMENT CONSOLIDATION:")

        atp_orig_tournaments = atp_matches_std_sample["tourney_name"].nunique()
        atp_norm_tournaments = atp_matches_norm_sample["tourney_name_normalized"].nunique()
        pbp_orig_tournaments = atp_pbp_std_sample["tny_name"].nunique() if "tny_name" in atp_pbp_std_sample.columns else 0
        pbp_norm_tournaments = atp_pbp_norm_sample["tny_name_normalized"].nunique()

        print(
            f"{'ATP Tournaments':<30} {atp_orig_tournaments:<15,} {atp_norm_tournaments:<15,} {atp_norm_tournaments - atp_orig_tournaments:+,}"
        )
        print(
            f"{'PBP Tournaments':<30} {pbp_orig_tournaments:<15,} {pbp_norm_tournaments:<15,} {pbp_norm_tournaments - pbp_orig_tournaments:+,}"
        )

    return {
        "matches_without": matches_without,
        "matches_with": matches_with,
        "match_rate_without": match_rate_without,
        "match_rate_with": match_rate_with,
        "time_without": total_time_without,
        "time_with": total_time_with,
        "improvement_percentage": (match_rate_with - match_rate_without),
        "improvement_absolute": matches_with - matches_without,
    }


def demonstrate_tournament_filtering():
    """Show how tournament normalization improves filtering."""
    print("\n🔍 TOURNAMENT FILTERING DEMONSTRATION")
    print("=" * 80)

    # Load sample data
    atp_matches, atp_pbp = load_cleaned_datasets()

    # Take a small sample for demonstration
    atp_pbp_sample = atp_pbp.head(100)

    print(f"📊 Analyzing tournament filtering for {len(atp_pbp_sample)} PBP records...")

    # Show raw tournament names
    if "tny_name" in atp_pbp_sample.columns:
        pbp_tournaments = atp_pbp_sample["tny_name"].value_counts().head(10)
        print("\n📋 Top PBP tournament names (raw):")
        for tournament, count in pbp_tournaments.items():
            print(f"   {count:2d}× {tournament[:60]}")

    # Show ATP tournament names
    if "tourney_name" in atp_matches.columns:
        atp_tournaments = atp_matches["tourney_name"].value_counts().head(10)
        print("\n📋 Top ATP tournament names:")
        for tournament, count in atp_tournaments.items():
            print(f"   {count:2d}× {tournament[:60]}")

    # Standardize with tournament normalization
    result = standardize_datasets(enable_tournament_normalization=True)
    atp_matches_norm = result["atp_matches"]
    atp_pbp_norm = result["atp_pbp"]

    # Filter to our sample
    atp_pbp_norm_sample = atp_pbp_norm[atp_pbp_norm["pbp_id"].isin(atp_pbp_sample["pbp_id"])]

    # Show normalized tournament names
    if "tny_name_normalized" in atp_pbp_norm_sample.columns:
        pbp_tournaments_norm = atp_pbp_norm_sample["tny_name_normalized"].value_counts().head(10)
        print("\n📋 Top PBP tournament names (normalized):")
        for tournament, count in pbp_tournaments_norm.items():
            print(f"   {count:2d}× {tournament[:60]}")

    if "tourney_name_normalized" in atp_matches_norm.columns:
        atp_tournaments_norm = atp_matches_norm["tourney_name_normalized"].value_counts().head(10)
        print("\n📋 Top ATP tournament names (normalized):")
        for tournament, count in atp_tournaments_norm.items():
            print(f"   {count:2d}× {tournament[:60]}")

    # Calculate overlap
    if "tny_name_normalized" in atp_pbp_norm_sample.columns and "tourney_name_normalized" in atp_matches_norm.columns:
        pbp_norm_set = set(atp_pbp_norm_sample["tny_name_normalized"].dropna().str.lower())
        atp_norm_set = set(atp_matches_norm["tourney_name_normalized"].dropna().str.lower())

        overlap = pbp_norm_set.intersection(atp_norm_set)
        pbp_only = pbp_norm_set - atp_norm_set
        atp_only = atp_norm_set - pbp_norm_set

        print("\n🎯 Tournament Overlap Analysis (Normalized):")
        print(f"   Shared tournaments: {len(overlap):,}")
        print(f"   PBP-only tournaments: {len(pbp_only):,}")
        print(f"   ATP-only tournaments: {len(atp_only):,}")
        print(f"   Overlap percentage: {len(overlap) / max(len(pbp_norm_set), 1) * 100:.1f}%")

        if overlap:
            print("\n📋 Sample shared tournaments:")
            for tournament in sorted(list(overlap))[:5]:
                print(f"   • {tournament}")


def main():
    """Run the complete tournament normalization demo."""
    print("🎾 TENNIS ANALYSIS - TOURNAMENT NORMALIZATION DEMO")
    print("=" * 80)
    print("Demonstrating the performance benefits of tournament name normalization")
    print("in the tennis data matching pipeline.")
    print()

    try:
        # 1. Show tournament normalization examples
        demonstrate_tournament_normalization()

        # 2. Show tournament filtering improvements
        demonstrate_tournament_filtering()

        # 3. Run performance comparison
        print("\n" + "🚀 PERFORMANCE BENCHMARK")
        print("=" * 80)
        print("Running side-by-side comparison of matching performance...")

        results = run_matching_comparison(sample_size=500)  # Smaller sample for demo

        # 4. Summary and recommendations
        print("\n" + "💡 SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        if results["improvement_percentage"] > 0:
            print("✅ Tournament normalization IMPROVED matching performance!")
            print(f"   • Match rate increased by {results['improvement_percentage']:+.1f}% ")
            print(f"   • Found {results['improvement_absolute']:+,} additional matches")

        elif results["improvement_percentage"] < 0:
            print("⚠️  Tournament normalization had mixed results:")
            print(f"   • Match rate changed by {results['improvement_percentage']:+.1f}%")
            print("   • But may have improved match quality and processing efficiency")

        else:
            print("➡️  Tournament normalization showed neutral impact on match rate")
            print("   • But likely improved match quality and data consistency")

        time_diff = results["time_with"] - results["time_without"]
        if time_diff < 0:
            print(f"⚡ Processing time IMPROVED by {abs(time_diff):.1f} seconds")
        elif time_diff > 0:
            print(f"⏱️  Processing time increased by {time_diff:.1f} seconds (due to normalization overhead)")
        else:
            print("⏱️  Processing time was similar")

        print("\n🎯 WHEN TO USE TOURNAMENT NORMALIZATION:")
        print("   ✅ When PBP data has messy/inconsistent tournament names")
        print("   ✅ When you want maximum match accuracy")
        print("   ✅ When data quality is more important than speed")
        print("   ✅ For production pipelines with data governance requirements")

        print("\n🚀 TO ENABLE IN MAIN PIPELINE:")
        print("   python main.py --tournament-normalization")
        print("   python main.py --phase 1 --tournament-normalization --force")
        print("   python main.py --phase 3 --tournament-normalization")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
