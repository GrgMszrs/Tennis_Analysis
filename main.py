#!/usr/bin/env python3
"""
Tennis Era Analysis - Main Pipeline
Clean entry point for the tennis data processing and analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project modules to path
sys.path.append(".")

from analysis.era_analysis import generate_era_analysis_report, load_player_match_data
from data_pipeline.caching import clear_embedding_cache, clear_matching_results_cache, precompute_embeddings_for_datasets
from data_pipeline.matching import run_matching_experiment
from data_pipeline.standardization import standardize_datasets
from data_pipeline.transformation import transform_to_player_match
from utils.helpers import ensure_directory_exists


def run_standardization(force: bool = False, tournament_normalization: bool = True) -> bool:
    """
    Run Phase 1: Data Standardization.

    Args:
        force: Force reprocessing even if output files exist
        tournament_normalization: Enable tournament name normalization

    Returns:
        True if successful, False otherwise
    """
    try:
        print("🎾 PHASE 1: DATA STANDARDIZATION")
        print("=" * 50)

        # Check if standardized files already exist
        standardized_files = ["data/cleaned_refactored/atp_matches_standardized.csv", "data/cleaned_refactored/atp_pbp_standardized.csv"]

        if not force and all(Path(f).exists() for f in standardized_files):
            print("   ⚡ Standardized files already exist, skipping...")
            print("   Use --force to reprocess")

            # Load existing files to get summary
            import pandas as pd

            atp_matches = pd.read_csv(standardized_files[0])
            atp_pbp = pd.read_csv(standardized_files[1])

            print("\n✅ Phase 1 completed (cached)!")
            print(f"   ATP Matches: {len(atp_matches):,} rows")
            print(f"   ATP PBP: {len(atp_pbp):,} rows")
            if tournament_normalization:
                print("   ⚠️  Note: Tournament normalization requested but using cached files")
                print("   Use --force to reprocess with tournament normalization")
            return True

        result = standardize_datasets(enable_tournament_normalization=tournament_normalization)

        print("\n✅ Phase 1 completed successfully!")
        print(f"   ATP Matches: {result['summary']['atp_matches_count']:,} rows")
        print(f"   ATP PBP: {result['summary']['atp_pbp_count']:,} rows")

        return True

    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False


def run_transformation(force: bool = False) -> bool:
    """
    Run Phase 2: Data Transformation.

    Args:
        force: Force reprocessing even if output files exist

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n🎾 PHASE 2: DATA TRANSFORMATION")
        print("=" * 50)

        # Check if enhanced transformed file already exists (per file naming strategy)
        enhanced_file = "data/cleaned_refactored/atp_player_match_enhanced.csv"

        if not force and Path(enhanced_file).exists():
            print("   ⚡ Enhanced transformed file already exists, skipping...")
            print("   Use --force to reprocess")

            # Load existing file to get summary
            import pandas as pd

            player_data = pd.read_csv(enhanced_file)

            print("\n✅ Phase 2 completed (cached)!")
            print(f"   Player-match rows: {len(player_data):,}")
            print(f"   Unique players: {player_data['player_name'].nunique():,}")
            print(f"   Unique matches: {player_data['match_id'].nunique():,}")

            # Show enhanced features if available
            z_score_cols = [col for col in player_data.columns if "_z_" in col]
            ranking_cols = [col for col in player_data.columns if "historical" in col]
            if z_score_cols or ranking_cols:
                print(f"   Enhanced features: {len(z_score_cols)} z-score + {len(ranking_cols)} ranking columns")

            return True

        result = transform_to_player_match()

        print("\n✅ Phase 2 completed successfully!")
        print(f"   Player-match rows: {result['summary']['total_rows']:,}")
        print(f"   Unique players: {result['summary']['unique_players']:,}")
        print(f"   Unique matches: {result['summary']['unique_matches']:,}")

        return True

    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        return False


def run_matching(use_cache: bool = True, precompute: bool = False, tournament_normalization: bool = True) -> bool:
    """
    Run Phase 3: PBP Matching with Optimized Caching.

    Args:
        use_cache: Whether to use cached results if available
        precompute: Whether to precompute embeddings for better performance
        tournament_normalization: Whether to use tournament normalization for better matching

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n🎾 PHASE 3: PBP MATCHING (OPTIMIZED)")
        print("=" * 50)

        # Load standardized data
        from data_pipeline.standardization import load_standardized_datasets

        atp_matches, atp_pbp = load_standardized_datasets()

        # Optional: Precompute embeddings for maximum speed
        if precompute:
            print("🔄 Precomputing embeddings for maximum performance...")
            precompute_embeddings_for_datasets(atp_pbp, atp_matches)

        # Run matching experiment with caching
        results = run_matching_experiment(
            atp_pbp,
            atp_matches,
            strategies=["embedding"],  # Test both strategies
            use_cache=use_cache,
            date_window_days=7,  # Optimized from analysis: 79.1% vs 59.7% match rate
            use_tournament_normalization=tournament_normalization,
        )

        print("\n✅ Phase 3 completed successfully!")
        print(f"   Matching results: {len(results):,} matches found")

        if use_cache:
            print("💾 Results cached for future runs")

        return True

    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        return False


def run_analysis() -> bool:
    """
    Run Phase 4: Era Analysis.

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n🎾 PHASE 4: ERA ANALYSIS")
        print("=" * 50)

        # Load player-match data
        player_data = load_player_match_data()

        # Generate comprehensive analysis report
        report = generate_era_analysis_report(player_data)

        print("\n✅ Phase 4 completed successfully!")
        print(f"   Analysis report generated with {len(report)} sections")

        return True

    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        return False


def run_full_pipeline(
    skip_matching: bool = False,
    force: bool = False,
    use_cache: bool = True,
    precompute: bool = False,
    tournament_normalization: bool = True,
) -> bool:
    """
    Run the complete tennis era analysis pipeline with optimized caching.

    Args:
        skip_matching: Whether to skip the matching phase
        force: Force reprocessing even if output files exist
        use_cache: Whether to use cached results for matching
        precompute: Whether to precompute embeddings for better performance
        tournament_normalization: Whether to use tournament normalization for better matching

    Returns:
        True if successful, False otherwise
    """
    print("🎾 TENNIS ERA ANALYSIS - FULL PIPELINE")
    print("=" * 60)

    # Phase 1: Standardization
    if not run_standardization(force=force, tournament_normalization=tournament_normalization):
        return False

    # Phase 2: Transformation
    if not run_transformation(force=force):
        return False

    # Phase 3: Matching (optional)
    if not skip_matching:
        if not run_matching(use_cache=use_cache, precompute=precompute, tournament_normalization=tournament_normalization):
            print("⚠️ Matching phase failed, but continuing with analysis...")

    # Phase 4: Analysis
    if not run_analysis():
        return False

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ All phases completed")
    print("📊 Analysis-ready datasets available in data/cleaned_refactored/")
    print("📈 Era analysis report generated")
    if use_cache:
        print("💾 Caches optimized for future runs")

    return True


def main():
    """Main entry point for the tennis era analysis pipeline."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Tennis Era Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline with tournament normalization
  python main.py --phase 1          # Run only standardization
  python main.py --phase 2          # Run only transformation
  python main.py --phase 3          # Run only matching with caching
  python main.py --phase 4          # Run only analysis
  python main.py --skip-matching    # Skip PBP matching phase
  python main.py --precompute       # Precompute embeddings for max speed
  python main.py --no-cache         # Disable caching (slower but fresh)
  python main.py --clear-cache all  # Clear all caches
  python main.py --no-tournament-normalization  # Disable tournament name normalization (enabled by default)
        """,
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific phase only (1=standardization, 2=transformation, 3=matching, 4=analysis)",
    )

    parser.add_argument("--skip-matching", action="store_true", help="Skip the PBP matching phase")

    parser.add_argument("--force", action="store_true", help="Force reprocessing even if output files exist")

    parser.add_argument("--no-cache", action="store_true", help="Disable caching for matching (slower but fresh results)")

    parser.add_argument("--precompute", action="store_true", help="Precompute embeddings for maximum matching performance")

    parser.add_argument("--clear-cache", choices=["embeddings", "results", "all"], help="Clear specified cache before running")

    parser.add_argument(
        "--no-tournament-normalization", action="store_true", help="Disable tournament name normalization (enabled by default)"
    )

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")

    args = parser.parse_args()

    # Set up logging, disabled for now
    # logger = setup_logging(args.log_level)

    # Ensure output directories exist
    ensure_directory_exists("data/cleaned_refactored")
    ensure_directory_exists("data/output")
    ensure_directory_exists("data/cache")

    # Handle cache management options
    if args.clear_cache:
        if args.clear_cache == "embeddings":
            clear_embedding_cache()
        elif args.clear_cache == "results":
            clear_matching_results_cache()
        elif args.clear_cache == "all":
            clear_embedding_cache()
            clear_matching_results_cache()
        print("Cache cleared successfully")

    # Determine caching settings
    use_cache = not args.no_cache
    precompute = args.precompute
    tournament_normalization = not args.no_tournament_normalization  # Inverted logic since it's enabled by default

    # Run based on arguments
    success = False

    if args.phase:
        if args.phase == 1:
            success = run_standardization(force=args.force, tournament_normalization=tournament_normalization)
        elif args.phase == 2:
            success = run_transformation(force=args.force)
        elif args.phase == 3:
            success = run_matching(use_cache=use_cache, precompute=precompute, tournament_normalization=tournament_normalization)
        elif args.phase == 4:
            success = run_analysis()

    else:
        # Run full pipeline
        success = run_full_pipeline(
            skip_matching=args.skip_matching,
            force=args.force,
            use_cache=use_cache,
            precompute=precompute,
            tournament_normalization=tournament_normalization,
        )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
