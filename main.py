#!/usr/bin/env python3
"""
Tennis Era Analysis - Main Pipeline
Clean entry point for the tennis data processing and analysis pipeline.
"""

import argparse
import sys
from typing import Optional

# Add project modules to path
sys.path.append(".")

from analysis.era_analysis import generate_era_analysis_report, load_player_match_data
from data_pipeline.matching import run_matching_experiment
from data_pipeline.standardization import standardize_datasets
from data_pipeline.transformation import transform_to_player_match
from utils.helpers import ensure_directory_exists


def run_standardization() -> bool:
    """
    Run Phase 1: Data Standardization.

    Returns:
        True if successful, False otherwise
    """
    try:
        print("🎾 PHASE 1: DATA STANDARDIZATION")
        print("=" * 50)

        result = standardize_datasets()

        print("\n✅ Phase 1 completed successfully!")
        print(f"   ATP Matches: {result['summary']['atp_matches_count']:,} rows")
        print(f"   ATP PBP: {result['summary']['atp_pbp_count']:,} rows")

        return True

    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False


def run_transformation() -> bool:
    """
    Run Phase 2: Data Transformation.

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n🎾 PHASE 2: DATA TRANSFORMATION")
        print("=" * 50)

        result = transform_to_player_match()

        print("\n✅ Phase 2 completed successfully!")
        print(f"   Player-match rows: {result['summary']['total_rows']:,}")
        print(f"   Unique players: {result['summary']['unique_players']:,}")
        print(f"   Unique matches: {result['summary']['unique_matches']:,}")

        return True

    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        return False


def run_matching(openai_api_key: Optional[str] = None) -> bool:
    """
    Run Phase 3: PBP Matching (Optional).

    Args:
        openai_api_key: OpenAI API key for LLM matching (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n🎾 PHASE 3: PBP MATCHING (OPTIONAL)")
        print("=" * 50)

        # Load standardized data
        from data_pipeline.standardization import load_raw_datasets

        atp_matches, atp_pbp = load_raw_datasets()

        # Run matching experiment
        results = run_matching_experiment(atp_pbp, atp_matches, openai_api_key)

        print("\n✅ Phase 3 completed successfully!")
        print(f"   Matching results: {len(results):,} matches found")

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


def run_full_pipeline(openai_api_key: Optional[str] = None, skip_matching: bool = False) -> bool:
    """
    Run the complete tennis era analysis pipeline.

    Args:
        openai_api_key: OpenAI API key for LLM matching (optional)
        skip_matching: Whether to skip the matching phase

    Returns:
        True if successful, False otherwise
    """
    print("🎾 TENNIS ERA ANALYSIS - FULL PIPELINE")
    print("=" * 60)

    # Phase 1: Standardization
    if not run_standardization():
        return False

    # Phase 2: Transformation
    if not run_transformation():
        return False

    # Phase 3: Matching (optional)
    if not skip_matching:
        if not run_matching(openai_api_key):
            print("⚠️ Matching phase failed, but continuing with analysis...")

    # Phase 4: Analysis
    if not run_analysis():
        return False

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ All phases completed")
    print("📊 Analysis-ready datasets available in data/cleaned_refactored/")
    print("📈 Era analysis report generated")

    return True


def run_test_pipeline() -> bool:
    """
    Run a quick test of the pipeline with minimal processing.

    Returns:
        True if successful, False otherwise
    """
    print("🎾 TENNIS ERA ANALYSIS - TEST MODE")
    print("=" * 50)

    try:
        # Test Phase 1
        print("\n🧪 Testing Phase 1: Standardization...")
        result = standardize_datasets()
        print(f"   ✅ Phase 1 test passed: {result['summary']['atp_matches_count']:,} matches")

        # Test Phase 2
        print("\n🧪 Testing Phase 2: Transformation...")
        result = transform_to_player_match()
        print(f"   ✅ Phase 2 test passed: {result['summary']['total_rows']:,} player-match rows")

        print("\n✅ TEST PIPELINE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ Test pipeline failed: {e}")
        return False


def main():
    """Main entry point for the tennis era analysis pipeline."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Tennis Era Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --test             # Run test pipeline
  python main.py --phase 1          # Run only standardization
  python main.py --phase 2          # Run only transformation
  python main.py --phase 4          # Run only analysis
  python main.py --skip-matching    # Skip PBP matching phase
        """,
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific phase only (1=standardization, 2=transformation, 3=matching, 4=analysis)",
    )

    parser.add_argument("--test", action="store_true", help="Run test pipeline with minimal processing")

    parser.add_argument("--skip-matching", action="store_true", help="Skip the PBP matching phase")

    parser.add_argument("--openai-key", type=str, help="OpenAI API key for LLM matching")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")

    args = parser.parse_args()

    # Set up logging, disabled for now
    # logger = setup_logging(args.log_level)

    # Ensure output directories exist
    ensure_directory_exists("data/cleaned_refactored")
    ensure_directory_exists("data/output")

    # Run based on arguments
    success = False

    if args.test:
        success = run_test_pipeline()

    elif args.phase:
        if args.phase == 1:
            success = run_standardization()
        elif args.phase == 2:
            success = run_transformation()
        elif args.phase == 3:
            success = run_matching(args.openai_key)
        elif args.phase == 4:
            success = run_analysis()

    else:
        # Run full pipeline
        success = run_full_pipeline(openai_api_key=args.openai_key, skip_matching=args.skip_matching)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
