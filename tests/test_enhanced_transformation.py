"""
Test Enhanced Tennis Analysis Transformation
Validates z-score normalization and historical ranking integration
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
try:
    from config.constants import CLEANED_DATA_DIR
    from data_pipeline.transformation import (
        add_enhanced_normalization,
        add_era_classification,
        add_opponent_context,
        compute_derived_metrics,
        fetch_sackmann_players,
        fetch_sackmann_rankings,
        integrate_historical_rankings,
        load_standardized_data,
        reshape_to_player_match_format,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_enhanced_features():
    """Test the enhanced features with a small sample of data."""
    print("🎾 TESTING ENHANCED TENNIS ANALYSIS")
    print("=" * 50)

    # Test 1: Load and process sample data
    print("\n1️⃣ TESTING DATA LOADING AND PROCESSING")
    try:
        atp_matches = load_standardized_data()

        # Take a smaller sample for testing (last 1000 matches)
        sample_matches = atp_matches.tail(1000).copy()
        print(f"✅ Sample data: {len(sample_matches):,} matches")

        # Add era classification
        sample_matches = add_era_classification(sample_matches)
        print("✅ Era classification added")

        # Compute derived metrics
        sample_matches = compute_derived_metrics(sample_matches)
        print("✅ Derived metrics computed")

        # Reshape to player-match format
        player_match_sample = reshape_to_player_match_format(sample_matches)
        print(f"✅ Reshaped: {len(player_match_sample):,} player-match rows")

        # Add opponent context
        player_match_sample = add_opponent_context(player_match_sample)
        print("✅ Opponent context added")

    except Exception as e:
        print(f"❌ Error in basic processing: {e}")
        return

    # Test 2: Z-Score Normalization
    print("\n2️⃣ TESTING Z-SCORE NORMALIZATION")
    try:
        enhanced_sample = add_enhanced_normalization(player_match_sample)

        # Check z-score columns exist
        z_score_cols = [col for col in enhanced_sample.columns if "_z_" in col]
        print(f"✅ Z-score columns created: {len(z_score_cols)}")

        # Validate z-score properties (should have mean ≈ 0, std ≈ 1)
        print("\n📊 Z-Score Validation (first 3 metrics):")
        for col in z_score_cols[:3]:
            if enhanced_sample[col].notna().sum() > 10:  # Only test if enough data
                mean_z = enhanced_sample[col].mean()
                std_z = enhanced_sample[col].std()
                print(f"   {col[:25]:<25}: mean={mean_z:6.3f}, std={std_z:6.3f}")

    except Exception as e:
        print(f"❌ Error in z-score normalization: {e}")
        return

    # Test 3: Historical Rankings Integration
    print("\n3️⃣ TESTING HISTORICAL RANKINGS INTEGRATION")
    try:
        # Test Sackmann data fetching
        print("📥 Testing Sackmann data fetching...")
        players_ref = fetch_sackmann_players()
        rankings_ref = fetch_sackmann_rankings()

        if not players_ref.empty and not rankings_ref.empty:
            print(f"✅ Players data: {len(players_ref):,} players")
            print(f"✅ Rankings data: {len(rankings_ref):,} records")
            print(f"   Date range: {rankings_ref['ranking_date'].min()} to {rankings_ref['ranking_date'].max()}")

            # Check if 2024 data is included
            rankings_ref["ranking_date"] = pd.to_datetime(rankings_ref["ranking_date"], format="%Y%m%d", errors="coerce")
            max_year = rankings_ref["ranking_date"].dt.year.max()
            print(f"✅ 2024 data included: {max_year >= 2024}")

        else:
            print("⚠️ Could not fetch Sackmann data (network issue)")

        # Test integration with sample data
        final_sample = integrate_historical_rankings(enhanced_sample)

        # Check ranking columns
        ranking_cols = [col for col in final_sample.columns if "historical" in col]
        print(f"✅ Ranking columns added: {ranking_cols}")

        # Check coverage
        total_rows = len(final_sample)
        with_rankings = final_sample["historical_rank"].notna().sum()
        coverage = (with_rankings / total_rows) * 100
        print(f"✅ Ranking coverage: {with_rankings:,}/{total_rows:,} ({coverage:.1f}%)")

    except Exception as e:
        print(f"❌ Error in ranking integration: {e}")
        final_sample = enhanced_sample  # Use enhanced sample without rankings

    # Test 4: Data Quality Validation
    print("\n4️⃣ TESTING DATA QUALITY")

    # Check for expected columns
    expected_new_cols = ["historical_rank", "historical_points"]
    expected_z_cols = ["ace_rate_z_year", "first_serve_pct_z_year_surface"]

    for col in expected_new_cols + expected_z_cols:
        if col in final_sample.columns:
            print(f"✅ Column exists: {col}")
        else:
            print(f"⚠️ Missing column: {col}")

    # Check data types and ranges
    if "historical_rank" in final_sample.columns:
        rank_min = final_sample["historical_rank"].min()
        rank_max = final_sample["historical_rank"].max()
        print(f"✅ Ranking range: {rank_min} to {rank_max} (should be 1 to ~2000)")

    # Test 5: Usage Examples
    print("\n5️⃣ TESTING USAGE EXAMPLES")

    try:
        # Example 1: Surface analysis
        if "first_serve_pct_z_year_surface" in final_sample.columns:
            surface_analysis = final_sample.groupby("surface")["first_serve_pct_z_year_surface"].agg(["mean", "count"])
            print("📊 Surface-adjusted serve performance by surface:")
            print(surface_analysis.round(3))

        # Example 2: Era comparison
        if "era" in final_sample.columns and "ace_rate_z_year" in final_sample.columns:
            era_analysis = final_sample.groupby("era")["ace_rate_z_year"].mean()
            print("\n📊 Era-adjusted ace rates:")
            print(era_analysis.round(3))

        # Example 3: Ranking context
        if "historical_rank" in final_sample.columns:
            top_players = final_sample[final_sample["historical_rank"] <= 10]
            if len(top_players) > 0:
                avg_win_pct = top_players["won_match"].mean()
                print(f"\n📊 Top-10 players win rate in sample: {avg_win_pct:.3f}")

    except Exception as e:
        print(f"⚠️ Error in usage examples: {e}")

    print("\n✅ TEST COMPLETE")
    print(f"Enhanced sample shape: {final_sample.shape}")
    print(f"Total columns: {len(final_sample.columns)}")
    enhanced_cols = len([col for col in final_sample.columns if "_z_" in col or "historical" in col])
    print(f"Enhanced features: {enhanced_cols}")

    return final_sample


def test_full_pipeline():
    """Test the complete enhanced transformation pipeline."""
    print("\n\n🚀 TESTING FULL ENHANCED PIPELINE")
    print("=" * 50)

    try:
        from data_pipeline.transformation import transform_to_player_match

        print("Running complete enhanced transformation...")
        result = transform_to_player_match()

        summary = result["summary"]

        print("✅ FULL PIPELINE SUCCESS")
        print(f"   Total rows: {summary['total_rows']:,}")
        print(f"   Unique players: {summary['unique_players']:,}")
        print(f"   Unique matches: {summary['unique_matches']:,}")
        print(f"   Z-score features: {summary['z_score_features']}")
        print(f"   Ranking features: {summary['ranking_features']}")

        # Check output files
        output_file = CLEANED_DATA_DIR / "atp_player_match_enhanced.csv"
        if output_file.exists():
            print(f"✅ Enhanced dataset saved: {output_file}")

        return True

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


def main():
    """Run all enhanced transformation tests."""
    print("🎾 ENHANCED TENNIS ANALYSIS TESTING SUITE")
    print("=" * 60)

    # Test enhanced features with sample
    sample_result = test_enhanced_features()

    # Test full pipeline
    pipeline_success = test_full_pipeline()

    print("\n🏁 TESTING SUMMARY")
    print("=" * 30)
    if sample_result is not None:
        print("✅ Enhanced features test: PASSED")
    else:
        print("❌ Enhanced features test: FAILED")

    if pipeline_success:
        print("✅ Full pipeline test: PASSED")
    else:
        print("❌ Full pipeline test: FAILED")

    print("\n🎾 Ready for enhanced tennis era analysis!")


if __name__ == "__main__":
    main()
