"""
tennis_era_analysis.py

Main orchestrator script for the cross‑era tennis performance comparison pipeline.

Run‑modes
---------
```bash
python tennis_era_analysis.py            # full ETL (downloads ≈80 MB)
python tennis_era_analysis.py test       # run quick unit tests only (no network)
python tennis_era_analysis.py --force-pbp # full ETL, forcing PBP reload
```
"""

import sys

# Local project modules
import constants as C
import utils as U
import data_loading as dl
import analytics as an
import pbp_processing as pbp
import qc as qc
import elo

# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------

def main(force_pbp_reload: bool = False, force_elo_reload: bool = False) -> None:
    """Runs the full ETL pipeline."""
    # Download raw data
    dl.download_data()
    dl.download_pbp_data()
    dl.download_slam_pbp_data()
    dl.download_slam_match_data()

    # Establish DB connection and load match-level data
    con = dl.load_duckdb()

    # Optionally clear existing Elo tables if force_elo_reload is True
    if force_elo_reload:
        print("\nForce reloading Elo tables...")
        con.execute("DROP TABLE IF EXISTS analytics.elo_player_season;")
        con.execute("DROP TABLE IF EXISTS analytics.elo_player_season_surface;")
        print("✔ Dropped existing Elo tables.")

    # Match-level analytics
    an.compute_yearly_summary(con)
    an.add_z_scores(con)

    # Point-by-Point processing
    pbp.parse_and_load_points(con, force_reload=force_pbp_reload)
    pbp.load_slam_points(con, force_reload=force_pbp_reload)
    pbp.create_points_view(con)
    qc.run_pbp_qc(con)

    # Rally-based analytics (using PBP data)
    an.compute_rally_stats(con, force_reload=force_pbp_reload)

    # Technical layer analytics
    an.compute_technical_stats(con, force_reload=force_pbp_reload)
    an.compute_return_stats(con, force_reload=force_pbp_reload)

    # --- Elo Rating Calculation ---
    print("\nCalculating Elo Ratings...")
    start_year = C.YEARS.start
    end_year = C.YEARS.stop - 1 # C.YEARS is a range, stop is exclusive
    
    print("Calculating Unified Elo...")
    elo.calculate_unified_elo_ratings(con, start_year, end_year)
    print("✔ Unified Elo calculation finished.")

    print("\nCalculating Surface-Specific Elo...")
    elo.calculate_surface_elo_ratings(con, start_year, end_year)
    print("✔ Surface-Specific Elo calculation finished.")
    # ------------------------------

    # --- Elo QC --- #
    qc.run_elo_qc(con)
    # -------------- #

    con.close()
    print("\n🏁 Pipeline finished. DB →", U._display_path(C.DB_PATH.resolve()))


# ---------------------------------------------------------------------------
# SCRIPT EXECUTION LOGIC
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    force_pbp = "--force-pbp" in sys.argv
    force_elo = "--force-elo" in sys.argv

    if force_pbp:
        print("Running main pipeline with forced PBP reload...")
    if force_elo:
        print("Running main pipeline with forced Elo reload...")
    if not force_pbp and not force_elo:
        print("Running main pipeline...")
        
    main(force_pbp_reload=force_pbp, force_elo_reload=force_elo)
