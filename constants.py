from pathlib import Path

# Match-level data constants
DATA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
YEARS: range = range(2005, 2025)  # 2005–2024 inclusive
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/tennis.duckdb")

# Point-by-point data constants
PBP_DATA_URL_TEMPLATE = "https://raw.githubusercontent.com/JeffSackmann/tennis_pointbypoint/master/{filename}"
# Focusing on ATP main draw for now, per blueprint recommendation
PBP_FILENAMES = [
    "pbp_matches_atp_main_archive.csv",
    "pbp_matches_atp_main_current.csv",
]
PBP_RAW_DIR = Path("data/raw_pbp")

# Slam Point-by-point data constants (contains rally counts)
SLAM_PBP_DATA_URL_TEMPLATE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/{year}-{slam}-points.csv"
SLAM_PBP_RAW_DIR = Path("data/raw_slam_pbp")
SLAM_NAMES = ["ausopen", "frenchopen", "wimbledon", "usopen"]
SLAM_YEARS = range(2011, 2025) # Slam data available from 2011 to 2024 (AO/RG stop after 2022)


# SQL Queries
SUMMARY_SQL = """WITH srv AS (
    SELECT season,
           surface,
           SUM(w_ace + l_ace)           AS aces,
           SUM(w_df + l_df)             AS dfs,
           SUM(w_svpt + l_svpt)         AS sv_pts,
           SUM(w_1stIn + l_1stIn)       AS first_in,
           SUM(w_1stWon + l_1stWon)     AS first_won,
           SUM(w_bpSaved + l_bpSaved)   AS bp_saved,
           SUM(w_bpFaced + l_bpFaced)   AS bp_faced
    FROM   matches_all
    GROUP  BY season, surface)
SELECT season,
       surface,
       aces      * 1.0 / sv_pts              AS ace_rate,
       dfs       * 1.0 / sv_pts              AS df_rate,
       first_in  * 1.0 / sv_pts              AS first_in_pct,
       first_won * 1.0 / first_in            AS first_win_pct,
       bp_saved  * 1.0 / NULLIF(bp_faced,0)  AS bp_save_pct
FROM   srv;"""

Z_SCORE_SQL = """CREATE OR REPLACE TABLE analytics.yearly_summary_z AS
SELECT *,
       (ace_rate      - AVG(ace_rate)   OVER (PARTITION BY season)) / STDDEV_SAMP(ace_rate)   OVER (PARTITION BY season) AS ace_rate_z,
       (df_rate       - AVG(df_rate)    OVER (PARTITION BY season)) / STDDEV_SAMP(df_rate)    OVER (PARTITION BY season) AS df_rate_z,
       (first_in_pct  - AVG(first_in_pct)  OVER (PARTITION BY season)) / STDDEV_SAMP(first_in_pct)  OVER (PARTITION BY season) AS first_in_z,
       (first_win_pct - AVG(first_win_pct) OVER (PARTITION BY season)) / STDDEV_SAMP(first_win_pct) OVER (PARTITION BY season) AS first_win_z,
       (bp_save_pct   - AVG(bp_save_pct)  OVER (PARTITION BY season)) / STDDEV_SAMP(bp_save_pct)  OVER (PARTITION BY season) AS bp_save_z
FROM analytics.yearly_summary;""" 