"""
Tennis Era Analysis - Configuration Constants
Centralized configuration for the tennis data processing pipeline.
"""

from pathlib import Path

# =============================================================================
# DATA SOURCES & PATHS
# =============================================================================

# Match-level data constants
DATA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
YEARS = range(2005, 2025)  # 2005–2024 inclusive
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/tennis.duckdb")

# Point-by-point data constants
PBP_DATA_URL_TEMPLATE = "https://raw.githubusercontent.com/JeffSackmann/tennis_pointbypoint/master/{filename}"
PBP_FILENAMES = [
    "pbp_matches_atp_main_archive.csv",
    "pbp_matches_atp_main_current.csv",
]
PBP_RAW_DIR = Path("data/raw_pbp")

# Slam Point-by-point data constants
SLAM_PBP_DATA_URL_TEMPLATE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/{year}-{slam}-points.csv"
SLAM_PBP_RAW_DIR = Path("data/raw_slam_pbp")
SLAM_NAMES = ["ausopen", "frenchopen", "wimbledon", "usopen"]
SLAM_YEARS = range(2011, 2025)

# Processed data paths
CLEANED_DATA_DIR = Path("data/cleaned_refactored")
OUTPUT_DATA_DIR = Path("data/output")

# =============================================================================
# ERA DEFINITIONS
# =============================================================================

ERA_DEFINITIONS = {"Classic": (2005, 2010), "Transition": (2011, 2015), "Modern": (2016, 2020), "Current": (2021, 2024)}

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Batch processing
DEFAULT_BATCH_SIZE = 10000
MATCHING_BATCH_SIZE = 100  # Reduced for better progress tracking and responsiveness

# Rally length categories
RALLY_SHORT_MAX = 4
RALLY_MEDIUM_MAX = 8

# Matching thresholds
FUZZY_MATCH_THRESHOLD = 80
EMBEDDING_MATCH_THRESHOLD = 85  # Convert cosine similarity (0-1) to percentage (0-100)

# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

# Core columns that remain the same
CORE_COLUMNS = [
    "match_id",
    "tourney_date",
    "tourney_name",
    "surface",
    "round",
    "best_of",
    "score",
    "minutes",
    "tourney_level",
    "tourney_id",
    "match_num",
    "tourney_date_int",
    "era",
    "year",
]

# Winner to player mappings
WINNER_TO_PLAYER_MAPPING = {
    "winner_id": "player_id",
    "winner_seed": "seed",
    "winner_entry": "entry",
    "winner_name": "player_name",
    "winner_hand": "hand",
    "winner_ht": "height",
    "winner_ioc": "country",
    "winner_age": "age",
    "winner_rank": "rank",
    "winner_rank_points": "rank_points",
    "w_ace": "aces",
    "w_df": "double_faults",
    "w_svpt": "serve_points",
    "w_1stIn": "first_serves_in",
    "w_1stWon": "first_serves_won",
    "w_2ndWon": "second_serves_won",
    "w_SvGms": "service_games",
    "w_bpSaved": "break_points_saved",
    "w_bpFaced": "break_points_faced",
}

# Loser to player mappings (same structure)
LOSER_TO_PLAYER_MAPPING = {
    "loser_id": "player_id",
    "loser_seed": "seed",
    "loser_entry": "entry",
    "loser_name": "player_name",
    "loser_hand": "hand",
    "loser_ht": "height",
    "loser_ioc": "country",
    "loser_age": "age",
    "loser_rank": "rank",
    "loser_rank_points": "rank_points",
    "l_ace": "aces",
    "l_df": "double_faults",
    "l_svpt": "serve_points",
    "l_1stIn": "first_serves_in",
    "l_1stWon": "first_serves_won",
    "l_2ndWon": "second_serves_won",
    "l_SvGms": "service_games",
    "l_bpSaved": "break_points_saved",
    "l_bpFaced": "break_points_faced",
}

# =============================================================================
# SQL QUERIES
# =============================================================================

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

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# Embedding model configuration
EMBEDDING_MODEL = "mxbai-embed-large:latest"  # Best accuracy/speed trade-off (~670MB)
EMBEDDING_SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for name matching
EMBEDDING_BATCH_SIZE = 50  # Names to embed in one batch request

# Alternative embedding models (for reference)
EMBEDDING_MODELS = {
    "best_accuracy": "bge-m3:latest",  # ~1.2GB, multi-lingual, SOTA
    "balanced": "mxbai-embed-large:latest",  # ~670MB, great accuracy/speed
    "fastest": "nomic-embed-text:latest",  # ~274MB, fastest, English-focused
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model": EMBEDDING_MODEL,
    "base_url": "http://localhost:11434",
    "similarity_threshold": EMBEDDING_SIMILARITY_THRESHOLD,
    "batch_size": EMBEDDING_BATCH_SIZE,
    "cache_size": 10000,  # Max unique names to cache
}
