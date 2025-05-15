# %% [markdown]
# Data Cleaning and Preprocessing Plan

# --- Objective ---
# To clean the aggregated tennis datasets, handle missing values, address outliers, 
# perform necessary data type corrections, and create a foundation for robust feature 
# engineering and analysis. This script aims to produce clean, analysis-ready datasets.

# --- Overall Strategy ---
# 1. Load aggregated datasets.
# 2. Implement robust player and tournament name standardization.
# 3. Refine and execute match linking strategies across datasets.
# 4. Systematically address missing values (imputation, removal, flagging).
# 5. Identify and treat outliers based on domain knowledge and statistical methods.
# 6. Correct data types for all relevant columns.
# 7. Perform initial feature engineering based on cleaned data.
# 8. Validate data consistency (e.g., scores vs. PBP outcomes).
# 9. Save cleaned datasets for downstream use.

# --- Tools and Libraries ---
# - pandas for data manipulation.
# - numpy for numerical operations.
# - scikit-learn for potential imputation or scaling.
# - fuzzywuzzy or similar for string matching (player/tournament names).
# - matplotlib and seaborn for visualizations during cleaning (optional, for verification).

# %% [markdown]
# # 1. Setup: Imports and Helper Functions

# %%
import pandas as pd
import numpy as np
import os
import re # For regex operations in cleaning
from sklearn.impute import SimpleImputer # Example imputer
from fuzzywuzzy import process, fuzz # For fuzzy string matching (install if needed)

# Optional: plotting for verification
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Placeholder for helper functions (e.g., loading, summary stats)
def load_cleaned_data_summary(df, description):
    print(f"\n--- {description} ---")
    print(f"Shape: {df.shape}")
    print("Missing values after initial cleaning:\n", df.isnull().sum().sort_values(ascending=False).head(10))
    # Add more summary details as needed

# %% [markdown]
# # 2. Load Aggregated Datasets
# - Load `aggregated_atp_matches.csv`
# - Load `aggregated_pbp_matches.csv` (potentially with sampling for PBP strings if memory is an issue initially)
# - Load `aggregated_slam_matches.csv`
# - Load `aggregated_slam_points.csv` (potentially with sampling)

# %%
# Define paths to the aggregated data files (replace with your actual paths)
# These should point to the outputs of aggregate_raw_data.py or similar aggregation process.
# Ensure these paths are correct and the files exist.
ATP_MATCHES_PATH = 'data/atp_matches/aggregated_atp_matches.csv'
ATP_PBP_PATH = 'data/atp_point_by_point/aggregated_pbp_matches.csv'
SLAM_MATCHES_PATH = 'data/slam_point_by_point/aggregated_slam_matches.csv'
SLAM_POINTS_PATH = 'data/slam_point_by_point/aggregated_slam_points.csv'

df_atp_matches = None
df_atp_pbp = None
df_slam_matches = None
df_slam_points = None

# Load functions from eda.py or define new ones for robust loading
def load_data_for_cleaning(file_path, description, nrows=None):
    print(f"\n--- Loading: {description} ---")
    print(f"File: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Loaded shape: {df.shape}")
        df.info(verbose=True, show_counts=True)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Example loading (uncomment and run when ready)
# df_atp_matches = load_data_for_cleaning(ATP_MATCHES_PATH, "ATP Match Data")
# df_atp_pbp = load_data_for_cleaning(ATP_PBP_PATH, "ATP PBP Data") #, nrows=50000) # Sample for dev
# df_slam_matches = load_data_for_cleaning(SLAM_MATCHES_PATH, "Slam Match Data")
# df_slam_points = load_data_for_cleaning(SLAM_POINTS_PATH, "Slam Points Data") #, nrows=100000) # Sample for dev


# %% [markdown]
# # 3. Player Name Standardization
# - **Goal**: Create a unique identifier for each player across all datasets.
# - **Steps**:
#   3.1. Collect all unique player names from:
#        - `df_atp_matches`: 'winner_name', 'loser_name'
#        - `df_atp_pbp`: 'server1', 'server2' (Note: EDA showed these are often Lastname F.)
#        - `df_slam_matches`: 'player1', 'player2'
#        - (Slam points might use IDs from slam_matches, but good to verify if names exist)
#   3.2. Develop a standardization function:
#        - Handle different formats (e.g., "Last, First M.", "First M. Last", "Last F.")
#        - Normalize case, remove extra spaces, accents.
#        - Consider using `fuzzywuzzy` or similar for matching variations to a canonical name.
#   3.3. Create a master player mapping (e.g., `player_raw_name` -> `player_canonical_name` -> `player_id`).
#        - ATP datasets have `winner_id`/`loser_id`. Slam datasets have `player1id`/`player2id`. Investigate if these can be harmonized or if a new system is needed.
#   3.4. Apply the mapping to create standardized player name/ID columns in each DataFrame.

# %%
# Placeholder for player name standardization logic

# def standardize_player_name(name):
#     # ... logic ...
#     return standardized_name

# def create_player_master_list(dfs_with_player_names):
#     # ... logic ...
#     return player_master_df


# %% [markdown]
# # 4. Tournament Name Standardization
# - **Goal**: Ensure consistent tournament identification for linking, especially between ATP and Slam datasets.
# - **Steps**:
#   4.1. Collect unique tournament names from:
#        - `df_atp_matches`: 'tourney_name'
#        - `df_atp_pbp`: 'tny_name'
#        - `df_slam_matches`: 'slam', 'event_name' (e.g., 'Wimbledon', 'Australian Open Men's Singles')
#   4.2. Develop a standardization function:
#        - Handle variations in naming, years in names, sponsor names.
#        - Map Grand Slam names to a common identifier (e.g., 'Wimbledon', 'Roland Garros', 'US Open', 'Australian Open').
#   4.3. Create a master tournament mapping if needed.
#   4.4. Apply to create standardized tournament name columns.

# %%
# Placeholder for tournament name standardization logic

# def standardize_tournament_name(name, year=None):
#     # ... logic ...
#     return standardized_t_name


# %% [markdown]
# # 5. Match Linking Strategy Refinement and Execution
# - **Goal**: Reliably link matches across different data sources.
# - **Strategies**:
#   5.1. **ATP PBP to ATP Matches**:
#        - Use standardized player names, standardized tournament name, and date.
#        - Create a robust composite `match_key` in both DataFrames.
#        - Evaluate merge success and fallout.
#   5.2. **Slam Points to Slam Matches**:
#        - Primarily use `match_id`. This should be relatively straightforward. Verify completeness.
#   5.3. **Linking ATP General Matches with Slam Matches**:
#        - This is the most complex.
#        - Requires standardized player names, standardized tournament names (identifying Slams in ATP data), and date ranges.
#        - Consider creating a unified `match_id` or linking table.
#   5.4. **Post-Linking**: After linking, decide on a primary source for overlapping information (e.g., if ATP PBP and ATP Matches both have score, which one to trust or how to reconcile).

# %%
# Placeholder for match linking logic

# def create_composite_match_key(df, date_col, tourney_col, p1_col, p2_col):
#    # ... logic to create a unique key after standardization ...
#    return key_series


# %% [markdown]
# # 6. Handling Missing Values (Iterative Process, Per Dataset/Merged Data)
# - **General Approach**: For each key column:
#   - Identify percentage of missing values.
#   - Determine reason for missingness (e.g., not recorded, not applicable, error).
#   - Choose a strategy:
#     - **Deletion**: If > X% missing and not critical, or if rows are unusable.
#     - **Imputation**:
#       - Mean/Median/Mode for numerical/categorical.
#       - Model-based (e.g., KNNImputer, regression) if appropriate.
#       - Using external data or relationships (e.g., impute player height from a master list).
#       - Zero or constant for specific cases (e.g., aces if not recorded might be 0, but flag it).
#     - **Flagging**: Create a binary column indicating if a value was imputed/missing.
# - **Specific Columns of Focus**:
#   - **Player Identifiers**: Should be 0% missing after standardization/linking if possible.
#   - **Match Identifiers**: Critical for joins.
#   - **Dates, Surface, Tournament Level, Round**: Impute with care or drop if essential and missing.
#   - **Scores**: Critical; attempt to reconstruct from PBP if available, otherwise problematic.
#   - **Serve Stats (aces, DFs, svpt, etc.)**:
#     - Often missing together.
#     - Imputation could be by player average, surface average, or more advanced models.
#     - Consider if `0` is a valid imputation or if it implies missing data.
#   - **Player Attributes (height, age, rank, points)**:
#     - Height: Can be imputed from other instances of the same player if a master list is built.
#     - Age: Calculable if birthdate is found/available, or impute with caution.
#     - Rank/Points: Very sensitive. Imputation can be tricky. May need to flag or exclude matches with missing ranks for certain analyses.
#   - **PBP String ('pbp' in `df_atp_pbp`)**: If missing, the record is not useful for PBP analysis.
#   - **Slam Points Detailed Metrics (`Speed_KMH`, `WinnerShotType`, etc.)**: Often sparse. Decide if to impute, flag, or use only when available. `Speed_KMH == 0` needs special handling (treat as NaN or analyze context).

# %%
# Placeholder for missing value handling logic (example for one column)

# def handle_missing_player_height(df, player_id_col, height_col):
#     # Example: Impute height using mean height for that player from other records
#     # df[height_col] = df.groupby(player_id_col)[height_col].transform(lambda x: x.fillna(x.mean()))
#     # Or fill with global mean if player-specific mean is not available
#     # df[height_col].fillna(df[height_col].mean(), inplace=True)
#     return df

# %% [markdown]
# # 7. Outlier Treatment (Iterative Process, Per Dataset/Merged Data)
# - **General Approach**:
#   - Identify potential outliers using:
#     - Boxplots, histograms.
#     - Z-scores (e.g., values > 3 or < -3 standard deviations from the mean).
#     - Interquartile Range (IQR) method.
#     - Domain knowledge (e.g., a match duration of 1 minute, 100 aces by one player in a 3-set match).
#   - Investigate outliers: Are they data entry errors or genuine extreme values?
#   - Choose a treatment strategy:
#     - **Correction**: If identified as an error and correct value is known/inferable.
#     - **Capping/Flooring (Winsorization)**: Limit extreme values to a certain percentile (e.g., 1st and 99th).
#     - **Transformation**: Log, sqrt, etc., can reduce skewness caused by outliers.
#     - **Removal**: If the outlier is likely an error and uncorrectable, or significantly skews analysis (use with caution).
#     - **Treat as Missing**: If the outlier indicates bad data.
# - **Specific Columns of Focus**:
#   - **Match Duration (`minutes`)**: Check for impossibly short or long matches.
#   - **Player Ages, Heights**: Validate against reasonable ranges.
#   - **Ranks, Rank Points**: Extreme values (e.g., rank 9999) might mean unranked.
#   - **Serve Stats (`w_ace`, `l_df`, etc.)**: Unusually high counts per match/game.
#   - **Slam PBP `Speed_KMH`**: Impossibly high or low speeds (after handling zeros).
#   - **Slam PBP `Rally`, `RallyCount`, `DistanceRun`**: Check for extreme values.

# %%
# Placeholder for outlier treatment logic

# def cap_outliers_iqr(df, column, factor=1.5):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - factor * IQR
#     upper_bound = Q3 + factor * IQR
#     df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
#     df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
#     return df

# %% [markdown]
# # 8. Data Type Correction
# - **Goal**: Ensure all columns have the most appropriate data type for analysis and memory efficiency.
# - **Steps**:
#   - Convert date columns (e.g., `tourney_date`, PBP `date`) to `datetime` objects.
#   - Convert numerical columns (e.g., ranks, points, stats) to appropriate numeric types (int, float). Handle non-numeric placeholders (e.g., '-', 'NA') before conversion.
#   - Convert categorical columns with a limited set of values (e.g., `surface`, `round`, `tourney_level`) to pandas `category` type.
#   - Ensure boolean-like columns (e.g., `adf_flag`) are `bool`.

# %%
# Placeholder for data type correction logic

# def correct_data_types(df):
#     # Example:
#     # if 'tourney_date' in df.columns:
#     #     df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
#     # if 'winner_rank' in df.columns:
#     #     df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce')
#     # if 'surface' in df.columns:
#     #     df['surface'] = df['surface'].astype('category')
#     return df

# %% [markdown]
# # 9. Feature Engineering (Initial Set)
# - **Goal**: Create new features that might be useful for analysis or modeling.
# - **Based on Cleaned Data**:
#   9.1. **From PBP Data (`df_atp_pbp`, `df_slam_points`)**:
#        - Parse PBP strings/sequences to extract detailed stats if not already available:
#          - Points won by server/returner.
#          - Break points won/faced/converted.
#          - Game types (love games, tiebreaks played/won).
#          - Serve percentages (1st serve in, 1st serve won, 2nd serve won) if not in match stats.
#          - Double fault frequency per game/set.
#   9.2. **Relational Features**:
#        - `age_diff` (winner_age - loser_age).
#        - `rank_diff`, `rank_points_diff`.
#        - `is_upset` (e.g., based on rank or seeding).
#   9.3. **Time-based Features**:
#        - Player experience (e.g., number of matches played up to a certain date).
#        - Rest days before a match (requires tournament schedule context).
#   9.4. **Surface Specialization**:
#        - Player win % on a given surface prior to the match.
#   9.5. **Derived Stats**:
#        - `serve_points_won_pct` = (`w_1stWon` + `w_2ndWon`) / `w_svpt`
#        - `return_points_won_pct` (requires opponent's serve stats).
#   9.6. **From Slam Points**:
#        - Average rally length per match.
#        - Average serve speed per match (1st, 2nd).
#        - Player distance covered per match/point.

# %%
# Placeholder for feature engineering logic

# def engineer_features(df):
#     # Example:
#     # if 'winner_age' in df.columns and 'loser_age' in df.columns:
#     #     df['age_diff'] = df['winner_age'] - df['loser_age']
#     return df

# %% [markdown]
# # 10. Score Parsing and Validation
# - **Goal**: Ensure consistency between recorded match scores and outcomes, potentially validated by PBP data.
# - **Steps**:
#   10.1. Parse textual score strings (e.g., "6-3 6-4", "7-6(5) 2-6 6-3 RET") into structured format (e.g., sets won by P1, sets won by P2, games won per set).
#   10.2. Handle retirements (RET), walkovers (W/O), defaults (DEF).
#   10.3. If PBP data is linked, verify that the PBP outcome matches the recorded score (e.g., number of sets, games).
#   10.4. Create flags for score inconsistencies or matches with non-standard outcomes.

# %%
# Placeholder for score parsing and validation logic

# def parse_score_string(score_str):
#     # ... complex logic to parse various score formats ...
#     # Returns (p1_sets, p2_sets, p1_games_set1, p2_games_set1, ..., status)
#     pass


# %% [markdown]
# # 11. Handling Specific Data Issues from EDA
# - Revisit issues identified in `eda.py`:
#   - `Speed_KMH == 0` in `df_slam_points`: Confirm if these should be NaN or if they have a specific meaning (e.g., fault, non-serve point). Likely treat as NaN for serve speed analysis.
#   - Erroneous rank/point values (e.g., 9999): Standardize to NaN or a specific value for unranked if not already handled.
#   - PBP string characters/delimiters: Ensure parsing logic correctly handles all variations found.
#   - Consistency of `adf_flag` with presence of Ace/DF in PBP string.

# %%
# Placeholder for addressing specific EDA issues

# %% [markdown]
# # 12. Saving Cleaned Data
# - **Goal**: Store the processed, cleaned, and feature-enriched datasets.
# - **Options**:
#   - Save as new CSV files (e.g., `atp_matches_cleaned.csv`).
#   - Save in a more efficient format like Parquet (good for columnar data, preserves types better).
#   - Load into a database (e.g., SQLite, DuckDB) for querying.
# - **Strategy**:
#   - Define a `data/processed/` directory.
#   - Save each key DataFrame (e.g., unified match list, PBP details) separately or as a combined dataset if appropriate.

# %%
# PROCESSED_DATA_DIR = 'data/processed'
# if not os.path.exists(PROCESSED_DATA_DIR):
#     os.makedirs(PROCESSED_DATA_DIR)

# Example saving:
# df_atp_matches_cleaned.to_csv(os.path.join(PROCESSED_DATA_DIR, 'atp_matches_cleaned.csv'), index=False)
# df_unified_matches.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'unified_matches.parquet'), index=False)

# %% [markdown]
# # 13. Documentation and Summary
# - Keep notes on all major cleaning decisions, assumptions made, and transformations applied.
# - Summarize the state of the data after cleaning (e.g., remaining missing values, size of datasets).
# - This markdown file serves as part of that documentation.

# %% [markdown]
# # Main Execution Block (for running the cleaning script)

# %%
if __name__ == '__main__':
    print("Starting Data Cleaning and Preprocessing...")

    # --- 2. Load Data ---
    df_atp_matches = load_data_for_cleaning(ATP_MATCHES_PATH, "ATP Match Data")
    df_atp_pbp = load_data_for_cleaning(ATP_PBP_PATH, "ATP PBP Data") #, nrows=50000) # Load full for actual run
    df_slam_matches = load_data_for_cleaning(SLAM_MATCHES_PATH, "Slam Match Data")
    df_slam_points = load_data_for_cleaning(SLAM_POINTS_PATH, "Slam Points Data") #, nrows=100000) # Load full

    # --- Check if dataframes are loaded ---
    if df_atp_matches is None or df_atp_pbp is None or df_slam_matches is None or df_slam_points is None:
        print("\nOne or more dataframes failed to load. Exiting cleaning process.")
    else:
        print("\nAll datasets loaded successfully.")
        # --- 3. Player Name Standardization ---
        # Implement and call player standardization functions here
        print("\n--- Step 3: Player Name Standardization (Placeholder) ---")
        # all_dfs = {'atp_matches': df_atp_matches, 'atp_pbp': df_atp_pbp, ...}
        # player_master = create_player_master_list(all_dfs)
        # df_atp_matches = apply_player_std(df_atp_matches, player_master) ... etc.

        # --- 4. Tournament Name Standardization ---
        print("\n--- Step 4: Tournament Name Standardization (Placeholder) ---")

        # --- 5. Match Linking ---
        print("\n--- Step 5: Match Linking (Placeholder) ---")
        
        # --- 6. Missing Values ---
        print("\n--- Step 6: Missing Value Handling (Placeholder) ---")
        # df_atp_matches = handle_missing_values_atp(df_atp_matches) ... etc.

        # --- 7. Outlier Treatment ---
        print("\n--- Step 7: Outlier Treatment (Placeholder) ---")

        # --- 8. Data Type Correction ---
        print("\n--- Step 8: Data Type Correction (Placeholder) ---")
        # df_atp_matches = correct_data_types(df_atp_matches) ... etc.
        
        # --- 9. Feature Engineering ---
        print("\n--- Step 9: Feature Engineering (Placeholder) ---")
        # df_atp_matches = engineer_features(df_atp_matches) ... etc.

        # --- 10. Score Parsing and Validation ---
        print("\n--- Step 10: Score Parsing and Validation (Placeholder) ---")

        # --- 11. Handle Specific EDA Issues ---
        print("\n--- Step 11: Handle Specific EDA Issues (Placeholder) ---")

        # --- 12. Saving Cleaned Data ---
        print("\n--- Step 12: Saving Cleaned Data (Placeholder) ---")
        # Example:
        # if df_atp_matches is not None:
        #     PROCESSED_DATA_DIR = 'data/processed'
        #     if not os.path.exists(PROCESSED_DATA_DIR):
        #         os.makedirs(PROCESSED_DATA_DIR)
        #     df_atp_matches.to_csv(os.path.join(PROCESSED_DATA_DIR, 'atp_matches_cleaned_placeholder.csv'), index=False)
        #     print(f"Saved placeholder cleaned ATP matches to {PROCESSED_DATA_DIR}")
            
        print("\nData Cleaning and Preprocessing script structure complete. Implement steps.")

pass # End of main execution block 