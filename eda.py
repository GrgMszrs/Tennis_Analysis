# %% [markdown]
# Tennis Data Exploratory Data Analysis (EDA)

# --- Data Overview ---
# data/atp_matches/: Contains aggregated ATP match data: `aggregated_atp_matches.csv`.
#                    - Columns include tournament details, winner/loser info, match stats (aces, DFs, serve pts, etc.), rankings.
# data/archive/: Contains historical data like `atp_matches_archive_2000_2020.csv`.
# data/atp_point_by_point/: Contains aggregated ATP point-by-point (PBP) data: `aggregated_pbp_matches.csv`.
# data/live/: Contains currently updating data, like `live_matches_current_week.csv`.
# data/slam_point_by_point/: Contains aggregated Grand Slam PBP data.
#                           - `aggregated_slam_matches.csv`: Similar to ATP matches but for slams.
#                           - `aggregated_slam_points.csv`: Detailed PBP info for slams, more structured.

# --- EDA Plan ---
# Objective: Gain a comprehensive understanding of the datasets to inform data cleaning, feature engineering, and modeling.

# General Steps for each aggregated dataset:
# 1. Load the aggregated data.
# 2. Initial Inspection:
#    - Display .head(), .info(), .describe().
#    - Check data types of columns.
#    - Identify an initial list of potentially important variables.
# 3. Missing Values Analysis:
#    - Quantify missing values per column.
#    - Understand patterns of missingness.
#    - Consider imputation strategies for later analysis.
# 4. Univariate Analysis:
#    - Numerical variables: Histograms, box plots for distributions, outliers.
#    - Categorical variables: Frequency counts, bar charts.
# 5. Bivariate/Multivariate Analysis:
#    - Scatter plots for numerical pairs.
#    - Correlation matrices.
#    - Grouped analysis (e.g., average aces by surface).
# 6. Text Data Exploration (for PBP string data in aggregated_pbp_matches.csv):
#    - Understand encoding scheme.
#    - Calculate lengths of PBP sequences.
#    - Frequency of PBP events.

# --- Specific EDA Steps for Each Aggregated File ---

# Part 1: ATP Match Data (data/atp_matches/aggregated_atp_matches.csv)
# 1.1. Load the aggregated ATP match data.
# 1.2. General Inspection (Shape, Data types, Summary statistics for numerical columns).
# 1.3. Missing Values (Identify columns with significant missing data).
# 1.4. Univariate Analysis (Distributions of match duration, player ages, heights, ranks, serve stats; Frequency of surfaces, tournament levels, rounds).
# 1.5. Bivariate Analysis (Relationship winner_rank vs loser_rank, Aces vs. Surface, Duration vs. Best_of, Serve stats correlations).
# 1.6. Sanity Checks (Rank points consistency, score sensibility).

# Part 2: ATP Point-by-Point Data (data/atp_point_by_point/aggregated_pbp_matches.csv)
# 2.1. Load aggregated ATP PBP data.
# 2.2. General Inspection.
# 2.3. PBP String ('pbp' column) Exploration.
# 2.4. Missing values in 'pbp' and other key columns.
# 2.5. Relationship with Match Data (Linking aggregated_atp_matches.csv and aggregated_pbp_matches.csv).

# Part 3: Slam Data
# 3.1. Slam Match Data (data/slam_point_by_point/aggregated_slam_matches.csv)
#      3.1.1. Load data.
#      3.1.2. Inspect columns and compare with aggregated_atp_matches.csv schema.
#      3.1.3. Missing values and completeness.

# 3.2. Slam Points Data (data/slam_point_by_point/aggregated_slam_points.csv)
#      3.2.1. Load data.
#      3.2.2. Inspect columns (PointWinner, ServeNumber, RallyLength, etc.). This is highly structured.
#      3.2.3. Data types and missing values, especially for key point metrics.
#      3.2.4. Univariate analysis of point-level variables.
#      3.2.5. Linking with aggregated_slam_matches.csv using 'match_id'.

# Part 4: Cross-Dataset Consistency and Linkage
# 4.1. Player Name/ID Consistency across all three main aggregated files.
# 4.2. Match Identification and linkage strategies.

# --- Tools and Libraries for EDA ---
# - pandas for data manipulation and analysis.
# - matplotlib and seaborn for plotting.
# - numpy for numerical operations.

# --- Initial Questions to Answer during EDA ---
# - Time range covered by each aggregated dataset.
# - Completeness of serve/return statistics.
# - Common surface types and match counts.
# - PBP data structure details.
# - Player and tournament identification consistency.
# - Match outcome coding (completed, W/O, retired).

# --- Next Steps After EDA ---
# - Data cleaning plan.
# - Feature engineering ideas.
# - Define specific analytical questions.
# - Plan for database schema (e.g., DuckDB).

# %% [markdown]
# # Importing Libraries

# %%   
import pandas as pd
import numpy as np # For potential numerical operations
import os

# For plotting (optional, can be uncommented in a notebook environment)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Configure pandas display options for better output in console/notebook
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # Tennis Data Exploratory Data Analysis (EDA)
# This script/notebook performs EDA on the aggregated tennis datasets.

# %%
# --- Helper function to load and provide initial info --- 
def load_and_inspect(file_path, file_description, nrows=None):
    print(f"\n--- Exploring: {file_description} ---")
    print(f"File: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:\n", df.head())
        print("\nInfo:")
        df.info(verbose=True, show_counts=True) # Ensure all columns are shown
        print("\nBasic Descriptive Statistics (Numerical):")
        print(df.describe(include=np.number))
        print("\nBasic Descriptive Statistics (Categorical/Object):")
        print(df.describe(include=['object', 'category']))
        return df
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None 

# %% [markdown]
# ## Part 1: ATP Match Data (aggregated_atp_matches.csv)

# %% 
def explore_atp_match_data():
    file_path = 'data/atp_matches/aggregated_atp_matches.csv'
    df_atp_matches = load_and_inspect(file_path, "Aggregated ATP Match Data")
    
    if df_atp_matches is not None:
        print("\n--- 1.3. Missing Values Analysis (ATP Matches) ---")
        missing_values = df_atp_matches.isnull().sum()
        missing_percent = (missing_values / len(df_atp_matches) * 100).sort_values(ascending=False)
        print("Percentage of missing values per column:\n", missing_percent[missing_percent > 0])
        print("\nDiscussion on Missing Values (ATP Matches):")
        print("- 'winner_entry', 'loser_entry', 'winner_seed', 'loser_seed': Often NaN, which is expected for unseeded players or non-wildcard entries. This is likely by design.")
        print("- 'winner_ht', 'loser_ht': Missing heights are common. Imputation might be complex (e.g., average, or from other sources if available).")
        print("- 'minutes': Some missing match durations. Could be imputed with care (e.g., average by set count) or matches excluded from duration-specific analysis.")
        print("- Serve stats (e.g., 'w_ace', 'l_ace', etc.): Missing values might indicate incomplete data recording for those matches.")
        print("- Rank/points: Missing values here can significantly impact ranking-based analysis.")

        print("\n--- 1.4. Univariate Analysis (ATP Matches) ---")
        print("Value counts for 'surface' (excluding NaN):\n", df_atp_matches['surface'].value_counts(dropna=False))
        print("\nValue counts for 'tourney_level' (excluding NaN):\n", df_atp_matches['tourney_level'].value_counts(dropna=False))
        print("\nValue counts for 'round' (excluding NaN):\n", df_atp_matches['round'].value_counts(dropna=False).head(10))
        print("\nValue counts for 'best_of' (excluding NaN):\n", df_atp_matches['best_of'].value_counts(dropna=False))

        numerical_cols_to_analyze = ['minutes', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 
                                     'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
                                     'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms',
                                     'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms']
        print("\nSummary for key numerical columns (excluding NaN during describe):")
        print(df_atp_matches[numerical_cols_to_analyze].describe().transpose())
        
        print("\nVisualizing distributions for key numerical columns (ATP Matches)...")
        print("Boxplots are useful for identifying statistical outliers.")
        for col in numerical_cols_to_analyze:
            if col in df_atp_matches.columns and pd.api.types.is_numeric_dtype(df_atp_matches[col]):
                plt.figure(figsize=(10, 4))
                sns.histplot(df_atp_matches[col].dropna(), kde=True, bins=50)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.show()

                plt.figure(figsize=(10, 2))
                sns.boxplot(x=df_atp_matches[col].dropna())
                plt.title(f'Boxplot of {col} (shows outliers)')
                plt.xlabel(col)
                plt.show()
            else:
                print(f"Column {col} not found for plotting.")
        plt.close('all')
        print("\nOutlier Discussion (ATP Matches):")
        print("- 'minutes': Very long matches could be outliers, or genuinely epic encounters. Context is key.")
        print("- Player heights/ages: Extreme values should be cross-verified if possible. Data entry errors are plausible.")
        print("- Serve stats (aces, DFs): Unusually high numbers in a single match might be valid or data errors.")
        print("- Ranks/Points: Outliers here (e.g., rank 9999) often signify unranked or error.")

        print("\nVisualizing distributions for key categorical columns (ATP Matches)...")
        categorical_cols_to_plot = ['surface', 'tourney_level', 'round', 'best_of']
        for col in categorical_cols_to_plot:
            if col in df_atp_matches.columns:
                plt.figure(figsize=(10, 5))
                order = df_atp_matches[col].value_counts().index if col == 'round' else None # Keep order for rounds for readability
                sns.countplot(y=df_atp_matches[col], order=order, palette="viridis")
                plt.title(f'Frequency of {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column {col} not found for plotting.")
        plt.close('all') # Close all figures from this loop

        print("\n--- 1.5. Bivariate Analysis (ATP Matches) ---")
        correlation_subset = ['minutes', 'winner_rank', 'loser_rank', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age', 'w_ace', 'l_ace', 'w_df', 'l_df', 'w_1stWon', 'l_1stWon', 'w_svpt', 'l_svpt']
        # Ensure all columns in correlation_subset exist in the DataFrame
        valid_correlation_cols = [col for col in correlation_subset if col in df_atp_matches.columns]

        if valid_correlation_cols:
            print("\nCorrelation matrix (subset of columns):")
            correlation_matrix = df_atp_matches[valid_correlation_cols].corr()
            print(correlation_matrix)
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
            plt.title('Correlation Matrix of Key ATP Match Variables')
            plt.show()
        else:
            print("Not enough valid columns for correlation_subset.")
        
        # Scatter plot: winner_rank vs loser_rank
        if 'winner_rank' in df_atp_matches.columns and 'loser_rank' in df_atp_matches.columns:
            plt.figure(figsize=(8, 6))
            # Using a sample for scatter plot if dataframe is too large, to avoid overplotting
            sample_df = df_atp_matches.sample(n=min(2000, len(df_atp_matches)), random_state=1)
            sns.scatterplot(data=sample_df, x='loser_rank', y='winner_rank', alpha=0.5)
            plt.title('Winner Rank vs Loser Rank (Sample)')
            plt.xlabel('Loser Rank')
            plt.ylabel('Winner Rank')
            plt.plot([0, 400], [0, 400], color='red', linestyle='--') # Diagonal line for reference
            plt.gca().invert_xaxis() # Typically lower rank is better
            plt.gca().invert_yaxis()
            plt.show()
        
        # Box plot: Aces by Surface type
        if 'w_ace' in df_atp_matches.columns and 'surface' in df_atp_matches.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df_atp_matches, x='surface', y='w_ace', palette="muted")
            plt.title('Winner Aces by Surface Type')
            plt.xlabel('Surface')
            plt.ylabel('Winner Aces')
            plt.show()

        # Box plot: Match duration (minutes) by Best_of (3 vs 5 sets)
        if 'minutes' in df_atp_matches.columns and 'best_of' in df_atp_matches.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df_atp_matches, x='best_of', y='minutes', palette="pastel")
            plt.title('Match Duration by Best Of Sets')
            plt.xlabel('Best Of (Sets)')
            plt.ylabel('Match Duration (Minutes)')
            plt.show()
        plt.close('all') # Close all figures from this section

        print("\n--- 1.6. Sanity Checks (ATP Matches) ---")
        # Example: Are winner_rank_points generally higher than loser_rank_points?
        # This is a simplified check; upsets are normal.
        # Note: Ranks can be NaN, so handle that if performing direct comparisons.
        upsets_by_rank_points = df_atp_matches[df_atp_matches['winner_rank_points'] < df_atp_matches['loser_rank_points']].shape[0]
        valid_rank_point_matches = df_atp_matches[['winner_rank_points', 'loser_rank_points']].dropna().shape[0]
        if valid_rank_point_matches > 0:
            print(f"Percentage of matches (with valid rank points) where winner had fewer rank points (potential upsets by points): { (upsets_by_rank_points / valid_rank_point_matches * 100) if valid_rank_point_matches > 0 else 0:.2f}%")
        else:
            print("Not enough valid rank point data for upset calculation.")
            
        # Check score sensibility (basic check - more detailed parsing would be needed for full validation)
        # e.g., score format, number of sets implied by score
        print("\nTop 5 most frequent scores:\n", df_atp_matches['score'].value_counts().head())

    return df_atp_matches

# %% [markdown]
# ## Part 2: ATP Point-by-Point Data (aggregated_pbp_matches.csv)

# %% 
def explore_atp_pbp_data():
    file_path = 'data/atp_point_by_point/aggregated_pbp_matches.csv'
    df_atp_pbp = load_and_display_sample(file_path, "Aggregated ATP PBP Data", nrows=5000) # Load a sample for PBP
    
    if df_atp_pbp is not None:
        # Ensure correct data types for player names and date for potential joins later
        for col in ['server1', 'server2', 'tny_name']:
            if col in df_atp_pbp.columns:
                df_atp_pbp[col] = df_atp_pbp[col].astype(str)
        if 'date' in df_atp_pbp.columns:
            df_atp_pbp['date'] = pd.to_datetime(df_atp_pbp['date'], errors='coerce')

        print("\n--- 2.3. PBP String ('pbp' column) Exploration (ATP PBP) - Revised based on README ---")
        if 'pbp' in df_atp_pbp.columns and not df_atp_pbp['pbp'].head(100).isnull().all():
            pbp_series = df_atp_pbp['pbp'].dropna().astype(str)
            sample_pbp_strings_for_chars = pbp_series.sample(min(1000, len(pbp_series)), random_state=1) # Larger sample for char count
            
            all_chars_in_sample = sorted(list(set(''.join(sample_pbp_strings_for_chars))))
            print(f"All unique characters found in PBP strings (sample of {len(sample_pbp_strings_for_chars)}): {all_chars_in_sample}")
            expected_chars = ['S', 'R', 'A', 'D', '.', ';', '/']
            unexpected_chars = [char for char in all_chars_in_sample if char not in expected_chars]
            if unexpected_chars:
                print(f"Unexpected characters found: {unexpected_chars}")
            else:
                print("All characters found are as per README specs (S, R, A, D, ., ;, /).")

            if 'pbp_length' not in df_atp_pbp.columns: # Calculate if not already done
                 df_atp_pbp['pbp_length'] = pbp_series.apply(len)
            print("\nDescriptive stats for 'pbp_length':\n", df_atp_pbp['pbp_length'].describe())

            # Delimiter counts based on README
            df_atp_pbp['num_sets_pbp'] = pbp_series.str.count('\.') + 1 # Sets are delimited by '.', so count + 1
            df_atp_pbp['num_games_pbp'] = pbp_series.str.count(';') + df_atp_pbp['num_sets_pbp'] # Games are delimited by ';', add num_sets because each set starts a new game count implicitly
            
            print("\nDescriptive stats for inferred number of sets (from PBP '.'):")
            print(df_atp_pbp['num_sets_pbp'].describe())
            print("\nValue counts for inferred number of sets (from PBP '.'):")
            print(df_atp_pbp['num_sets_pbp'].value_counts().sort_index())

            print("\nDescriptive stats for inferred number of games (from PBP ';'):")
            print(df_atp_pbp['num_games_pbp'].describe())

            # Plot distribution of inferred sets and games
            plt.figure(figsize=(10,4))
            sns.countplot(data=df_atp_pbp, x='num_sets_pbp')
            plt.title('Distribution of Inferred Number of Sets (from PBP delimiters)')
            plt.xlabel('Number of Sets')
            plt.ylabel('Frequency')
            plt.show()

            # adf_flag analysis
            if 'adf_flag' in df_atp_pbp.columns:
                print("\nAnalysis of 'adf_flag':")
                print(df_atp_pbp['adf_flag'].value_counts(normalize=True, dropna=False) * 100)
                sns.countplot(data=df_atp_pbp, x='adf_flag')
                plt.title('Distribution of adf_flag (Aces/DFs noted)')
                plt.xlabel('adf_flag')
                plt.ylabel('Frequency')
                plt.show()
            else:
                print("'adf_flag' column not found.")
            plt.close('all')

        else:
            print("'pbp' column is missing or contains all NaNs in the sample.")

        print("\n--- 2.4. Missing Values Analysis (ATP PBP) - Revisit ---")
        missing_values = df_atp_pbp.isnull().sum()
        missing_percent = (missing_values / len(df_atp_pbp) * 100).sort_values(ascending=False)
        print("Percentage of missing values per column:\n", missing_percent[missing_percent > 0])
        print("\nDiscussion on Missing Values (ATP PBP):")
        print("- 'pbp': If this is missing, the core point-by-point data is absent for that row.")
        print("- 'score': Missing scores make it hard to validate PBP sequences or link to match outcomes.")
        print("- Player names ('server1', 'server2'): Essential for linking.")
        print("- Other columns like 'date', 'tny_name', 'tour', 'draw' are critical for context and linking.")
        # No direct outlier handling for PBP strings themselves, but derived features (like game counts) could have outliers if PBP is malformed.

        print("\n--- 2.5. Relationship with Match Data (ATP Matches vs ATP PBP) ---")
        # Attempt to create a common join key. This is complex due to name variations.
        # For PBP data:
        # Standardize player names to F. Last (e.g. from "Federer Roger" to "R. Federer") for a basic match attempt
        def standardize_pbp_name(name):
            if pd.isna(name) or not isinstance(name, str): return None
            parts = name.split(' ')
            if len(parts) > 1:
                return f"{parts[-1][0]}. {parts[0]}" # L. Firstname
            return name # Return as is if single name
        
        df_atp_pbp['server1_standardized'] = df_atp_pbp['server1'].apply(standardize_pbp_name)
        df_atp_pbp['server2_standardized'] = df_atp_pbp['server2'].apply(standardize_pbp_name)
        
        # Create a sorted player pair string and then a match key
        df_atp_pbp['player_pair_pbp'] = df_atp_pbp.apply(lambda row: '_'.join(sorted([str(row['server1_standardized']), str(row['server2_standardized'])])), axis=1)
        df_atp_pbp['match_key_pbp'] = df_atp_pbp['date'].dt.strftime('%Y-%m-%d') + "_" + df_atp_pbp['tny_name'].str.lower() + "_" + df_atp_pbp['player_pair_pbp']

        print("\nAttempting to link with aggregated_atp_matches.csv...")
        # Load ATP match data (if not already loaded globally - for this function scope, we reload a small part)
        atp_matches_file = 'data/atp_matches/aggregated_atp_matches.csv'
        if os.path.exists(atp_matches_file):
            df_atp_matches_ref = pd.read_csv(atp_matches_file, usecols=['tourney_date', 'tourney_name', 'winner_name', 'loser_name', 'score'], nrows=10000 if __name__ != '__main__' else None) # load more if run as script
            df_atp_matches_ref['tourney_date'] = pd.to_datetime(df_atp_matches_ref['tourney_date'], errors='coerce')
            
            # Standardize names in ATP Match data (winner_name, loser_name often First Last or F. Last)
            # This is a very basic standardization. Robust name matching is a separate large problem.
            def standardize_atp_name(name):
                if pd.isna(name) or not isinstance(name, str): return None
                parts = name.split(' ')
                if len(parts) > 1:
                    if len(parts[0]) == 1 and parts[0].isupper(): # F. Last
                        return name
                    return f"{parts[0][0]}. {parts[-1]}" # First Last -> F. Last
                return name

            df_atp_matches_ref['winner_name_standardized'] = df_atp_matches_ref['winner_name'].apply(standardize_atp_name)
            df_atp_matches_ref['loser_name_standardized'] = df_atp_matches_ref['loser_name'].apply(standardize_atp_name)
            df_atp_matches_ref['player_pair_atp'] = df_atp_matches_ref.apply(lambda row: '_'.join(sorted([str(row['winner_name_standardized']), str(row['loser_name_standardized'])])), axis=1)
            df_atp_matches_ref['match_key_atp'] = df_atp_matches_ref['tourney_date'].dt.strftime('%Y-%m-%d') + "_" + df_atp_matches_ref['tourney_name'].str.lower() + "_" + df_atp_matches_ref['player_pair_atp']

            # Attempt merge
            merged_df = pd.merge(df_atp_pbp, df_atp_matches_ref, left_on='match_key_pbp', right_on='match_key_atp', suffixes=['_pbp', '_atp'])
            print(f"\nNumber of matches merged based on a composite key (date, tourney, players): {merged_df.shape[0]}")
            if not merged_df.empty:
                print("Sample of merged data (first 5 rows):\n", merged_df[['date', 'tny_name', 'server1', 'server2', 'score_pbp', 'winner_name', 'loser_name', 'score_atp']].head())
                
                # Compare scores for a small sample of matched records
                print("\nComparing scores for a sample of matched records:")
                score_comparison_sample = merged_df[['score_pbp', 'score_atp']].sample(min(5, len(merged_df)), random_state=1)
                print(score_comparison_sample)
                # A more rigorous comparison would involve parsing the score strings.
            else:
                print("No matches could be merged with the current key strategy on the sample data.")
                print("Match key from PBP (sample):\n", df_atp_pbp['match_key_pbp'].head())
                print("Match key from ATP Matches (sample):\n", df_atp_matches_ref['match_key_atp'].head())
                print("This suggests challenges in name/tournament standardization or date alignment.")
        else:
            print(f"ATP Match reference file not found: {atp_matches_file}")
        
        print("\n--- Part 2 (ATP PBP Data) Exploration Complete (Initial Pass) ---")

    return df_atp_pbp

# %% [markdown]
# ## Part 3: Slam Data

# %% [markdown]
# ### 3.1 Slam Match Data (aggregated_slam_matches.csv)

# %% 
def explore_slam_match_data():
    file_path = 'data/slam_point_by_point/aggregated_slam_matches.csv'
    df_slam_matches = load_and_display_sample(file_path, "Aggregated Slam Match Data")
    
    if df_slam_matches is not None:
        print("\n--- 3.1.2. Inspecting Columns (Slam Matches) ---")
        # Already done by load_and_display_sample, here we focus on categorical distributions
        categorical_slam_match_cols = ['slam', 'round', 'status', 'event_name', 'court_name'] # 'nation1', 'nation2' might be too many unique values for direct plotting
        print("\nVisualizing distributions for key categorical columns (Slam Matches)...")
        for col in categorical_slam_match_cols:
            if col in df_slam_matches.columns:
                plt.figure(figsize=(12, 6))
                # For columns with many unique values, show top N
                top_n = 20 if df_slam_matches[col].nunique() > 20 else df_slam_matches[col].nunique()
                order = df_slam_matches[col].value_counts().nlargest(top_n).index
                sns.countplot(y=df_slam_matches[col], order=order, palette="viridis_r")
                plt.title(f'Frequency of {col} (Top {top_n})')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column {col} not found for plotting.")
        plt.close('all')
        
        if 'year' in df_slam_matches.columns:
            plt.figure(figsize=(10,4))
            sns.countplot(data=df_slam_matches, x='year', palette="crest")
            plt.title('Number of Slam Matches per Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Matches')
            plt.xticks(rotation=45)
            plt.show()
            plt.close('all')

        print("\n--- 3.1.3. Missing Values Analysis (Slam Matches) ---")
        missing_values = df_slam_matches.isnull().sum()
        missing_percent = (missing_values / len(df_slam_matches) * 100).sort_values(ascending=False)
        print("Percentage of missing values per column:\n", missing_percent[missing_percent > 0])
        print("\nDiscussion on Missing Values (Slam Matches):")
        print("- 'status', 'court_name', 'court_id', 'nation1/2': Missing values here reduce contextual information about the match.")
        print("- If 'winner' is missing, it's a significant data quality issue for that match record.")
        # Outliers in Slam Matches would typically be in `year` if data entry was wrong, or if string fields have unusual values.
        # The primary keys like `match_id` should be unique.

    return df_slam_matches

# %% [markdown]
# ### 3.2 Slam Points Data (aggregated_slam_points.csv)

# %% 
def explore_slam_points_data():
    file_path = 'data/slam_point_by_point/aggregated_slam_points.csv'
    df_slam_points = load_and_display_sample(file_path, "Aggregated Slam Points Data", nrows=20000)
    
    if df_slam_points is not None:
        print("\n--- 3.2.3. Missing Values Analysis (Slam Points) ---")
        missing_values = df_slam_points.isnull().sum()
        missing_percent = (missing_values / len(df_slam_points) * 100).sort_values(ascending=False)
        print("Percentage of missing values per column (showing top 20 with missing values):\n", missing_percent[missing_percent > 0].head(20))
        print("\nDiscussion on Missing Values (Slam Points):")
        print("- 'Speed_KMH', 'Speed_MPH': Frequently missing or zero (see outlier analysis below). Critical for serve speed analysis.")
        print("- 'WinnerShotType', 'ServeWidth', 'ServeDepth', 'ReturnDepth', etc.: These advanced metrics are often sparsely populated.")
        print("- 'P1DistanceRun', 'P2DistanceRun': Also can be sparse. Presence depends on tracking system capabilities.")
        print("- Core identifiers like 'match_id', 'PointNumber', 'PointWinner', 'PointServer' should ideally have no missing values.")

        print("\n--- 3.2.4. Univariate Analysis & Outlier Investigation (Slam Points) ---")
        numerical_point_cols = ['ElapsedTime', 'SetNo', 'P1GamesWon', 'P2GamesWon', 'GameNo', 'PointNumber', 
                                'Speed_KMH', 'Rally', 'ServeNumber', 'P1PointsWon', 'P2PointsWon',
                                'P1DistanceRun', 'P2DistanceRun', 'RallyCount']
        
        print("\nVisualizing distributions and investigating outliers for numerical point columns...")
        print("Boxplots are useful for identifying statistical outliers.")

        # Specific investigation for Speed_KMH == 0
        if 'Speed_KMH' in df_slam_points.columns and pd.api.types.is_numeric_dtype(df_slam_points['Speed_KMH']):
            print("\nInvestigating 'Speed_KMH' for zero values:")
            speed_zero_count = (df_slam_points['Speed_KMH'] == 0).sum()
            speed_total_count = df_slam_points['Speed_KMH'].notna().sum()
            if speed_total_count > 0:
                speed_zero_percent = (speed_zero_count / speed_total_count) * 100
                print(f"Number of points with Speed_KMH == 0: {speed_zero_count} (out of {speed_total_count} non-NaN speeds, {speed_zero_percent:.2f}%)")
            else:
                print("No non-NaN Speed_KMH values to analyze for zeros.")

            # Plot distribution of Speed_KMH excluding 0s
            plt.figure(figsize=(10, 4))
            sns.histplot(df_slam_points[df_slam_points['Speed_KMH'] > 0]['Speed_KMH'].dropna(), kde=True, bins=50)
            plt.title('Distribution of Speed_KMH (excluding 0 values)')
            plt.xlabel('Speed_KMH (>0)')
            plt.ylabel('Frequency')
            plt.show()
            print("Discussion: Speed_KMH == 0 likely indicates missing speed data, a fault, or a non-serve point where speed isn't measured. For serve speed analysis, these should be filtered or treated as NaN.")

        for col in numerical_point_cols:
            if col in df_slam_points.columns and pd.api.types.is_numeric_dtype(df_slam_points[col]):
                # Skip Speed_KMH full plot as we handled it above with a focus on zeros
                if col == 'Speed_KMH': 
                    plt.figure(figsize=(10, 2))
                    sns.boxplot(x=df_slam_points[df_slam_points['Speed_KMH'] > 0]['Speed_KMH'].dropna())
                    plt.title(f'Boxplot of {col} (excluding 0 values)')
                    plt.xlabel(col)
                    plt.show()
                    continue # Move to next column
                
                plt.figure(figsize=(10, 4))
                sns.histplot(df_slam_points[col].dropna(), kde=False, bins=50)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.show()

                plt.figure(figsize=(10, 2))
                sns.boxplot(x=df_slam_points[col].dropna())
                plt.title(f'Boxplot of {col} (shows outliers)')
                plt.xlabel(col)
                plt.show()
            elif col in df_slam_points.columns:
                 print(f"Column '{col}' is not numeric, skipping histogram/boxplot.")
            else:
                print(f"Column {col} not found for plotting.")
        plt.close('all')
        print("\nOutlier Discussion (Slam Points):")
        print("- 'ElapsedTime': Could have outliers if recording errors occurred.")
        print("- 'Rally', 'RallyCount': Very long rallies are possible but extreme values warrant scrutiny.")
        print("- 'P1DistanceRun', 'P2DistanceRun': Similar to Rally, extreme values might be data issues.")
        print("Other columns like PointNumber, GameNo, SetNo should follow logical sequences; outliers here would indicate structural issues.")

        categorical_point_cols = ['PointWinner', 'PointServer', 'SetWinner', 'GameWinner', 
                                  'P1Score', 'P2Score', 'ServeIndicator', 'WinnerShotType'] # From observed columns
        print("\nVisualizing distributions for key categorical point columns (Slam Points)...")
        for col in categorical_point_cols:
            if col in df_slam_points.columns:
                plt.figure(figsize=(10, 5))
                top_n = 15 if df_slam_points[col].nunique() > 15 else df_slam_points[col].nunique()
                order = df_slam_points[col].value_counts().nlargest(top_n).index
                sns.countplot(y=df_slam_points[col].astype(str), order=order, palette="mako") # astype(str) for mixed types
                plt.title(f'Frequency of {col} (Top {top_n})')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column {col} not found for plotting.")
        plt.close('all')
        
        print("\n--- 3.2.5. Linking with Slam Match Data (Conceptual) ---")
        if 'match_id' in df_slam_points.columns:
            print("The 'match_id' column is present in Slam Points data.")
            print(f"Number of unique match_ids in this sample: {df_slam_points['match_id'].nunique()}")
            print("This column can be used to link with 'aggregated_slam_matches.csv' (which also should have 'match_id').")
            # Example: If df_slam_matches was loaded:
            # if df_slam_matches is not None and 'match_id' in df_slam_matches.columns:
            #     merged_slam_sample = pd.merge(df_slam_points.head(1000), df_slam_matches, on='match_id', how='left', suffixes=['_point', '_match'])
            #     print(f"Sample merge of 1000 points with slam matches resulted in {merged_slam_sample.shape[0]} rows.")
            #     print(merged_slam_sample.head())
        else:
            print("'match_id' column not found in the Slam Points data sample.")

    return df_slam_points

# %% [markdown]
# ## Part 4: Cross-Dataset Consistency and Linkage

# %% 
def explore_cross_dataset_consistency(df_atp_matches, df_atp_pbp, df_slam_matches, df_slam_points):
    print("\n--- Part 4: Cross-Dataset Consistency and Linkage ---")

    if df_atp_matches is None or df_atp_pbp is None or df_slam_matches is None or df_slam_points is None:
        print("One or more dataframes not available. Skipping cross-dataset consistency checks.")
        print("Please ensure df_atp_matches, df_atp_pbp, df_slam_matches, and df_slam_points are loaded.")
        return

    # --- 4.1. Player Name/ID Consistency ---
    print("\n--- 4.1. Player Name/ID Consistency ---")

    # ATP Matches Player Info
    print("\nATP Matches Player Info:")
    if 'winner_name' in df_atp_matches.columns and 'loser_name' in df_atp_matches.columns:
        print("Sample winner names (ATP Matches):\n", df_atp_matches['winner_name'].dropna().sample(min(5, len(df_atp_matches['winner_name'].dropna())), random_state=1).tolist())
    if 'winner_id' in df_atp_matches.columns and 'loser_id' in df_atp_matches.columns:
        print(f"Unique winner_id count (ATP Matches): {df_atp_matches['winner_id'].nunique()}")
        print(f"Missing winner_id count (ATP Matches): {df_atp_matches['winner_id'].isnull().sum()}")

    # ATP PBP Player Info
    print("\nATP PBP Player Info:")
    if 'server1' in df_atp_pbp.columns and 'server2' in df_atp_pbp.columns:
        print("Sample server1 names (ATP PBP):\n", df_atp_pbp['server1'].dropna().sample(min(5, len(df_atp_pbp['server1'].dropna())), random_state=1).tolist())
        # ATP PBP data doesn't seem to have explicit player IDs in the raw files, relies on names.

    # Slam Matches Player Info
    print("\nSlam Matches Player Info:")
    if 'player1' in df_slam_matches.columns and 'player2' in df_slam_matches.columns:
        print("Sample player1 names (Slam Matches):\n", df_slam_matches['player1'].dropna().sample(min(5, len(df_slam_matches['player1'].dropna())), random_state=1).tolist())
    if 'player1id' in df_slam_matches.columns and 'player2id' in df_slam_matches.columns:
        print(f"Unique player1id count (Slam Matches): {df_slam_matches['player1id'].nunique()}")
        print(f"Missing player1id count (Slam Matches): {df_slam_matches['player1id'].isnull().sum()}")
    
    print("\nDiscussion on Player Identification:")
    print("- ATP Matches have 'winner_id', 'loser_id'. Slam Matches have 'player1id', 'player2id'. These ID systems might be different.")
    print("- ATP PBP primarily uses names ('server1', 'server2').")
    print("- Player name formats vary (e.g., 'First Last', 'Last F.', 'Last First'). Robust matching requires careful standardization (more than the basic attempt in Part 2.5). Libraries like `fuzzywuzzy` or custom logic might be needed.")
    print("- Goal for cleaning: Create a unique player master ID and map all player instances across datasets to it.")

    # --- 4.2. Match Identification --- 
    print("\n--- 4.2. Match Identification ---")
    print("\nRecap of Match Linkage Strategies:")
    print("- ATP PBP to ATP Matches: Attempted using a composite key (date, tourney_name_lower, sorted_standardized_player_names). Success depends heavily on name/tournament standardization.")
    print("  Example PBP match key (first record):", df_atp_pbp['match_key_pbp'].iloc[0] if 'match_key_pbp' in df_atp_pbp and not df_atp_pbp.empty else "N/A")
    
    print("- Slam Points to Slam Matches: Direct link via 'match_id'. This is generally reliable.")
    print("  Example Slam Points match_id (first record):", df_slam_points['match_id'].iloc[0] if 'match_id' in df_slam_points and not df_slam_points.empty else "N/A")
    print("  Unique match_ids in Slam Matches sample:", df_slam_matches['match_id'].nunique() if 'match_id' in df_slam_matches else "N/A")
    print("  Unique match_ids in Slam Points sample:", df_slam_points['match_id'].nunique() if 'match_id' in df_slam_points else "N/A")

    print("\nLinking ATP Tour Matches with Grand Slam Matches:")
    print("- This is challenging due to different tournament naming/ID conventions and data sources.")
    print("- Primary linking factors would be player names (standardized) and match dates (allowing for date ranges for a tournament).")
    print("- For example, finding all matches involving 'Roger Federer' in `df_atp_matches` around the date of the 'Wimbledon' tournament in `df_slam_matches`.")
    print("- This often requires external tournament schedule information or very careful fuzzy matching of tournament names and date ranges.")
    
    print("\nColumn Comparison: ATP Matches vs Slam Matches (already in main block, reiterated here for completeness)")
    atp_cols = set(df_atp_matches.columns)
    slam_cols = set(df_slam_matches.columns)
    print(f"Columns unique to ATP Matches: {atp_cols - slam_cols}")
    print(f"Columns unique to Slam Matches: {slam_cols - atp_cols}")
    print(f"Common columns: {atp_cols.intersection(slam_cols)}")
    print("Differences in common columns (e.g., player IDs, stats available) are key for deciding how to merge or use data together.")

    print("\n--- Part 4 Exploration Complete ---")

# %% [markdown]
# ## Main Execution Block

# %% 
if __name__ == '__main__':
    print("Starting EDA exploration using aggregated files...")
    
    # %% [markdown]
    # ### Loading Data (or ensuring it's loaded for notebook context)
    # %% 
    # In a script, these would run sequentially. In a notebook, you might run these cells once.
    df_atp_matches = explore_atp_match_data()
    df_atp_pbp = explore_atp_pbp_data()
    df_slam_matches = explore_slam_match_data()
    df_slam_points = explore_slam_points_data()

    # %% [markdown]
    # ### Executing Part 4: Cross-Dataset Consistency and Linkage
    # %% 
    explore_cross_dataset_consistency(df_atp_matches, df_atp_pbp, df_slam_matches, df_slam_points)

    print("\nEDA exploration functions executed. Review the output above.")
    # The cross-dataset column comparison for ATP vs Slam matches is also in the main block:
    if df_atp_matches is not None and df_slam_matches is not None:
        print("\n--- Cross-Dataset Column Comparison (ATP Matches vs Slam Matches) - From Main Block ---")
        atp_cols_main = set(df_atp_matches.columns)
        slam_cols_main = set(df_slam_matches.columns)
        print(f"Columns unique to ATP Matches: {atp_cols_main - slam_cols_main}")
        print(f"Columns unique to Slam Matches: {slam_cols_main - atp_cols_main}")
        print(f"Common columns: {atp_cols_main.intersection(slam_cols_main)}")
    
    print("\nNext steps would involve deeper dives into specific features, visualizations, and building a robust data cleaning/preprocessing pipeline.")
    pass

# %% [markdown]
# # EDA Summary and Achievements
# 
# This EDA process aimed to understand the structure, content, quality, and relationships within the various tennis datasets provided.
# 
# ## Achievements by Part:
# 
# ### Setup & Aggregation (Implicit, prior to this script focus)
# - All individual CSV files within `data/atp_matches/`, `data/atp_point_by_point/`, and `data/slam_point_by_point/` were aggregated into five main pandas DataFrames:
#   - `aggregated_atp_matches.csv`
#   - `aggregated_pbp_matches.csv`
#   - `aggregated_slam_matches.csv`
#   - `aggregated_slam_points.csv` (though `raw_slam_pbp` had two types, these were aggregated from many files each)
# - This step simplified data loading and provided a consolidated view for analysis.
# 
# ### Part 1: ATP Match Data (`aggregated_atp_matches.csv`)
# - **Initial Inspection**: Loaded data, viewed shape, head, info, and descriptive statistics.
# - **Missing Values**: Identified columns with missing data (e.g., player entries, seeds, heights, minutes, serve stats, ranks) and discussed their potential implications and handling.
# - **Univariate Analysis**: 
#   - Explored distributions of categorical variables (surface, tournament level, round, best_of) using value counts and countplots.
#   - Analyzed distributions of key numerical variables (minutes, ages, heights, ranks, serve stats) using descriptive stats, histograms, and boxplots.
# - **Outlier Discussion**: Highlighted how boxplots can identify statistical outliers and discussed context for interpreting them (e.g., very long matches, extreme player attributes, data errors).
# - **Bivariate Analysis**: 
#   - Generated a correlation matrix heatmap for key numerical variables.
#   - Plotted winner rank vs. loser rank (scatter), winner aces by surface (boxplot), and match duration by best_of sets (boxplot).
# - **Sanity Checks**: Performed basic checks like upset percentages by rank points and frequency of score formats.
# 
# ### Part 2: ATP Point-by-Point Data (`aggregated_pbp_matches.csv`)
# - **README Integration**: Leveraged the provided README to understand PBP string encoding (`S`, `R`, `A`, `D`) and delimiters (`.` for sets, `;` for games, `/` for tiebreak serve changes).
# - **PBP String Exploration**: 
#   - Verified unique characters in PBP strings against README specs.
#   - Calculated and analyzed `pbp_length`.
#   - Parsed PBP strings to infer `num_sets_pbp` and `num_games_pbp`, visualizing their distributions.
# - **`adf_flag` Analysis**: Investigated the distribution of the `adf_flag` (indicates if aces/DFs are noted in PBP).
# - **Missing Values**: Analyzed missing data, emphasizing the impact of missing `pbp` or key linking fields, and discussed implications.
# - **Relationship with Match Data**: Attempted to create a composite `match_key` (date, tournament, standardized player names) to link with `aggregated_atp_matches.csv`. Highlighted the challenges of name and tournament standardization and showed a sample merge result/diagnosis.
# 
# ### Part 3: Slam Data
# #### 3.1 Slam Match Data (`aggregated_slam_matches.csv`)
# - **Initial Inspection & Univariate Analysis**: Loaded data, viewed structure. Generated countplots for categorical variables (`slam`, `round`, `status`, `event_name`, `court_name`) and for `year`.
# - **Missing Values**: Identified and discussed missing data points (e.g., `status`, `court_id`).
# 
# #### 3.2 Slam Points Data (`aggregated_slam_points.csv`)
# - **Initial Inspection & Univariate Analysis**: Loaded a sample, viewed structure. 
#   - Generated histograms and boxplots for key numerical point-level variables (e.g., `ElapsedTime`, `Speed_KMH`, `Rally`, `ServeNumber`, `DistanceRun`).
#   - Generated countplots for categorical point-level variables (e.g., `PointWinner`, `PointServer`, `P1Score`, `WinnerShotType`).
# - **Missing Values & Outlier Investigation**: 
#   - Discussed common missing values (e.g., `Speed_KMH`, advanced metrics like `WinnerShotType`, `ServeWidth`).
#   - Specifically investigated `Speed_KMH == 0` values, calculating their frequency and plotting the distribution of speeds excluding zeros. Discussed interpretation of these zero values.
#   - General discussion on interpreting outliers from boxplots for other numerical columns.
# - **Linking**: Confirmed the presence of `match_id` for reliable linkage with `aggregated_slam_matches.csv`.
# 
# ### Part 4: Cross-Dataset Consistency and Linkage (This Implementation)
# - **Player Identification**: 
#   - Compared player name and ID columns across `df_atp_matches` (`winner/loser_name`, `winner/loser_id`), `df_atp_pbp` (`server1/2`), and `df_slam_matches` (`player1/2`, `player1/2id`).
#   - Discussed variations in name formats and the potential differences in ID systems, emphasizing the need for robust standardization for effective player tracking.
# - **Match Identification**:
#   - Recapped linkage strategies: composite key for ATP PBP to ATP Matches, `match_id` for Slam Points to Slam Matches.
#   - Discussed challenges in linking ATP tour matches with Grand Slam matches (different tournament names/IDs, reliance on player names and dates).
# - **Schema Comparison**: Compared column sets between `df_atp_matches` and `df_slam_matches` to identify unique and common fields, aiding in understanding how these datasets complement or differ from each other.
# 
# Overall, this EDA provides a solid foundation for subsequent data cleaning, feature engineering, and more targeted analysis or modeling tasks.


# %%
