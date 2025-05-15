import duckdb
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict

# Define constants for Elo calculation (as per blueprint)
INITIAL_RATING = 1500.0
MIN_K_FACTOR = 10 # Minimum K-factor (can be adjusted)
MAX_K_FACTOR = 40 # Maximum K-factor cap
K_FACTOR_DECAY = 250 # Factor for K-factor calculation based on matches played


def _create_input_view(con: duckdb.DuckDBPyConnection):
    """Creates the necessary input view from matches_all if it doesn't exist."""
    # View still includes surface, might be useful for other analysis later
    # but the Elo calculation itself won't use it directly for splitting.
    con.execute("""
        CREATE SCHEMA IF NOT EXISTS analytics;
    """)
    # Use the fully qualified name suggested by the error message
    # Use strptime for YYYYMMDD format
    con.execute("""
        CREATE OR REPLACE VIEW analytics.match_results AS
        SELECT
            match_id,
            EXTRACT(YEAR FROM strptime(CAST(tourney_date AS VARCHAR), '%Y%m%d'))::INTEGER AS season,
            surface, -- Keep surface in view for potential filtering/other uses
            winner_id AS player_a,
            loser_id AS player_b,
            tourney_date -- Need date for sorting (YYYYMMDD integer sorts correctly)
        FROM tennis.main.matches_all 
        WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL;
    """)
    print("Created or replaced view analytics.match_results")


def _create_target_table(con: duckdb.DuckDBPyConnection):
    """Creates the target table for storing Elo ratings if it doesn't exist."""
    con.execute("""
        CREATE SCHEMA IF NOT EXISTS analytics;
    """)
    # Removed 'surface' column and from PRIMARY KEY
    con.execute("""
        CREATE TABLE IF NOT EXISTS analytics.elo_player_season (
            player_id       INTEGER,
            season          INTEGER,
            elo_start       FLOAT,     -- Elo at the start of the season (after previous year's adjustment)
            elo_end         FLOAT,     -- Elo at the end of the season (before adjustment)
            matches_played  INTEGER,   -- Total matches played in the season
            PRIMARY KEY (player_id, season)
        );
    """)
    print("Created or verified table analytics.elo_player_season (unified Elo)")


def calculate_unified_elo_ratings(con: duckdb.DuckDBPyConnection, start_year: int, end_year: int):
    """
    Calculates unified, era-adjusted Elo ratings for players (not surface-specific).

    Args:
        con: DuckDB connection object.
        start_year: The first year to process.
        end_year: The last year to process (inclusive).
    """
    _create_input_view(con)
    _create_target_table(con)

    all_years = range(start_year, end_year + 1)

    # --- Initialization ---
    # latest_ratings now stores {player_id: rating}
    latest_ratings = defaultdict(lambda: INITIAL_RATING)
    max_season_processed = -1
    try:
        count_result = con.execute("SELECT COUNT(*) FROM analytics.elo_player_season").fetchone()
        if count_result and count_result[0] > 0:
            max_season_result = con.execute("SELECT MAX(season) FROM analytics.elo_player_season").fetchone()
            if max_season_result and max_season_result[0] is not None:
                 max_season_processed = max_season_result[0]

            if max_season_processed >= start_year:
                print(f"Loading latest ratings from season {max_season_processed}...")
                # Fetch player_id and elo_end (no surface)
                latest_df = con.execute(f"""
                    SELECT player_id, elo_end
                    FROM analytics.elo_player_season
                    WHERE season = {max_season_processed}
                """).df()
                for r in latest_df.itertuples():
                    latest_ratings[r.player_id] = r.elo_end # Key is just player_id
                print(f"Loaded {len(latest_df)} ratings.")
                # Adjust start_year if we've already processed some years
                start_year = max(start_year, max_season_processed + 1)
                all_years = range(start_year, end_year + 1)
                if not all_years:
                    print("All requested years seem to be processed already.")
                    return
    except (duckdb.CatalogException, duckdb.BinderException):
         print("Table analytics.elo_player_season not found or schema mismatch. Starting fresh.")
         # Optional: Drop table if schema incompatible?
         # con.execute("DROP TABLE IF EXISTS analytics.elo_player_season;")
         # _create_target_table(con) # Recreate with correct schema
         max_season_processed = -1 # Ensure we start from the beginning
         latest_ratings = defaultdict(lambda: INITIAL_RATING)
         all_years = range(start_year, end_year + 1) # Reset years range based on input


    print(f"Calculating unified Elo ratings from {start_year} to {end_year}...")

    for year in all_years:
        print(f"Processing year: {year}")
        # Ratings at the start of this year (carried over from end of last year)
        year_start_ratings: Dict[int, float] = latest_ratings.copy()
        # Live ratings updated after each match within the year
        current_elo: Dict[int, float] = year_start_ratings.copy()
        # Matches played *this season* per player
        matches_played_season: Counter[int] = Counter()
        # Store results for bulk insert: (player_id, season, elo_start, elo_end, matches_played)
        season_player_data = []

        # Fetch all matches for the year, sorted by date
        query = f"""
            SELECT player_a, player_b
            FROM analytics.match_results
            WHERE season = {year}
            ORDER BY tourney_date;
        """
        try:
            df = con.execute(query).df()
        except (duckdb.CatalogException, duckdb.BinderException):
            print(f"Warning: View analytics.match_results not found or query failed for {year}. Skipping year.")
            continue
        except Exception as e:
             print(f"Error fetching data for {year}: {e}. Skipping year.")
             continue

        if df.empty:
            print(f"  No matches found for {year}.")
            # Need to handle players who didn't play but had a rating carry over
            # They should still be included in inflation adjustment later if needed
            # Or simply store their start_elo = end_elo = carried-over rating?
            # For simplicity now, we only store data for players who played.
            # Inflation adj below handles averaging only over active players.
            # But we need to make sure their ratings are carried over correctly.
            latest_ratings = current_elo.copy() # Carry over unchanged ratings
            continue # Skip to next year if no matches

        print(f"  Processing {len(df)} matches...")
        for r in df.itertuples():
            p_a = r.player_a
            p_b = r.player_b

            # Get current Elo ratings
            Ra = current_elo[p_a]
            Rb = current_elo[p_b]

            # Calculate expected scores
            Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
            Eb = 1 - Ea

            # Calculate K-factors (using total matches played *this season*)
            K_a = min(MAX_K_FACTOR, K_FACTOR_DECAY / (matches_played_season[p_a] + 1)**0.5)
            K_b = min(MAX_K_FACTOR, K_FACTOR_DECAY / (matches_played_season[p_b] + 1)**0.5)
            K_a = max(MIN_K_FACTOR, K_a) # Ensure minimum K
            K_b = max(MIN_K_FACTOR, K_b) # Ensure minimum K

            # Update Elo ratings *immediately*
            new_Ra = Ra + K_a * (1 - Ea)
            new_Rb = Rb + K_b * (0 - Eb) # 0 - Eb = -Eb

            current_elo[p_a] = new_Ra
            current_elo[p_b] = new_Rb

            # Increment matches played count for the season
            matches_played_season[p_a] += 1
            matches_played_season[p_b] += 1

        # --- Store Season Snapshot ---
        # Gather data for all players who played this season
        active_players = set(matches_played_season.keys())
        for p_id in active_players:
            start_elo = year_start_ratings.get(p_id, INITIAL_RATING) # Use initial if player was new
            end_elo = current_elo[p_id]
            count = matches_played_season[p_id]
            season_player_data.append((p_id, year, start_elo, end_elo, count))

        # Include players who had a rating but didn't play this year?
        # Their start/end elo would be the same carried-over value.
        # For now, only storing players who played. Inflation adj handles this.

        # Bulk insert season data
        if season_player_data:
            print(f"  Inserting {len(season_player_data)} player-season records for {year}...")
            df_to_insert = pd.DataFrame(season_player_data, columns=[
                'player_id', 'season', 'elo_start', 'elo_end', 'matches_played'
            ])
            try:
                # Use INSERT OR REPLACE to handle reruns/updates
                con.execute("INSERT OR REPLACE INTO analytics.elo_player_season BY NAME SELECT * FROM df_to_insert;")
            except Exception as e:
                print(f"Error inserting data for year {year}: {e}")
                # Decide on error handling: stop, log, continue?
                continue # Continue to inflation adjustment for now
        else:
             print(f"  No player data processed for {year} to insert.")


        # --- Inflation Adjustment ---
        # Recenter ratings *after* the full year is processed and stored
        print(f"  Applying inflation adjustment for {year}...")
        adjusted_player_elos: Dict[int, float] = {} # Store adjusted values to update latest_ratings

        try:
            # Calculate mean elo_end for the current year across all players who played
            mean_query = f"""
                SELECT AVG(elo_end)
                FROM analytics.elo_player_season
                WHERE season = {year} AND matches_played > 0;
            """
            result = con.execute(mean_query).fetchone()
            mean_elo_end = result[0] if result and result[0] is not None else INITIAL_RATING

            # Check if any players actually played
            player_count_query = f"SELECT COUNT(*) FROM analytics.elo_player_season WHERE season = {year} AND matches_played > 0"
            players_in_season = con.execute(player_count_query).fetchone()[0]

            if players_in_season == 0:
                 print(f"    No players found who played in {year} to adjust. Skipping adjustment.")
                 # Still need to carry over ratings correctly
                 latest_ratings = current_elo.copy() # Carry over potentially updated ratings if logic changes
                 continue # Skip to next year

            adjustment_value = mean_elo_end - INITIAL_RATING # Amount to subtract
            print(f"    Year {year}: Mean Elo = {mean_elo_end:.2f}, Adjustment = {-adjustment_value:.2f}")

            # Update the table for the current year
            update_query = f"""
                UPDATE analytics.elo_player_season
                SET elo_start = elo_start - ?,
                    elo_end = elo_end - ?
                WHERE season = ?;
            """
            con.execute(update_query, (adjustment_value, adjustment_value, year))

            # Fetch the adjusted end-of-year ratings to carry over to the next year
            adjusted_df = con.execute(f"""
                 SELECT player_id, elo_end
                 FROM analytics.elo_player_season
                 WHERE season = {year}
            """).df()

            # Update the dictionary holding the latest ratings for *all* players processed this year
            temp_adjusted_ratings = {}
            for r_adj in adjusted_df.itertuples():
                temp_adjusted_ratings[r_adj.player_id] = r_adj.elo_end

            # Carry over ratings for players who had a rating but didn't play this year
            # Their rating should also be adjusted.
            final_carry_over = year_start_ratings.copy() # Start with previous year's ratings
            for pid, rating in final_carry_over.items():
                 if pid not in temp_adjusted_ratings: # If player didn't play this year
                      final_carry_over[pid] = rating - adjustment_value # Adjust their carried-over rating
                 else: # If player did play, use their adjusted rating from DB
                      final_carry_over[pid] = temp_adjusted_ratings[pid]

            # Add any *new* players from this year
            for pid, rating in temp_adjusted_ratings.items():
                 if pid not in final_carry_over:
                      final_carry_over[pid] = rating # Already adjusted


            latest_ratings = defaultdict(lambda: INITIAL_RATING, final_carry_over)
            print(f"  Updated latest_ratings with {len(latest_ratings)} adjusted values for carry-over.")


        except Exception as e:
            print(f"Error during inflation adjustment for {year}: {e}")
            # If adjustment fails, carry over the unadjusted end-of-year ratings
            latest_ratings = current_elo.copy()


    print("Unified Elo calculation finished.")


# --- Surface-Specific Elo Calculation --- #

def _create_surface_target_table(con: duckdb.DuckDBPyConnection):
    """Creates the target table for storing SURFACE-SPECIFIC Elo ratings."""
    con.execute("""
        CREATE SCHEMA IF NOT EXISTS analytics;
    """)
    # Target table includes 'surface' column and in PRIMARY KEY
    con.execute("""
        CREATE TABLE IF NOT EXISTS analytics.elo_player_season_surface (
            player_id       INTEGER,
            surface         VARCHAR,   -- 'Hard', 'Clay', 'Grass', 'Carpet' ?
            season          INTEGER,
            elo_start       FLOAT,     -- Elo at the start of the season (after previous year's adjustment)
            elo_end         FLOAT,     -- Elo at the end of the season (before adjustment)
            matches_played  INTEGER,   -- Matches played on this surface in the season
            PRIMARY KEY (player_id, surface, season)
        );
    """)
    print("Created or verified table analytics.elo_player_season_surface")


def calculate_surface_elo_ratings(con: duckdb.DuckDBPyConnection, start_year: int, end_year: int):
    """
    Calculates SURFACE-SPECIFIC, era-adjusted Elo ratings for players.
    Follows the original blueprint more closely.

    Args:
        con: DuckDB connection object.
        start_year: The first year to process.
        end_year: The last year to process (inclusive).
    """
    # Uses the same input view as the unified calculation
    _create_input_view(con)
    _create_surface_target_table(con)

    all_years = range(start_year, end_year + 1)
    # Determine surfaces dynamically from the input view for robustness
    try:
        surfaces_df = con.execute("SELECT DISTINCT surface FROM analytics.match_results WHERE season BETWEEN ? AND ?", [start_year, end_year]).df()
        surfaces = surfaces_df['surface'].dropna().tolist()
        print(f"Detected surfaces for Elo calculation: {surfaces}")
    except Exception as e:
        print(f"Warning: Could not detect surfaces dynamically ({e}). Falling back to default list.")
        surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']

    # --- Initialization ---
    # latest_ratings now stores {(player_id, surface): rating}
    latest_ratings = defaultdict(lambda: INITIAL_RATING)
    target_table = "analytics.elo_player_season_surface"
    max_season_processed = -1
    try:
        count_result = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()
        if count_result and count_result[0] > 0:
            max_season_result = con.execute(f"SELECT MAX(season) FROM {target_table}").fetchone()
            if max_season_result and max_season_result[0] is not None:
                 max_season_processed = max_season_result[0]

            if max_season_processed >= start_year:
                print(f"Loading latest surface ratings from season {max_season_processed}...")
                latest_df = con.execute(f"""
                    SELECT player_id, surface, elo_end
                    FROM {target_table}
                    WHERE season = {max_season_processed}
                """).df()
                for r in latest_df.itertuples():
                    latest_ratings[(r.player_id, r.surface)] = r.elo_end
                print(f"Loaded {len(latest_df)} surface ratings.")
                start_year = max(start_year, max_season_processed + 1)
                all_years = range(start_year, end_year + 1)
                if not all_years:
                    print("All requested years seem to be processed already for surface Elo.")
                    return
    except (duckdb.CatalogException, duckdb.BinderException):
         print(f"Table {target_table} not found or schema mismatch. Starting fresh for surface Elo.")
         max_season_processed = -1
         latest_ratings = defaultdict(lambda: INITIAL_RATING)
         all_years = range(start_year, end_year + 1)

    print(f"Calculating SURFACE-SPECIFIC Elo ratings from {start_year} to {end_year}...")

    for year in all_years:
        print(f"Processing year: {year}")
        # Ratings at the start of this year (carried over from end of last year, per surface)
        year_start_ratings = latest_ratings.copy()
        # Ratings to be updated within the year, per surface
        year_end_ratings = year_start_ratings.copy()
        # Store results for bulk insert: (player_id, surface, season, elo_start, elo_end, matches_played)
        season_surface_data = []

        for surface in surfaces:
            print(f"  Surface: {surface}")
            # Fetch matches for this year and surface, sorted by date
            query = f"""
                SELECT player_a, player_b
                FROM analytics.match_results
                WHERE season = {year} AND surface = ?
                ORDER BY tourney_date;
            """
            try:
                # Use parameterized query for surface
                df = con.execute(query, [surface]).df()
            except (duckdb.CatalogException, duckdb.BinderException):
                print(f"Warning: View analytics.match_results not found or query failed for {year}/{surface}. Skipping.")
                continue
            except Exception as e:
                 print(f"Error fetching data for {year}/{surface}: {e}. Skipping.")
                 continue

            if df.empty:
                print(f"    No matches found for {year}/{surface}.")
                continue

            # Elo ratings specific to this surface, initialized from year_end_ratings carry-over
            surface_elo = year_end_ratings.copy() # Use snapshot including previous surfaces this year
            # Matches played *this season on this surface* per player
            matches_played_season_surface = Counter()

            print(f"    Processing {len(df)} matches...")
            for r in df.itertuples():
                p_a = r.player_a
                p_b = r.player_b
                key_a = (p_a, surface)
                key_b = (p_b, surface)

                # Get current Elo ratings for this surface
                Ra = surface_elo[key_a]
                Rb = surface_elo[key_b]

                # Calculate expected scores
                Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
                Eb = 1 - Ea

                # Calculate K-factors (using matches played *this season on this surface*)
                K_a = min(MAX_K_FACTOR, K_FACTOR_DECAY / (matches_played_season_surface[key_a] + 1)**0.5)
                K_b = min(MAX_K_FACTOR, K_FACTOR_DECAY / (matches_played_season_surface[key_b] + 1)**0.5)
                K_a = max(MIN_K_FACTOR, K_a)
                K_b = max(MIN_K_FACTOR, K_b)

                # Update Elo ratings *immediately*
                new_Ra = Ra + K_a * (1 - Ea)
                new_Rb = Rb + K_b * (0 - Eb)

                surface_elo[key_a] = new_Ra
                surface_elo[key_b] = new_Rb

                # Increment matches played count for the season/surface
                matches_played_season_surface[key_a] += 1
                matches_played_season_surface[key_b] += 1

            # After processing all matches for the surface, store the results for this surface
            processed_players_surface = set(matches_played_season_surface.keys())
            for player_surf_tuple, count in matches_played_season_surface.items():
                 p_id, surf = player_surf_tuple
                 # Get start_elo from the beginning of the year for this player/surface
                 start_elo = year_start_ratings.get(player_surf_tuple, INITIAL_RATING)
                 end_elo = surface_elo[player_surf_tuple] # End elo is the final calculated value
                 season_surface_data.append((p_id, surf, year, start_elo, end_elo, count))

            # Update the main year_end_ratings dict with results from this surface
            # This ensures the next surface starts with the updated ratings
            year_end_ratings.update(surface_elo)

        # --- End Surface Loop ---

        # Bulk insert season data after processing all surfaces for the year
        if season_surface_data:
            print(f"  Inserting {len(season_surface_data)} player-surface-season records for {year}...")
            df_to_insert = pd.DataFrame(season_surface_data, columns=[
                'player_id', 'surface', 'season', 'elo_start', 'elo_end', 'matches_played'
            ])
            try:
                con.execute(f"INSERT OR REPLACE INTO {target_table} BY NAME SELECT * FROM df_to_insert;")
            except Exception as e:
                print(f"Error inserting surface Elo data for year {year}: {e}")
                continue
        else:
             print(f"  No surface match data processed for {year} to insert.")

        # --- Inflation Adjustment (Surface Specific) ---
        print(f"  Applying surface-specific inflation adjustment for {year}...")
        adjusted_surface_elos = {} # Store adjusted values for carry-over

        for surface in surfaces:
            try:
                # Calculate mean elo_end for this year and surface
                mean_query = f"""
                    SELECT AVG(elo_end)
                    FROM {target_table}
                    WHERE season = ? AND surface = ? AND matches_played > 0;
                """
                result = con.execute(mean_query, [year, surface]).fetchone()
                mean_elo_end = result[0] if result and result[0] is not None else INITIAL_RATING

                player_count_query = f"SELECT COUNT(*) FROM {target_table} WHERE season = ? AND surface = ? AND matches_played > 0"
                players_in_season_surface = con.execute(player_count_query, [year, surface]).fetchone()[0]

                if players_in_season_surface == 0:
                     print(f"    No players found for {surface} in {year} to adjust. Skipping surface adjustment.")
                     continue

                adjustment_value = mean_elo_end - INITIAL_RATING
                print(f"    Surface '{surface}': Mean Elo = {mean_elo_end:.2f}, Adjustment = {-adjustment_value:.2f}")

                update_query = f"""
                    UPDATE {target_table}
                    SET elo_start = elo_start - ?,
                        elo_end = elo_end - ?
                    WHERE season = ? AND surface = ?;
                """
                con.execute(update_query, (adjustment_value, adjustment_value, year, surface))

                # Fetch adjusted end-of-year ratings for this surface
                adjusted_df = con.execute(f"""
                     SELECT player_id, elo_end
                     FROM {target_table}
                     WHERE season = ? AND surface = ?
                 """, [year, surface]).df()

                for r_adj in adjusted_df.itertuples():
                    adjusted_surface_elos[(r_adj.player_id, surface)] = r_adj.elo_end

            except Exception as e:
                print(f"Error during surface inflation adjustment for {year}/{surface}: {e}")

        # Update latest_ratings with adjusted values for carry-over to next year
        # Need to handle players who had ratings but didn't play/weren't adjusted
        final_carry_over = year_start_ratings.copy()
        final_carry_over.update(adjusted_surface_elos) # Overwrite with adjusted where available

        # Re-apply adjustment logic for those not directly updated (had rating but didn't play on any surface?)
        # This part is tricky. Simpler approach: just use the adjusted values we fetched.
        # If a player didn't play on surface X this year, their carry-over from year_start_ratings is used,
        # but it *won't* reflect the adjustment made this year on other surfaces.
        # For now, directly update latest_ratings only with players processed/adjusted this year.
        # A truly robust carry-over would need careful handling of adjustments across surfaces.

        latest_ratings = defaultdict(lambda: INITIAL_RATING)
        latest_ratings.update(final_carry_over) # Use the merged dict

        # Corrected approach: Update based on adjusted_surface_elos which contains *only* adjusted values
        # latest_ratings = defaultdict(lambda: INITIAL_RATING) # Start fresh for carry-over dict
        # latest_ratings.update(adjusted_surface_elos) # Populate ONLY with adjusted values
        # This might lose players who were inactive but should still be carried over. Let's stick with the update approach.

        if adjusted_surface_elos: # Check if any adjustments were made
             print(f"  Updated latest_surface_ratings with {len(adjusted_surface_elos)} adjusted values for carry-over.")
        # Carry over non-adjusted players from year_end_ratings? No, year_start_ratings.
        # Stick to the simple update: latest_ratings = final_carry_over

    # --- End Year Loop ---
    print("Surface-Specific Elo calculation finished.")


# Example usage (assuming you have a duckdb connection `con`)
if __name__ == "__main__":
    db_path = 'data/tennis.duckdb' # Replace with your DB path
    con = duckdb.connect(database=db_path, read_only=False)

    try:
        # --- REMOVE TEMPORARY DEBUGGING CODE --- 
        # (Debug code was here)
        # --- END REMOVE TEMPORARY DEBUGGING CODE ---

        # Determine year range from data if needed
        # Use strptime for YYYYMMDD format
        min_year_res = con.execute("SELECT MIN(EXTRACT(YEAR FROM strptime(CAST(tourney_date AS VARCHAR), '%Y%m%d'))) FROM tennis.main.matches_all WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL").fetchone()
        max_year_res = con.execute("SELECT MAX(EXTRACT(YEAR FROM strptime(CAST(tourney_date AS VARCHAR), '%Y%m%d'))) FROM tennis.main.matches_all WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL").fetchone()
        min_year = min_year_res[0] if min_year_res else None
        max_year = max_year_res[0] if max_year_res else None

        if min_year and max_year:
             print(f"Detected year range in data: {min_year} - {max_year}")
             process_start_year = min_year
             process_end_year = max_year

             # --- !! DANGER ZONE !! ---
             # Uncomment below ONLY if you want a complete reset for the unified calculation
             # print("WARNING: Dropping existing elo_player_season table for a full rerun.")
             # con.execute("DROP TABLE IF EXISTS analytics.elo_player_season;") 
             # --- End Danger Zone ---

             calculate_unified_elo_ratings(con, process_start_year, process_end_year)
        else:
             print("Could not determine year range from tennis.main.matches_all table.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        print("Closing database connection.")
        con.close() 