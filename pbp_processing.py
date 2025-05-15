from pathlib import Path
import csv
import duckdb
from tqdm import tqdm

import constants as C
import utils as U # Need this for display_path

def _parse_pbp_string(pbp_string: str, server1_id: int, server2_id: int) -> list[dict]:
    """Parses the PBP string into individual point data."""
    points = []
    set_no, game_no, point_in_game = 1, 1, 0
    server_is_player1 = True  # Player 1 serves first game by default
    in_tiebreak = False
    tb_points_played = 0
    points_this_set = 0

    # Score tracking (raw points within game/tiebreak)
    game_score1, game_score2 = 0, 0
    tb_score1, tb_score2 = 0, 0

    for i, char in enumerate(pbp_string):
        if char == ";":  # End of Game
            game_no += 1
            point_in_game = 0
            server_is_player1 = not server_is_player1
            in_tiebreak = False
            tb_points_played = 0
            game_score1, game_score2 = 0, 0 # Reset game score
            tb_score1, tb_score2 = 0, 0   # Reset TB score (just in case)
            continue
        if char == ".":  # End of Set
            set_no += 1
            game_no = 1
            point_in_game = 0
            server_is_player1 = True # Player 1 serves first game of new set
            in_tiebreak = False
            tb_points_played = 0
            points_this_set = 0
            game_score1, game_score2 = 0, 0 # Reset game score
            tb_score1, tb_score2 = 0, 0   # Reset TB score
            continue
        if char == "/": # Start of Tiebreak (or server switch within TB)
            in_tiebreak = True
            server_is_player1 = not server_is_player1
            # Don't reset scores here, TB continues across the marker
            continue

        if char in "SRAD":
            point_idx = len(points)
            server_won = char in "SA"
            is_ace = char == "A"
            is_df = char == "D"

            current_score_server = 0
            current_score_returner = 0

            if in_tiebreak:
                # Use TB scores directly
                current_score_server = tb_score1 if server_is_player1 else tb_score2
                current_score_returner = tb_score2 if server_is_player1 else tb_score1

                # Update TB score *after* capturing current state
                if server_won:
                    if server_is_player1: tb_score1 += 1
                    else: tb_score2 += 1
                else:
                    if server_is_player1: tb_score2 += 1
                    else: tb_score1 += 1

                tb_points_played += 1
                # Determine server for *next* point (if needed for complex logic later)
                if tb_points_played == 1:
                     server_is_player1_next = False # P2 serves next
                elif (tb_points_played -1) % 4 < 2:
                     server_is_player1_next = True # P1 serves next (after P2 served 2,3 or 6,7 etc.)
                else:
                     server_is_player1_next = False # P2 serves next (after P1 served 1 or 4,5 etc.)
                # Server switch via '/' marker overrides this logic
                if i + 1 < len(pbp_string) and pbp_string[i+1] == '/':
                    pass # Already handled by '/' block setting server_is_player1
                else:
                    # Update server state for the *next* point if no explicit switch
                    # NOTE: server_is_player1 already reflects the *current* server
                    # We don't need to change it here based on tb_points_played
                    pass
            else:
                # Use Game scores directly (0, 1, 2, 3... corresponding to 0, 15, 30, 40...)
                current_score_server = game_score1 if server_is_player1 else game_score2
                current_score_returner = game_score2 if server_is_player1 else game_score1

                # Update game score *after* capturing current state
                if server_won:
                    if server_is_player1: game_score1 += 1
                    else: game_score2 += 1
                else:
                    if server_is_player1: game_score2 += 1
                    else: game_score1 += 1

            point_data = {
                "point_idx": point_idx,
                "set_no": set_no,
                "game_no": game_no,
                "point_in_game": point_in_game,
                "server_id": server1_id if server_is_player1 else server2_id,
                "returner_id": server2_id if server_is_player1 else server1_id,
                "server_won": server_won,
                "is_ace": is_ace,
                "is_df": is_df,
                "is_tb": in_tiebreak,
                "score_server": current_score_server,    # Added
                "score_returner": current_score_returner, # Added
                "rally_len": None,
            }
            points.append(point_data)
            point_in_game += 1
            points_this_set += 1
        else:
            print(f"Warning: Unexpected character '{char}' in pbp_string near index {i}")

    return points


def parse_and_load_points(
    con: duckdb.DuckDBPyConnection,
    pbp_raw_dir: Path = C.PBP_RAW_DIR, # Use constant
    force_reload: bool = False,
) -> None:
    """Loads PBP CSVs, joins to find match_id, parses PBP strings, and loads points_fact table."""
    con.execute("CREATE SCHEMA IF NOT EXISTS points;")
    if force_reload:
        con.execute("DROP TABLE IF EXISTS points.points_fact;")
        con.execute("DROP TABLE IF EXISTS points.points_atp;") # Drop new name too
        con.execute("DROP TABLE IF EXISTS pbp_raw;") 
        con.execute("DROP TABLE IF EXISTS pbp_joined;")

    # Create the ATP-specific points table
    table_name = "points.points_atp"
    try:
        # Check if the new table exists
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if count > 0 and not force_reload:
            print(f"✔ {table_name} already exists with {count} rows. Skipping load.")
            return
        elif count > 0 and force_reload:
             # If force_reload, drop it (redundant with above drop, but safe)
            con.execute(f"DROP TABLE IF EXISTS {table_name};")

    except duckdb.CatalogException:
        pass # Table doesn't exist, proceed

    con.execute(f""" 
        CREATE TABLE {table_name} (
            match_id VARCHAR,
            point_idx INTEGER,
            set_no INTEGER,
            game_no INTEGER,
            point_in_game INTEGER,
            server_id INTEGER,
            returner_id INTEGER,
            server_won BOOLEAN,
            is_ace BOOLEAN,
            is_df BOOLEAN,
            is_tb BOOLEAN,
            score_server TINYINT,    -- Added
            score_returner TINYINT, -- Added
            rally_len INTEGER NULL,
            winner_name VARCHAR,
            loser_name VARCHAR,
            tourney_name VARCHAR,
            season INTEGER
        );
    """)

    pbp_files = list(pbp_raw_dir.glob("pbp_matches_atp_main_*.csv"))
    if not pbp_files:
        print("No PBP files found in", pbp_raw_dir)
        return

    print(f"Loading {len(pbp_files)} PBP files into temporary table...")
    pbp_glob = str(pbp_raw_dir / "pbp_matches_atp_main_*.csv")
    con.execute(f"CREATE OR REPLACE TEMP TABLE pbp_raw AS SELECT * FROM read_csv_auto('{pbp_glob}', header=True);")
    raw_count = con.execute("SELECT COUNT(*) FROM pbp_raw").fetchone()[0]
    if raw_count == 0:
        print("No data loaded into pbp_raw table.")
        return

    print("Joining PBP data with match metadata (using names and corrected date)...")
    try:
        join_sql = f"""
            CREATE OR REPLACE TEMP TABLE pbp_joined AS
            SELECT
                m.match_id,
                m.winner_id,
                m.loser_id,
                m.winner_name,
                m.loser_name,
                m.tourney_name,
                m.season,
                p.*,
                CASE
                    WHEN lower(p.server1) = lower(m.winner_name) THEN m.winner_id
                    WHEN lower(p.server1) = lower(m.loser_name) THEN m.loser_id
                    ELSE NULL
                END AS server1_id,
                 CASE
                    WHEN lower(p.server2) = lower(m.winner_name) THEN m.winner_id
                    WHEN lower(p.server2) = lower(m.loser_name) THEN m.loser_id
                    ELSE NULL
                END AS server2_id
            FROM pbp_raw p
            JOIN matches_all m
            ON (
                 (lower(p.server1) = lower(m.winner_name) AND lower(p.server2) = lower(m.loser_name))
                 OR
                 (lower(p.server1) = lower(m.loser_name) AND lower(p.server2) = lower(m.winner_name))
               )
            AND CAST(STRPTIME(p.date, '%d %b %y') AS DATE) >= CAST(STRPTIME(CAST(m.tourney_date AS VARCHAR), '%Y%m%d') AS DATE)
            AND CAST(STRPTIME(p.date, '%d %b %y') AS DATE) < CAST(STRPTIME(CAST(m.tourney_date AS VARCHAR), '%Y%m%d') AS DATE) + INTERVAL '8' DAY
            WHERE m.match_id IS NOT NULL;
        """
        con.execute(join_sql)
    except Exception as e:
        print(f"Error during PBP join: {e}")
        return

    join_count = con.execute("SELECT COUNT(*) FROM pbp_joined").fetchone()[0]
    print(f"Joined {join_count} / {raw_count} PBP matches.")
    if join_count == 0:
        print("Warning: No PBP matches could be joined.")
        return
    elif join_count < raw_count * 0.8:
        print(f"Warning: Only {join_count} / {raw_count} PBP matches were joined. Potential data inconsistencies.")

    try:
        # Added DISTINCT here to prevent processing duplicate PBP entries for the same match
        pbp_df = con.execute("SELECT DISTINCT match_id, pbp, server1_id, server2_id, winner_name, loser_name, tourney_name, season FROM pbp_joined WHERE server1_id IS NOT NULL AND server2_id IS NOT NULL").df()
    except Exception as e:
        print(f"Error fetching joined PBP data: {e}")
        con.execute("DROP TABLE IF EXISTS pbp_joined;")
        return

    if pbp_df.empty:
        print("Warning: No rows found in pbp_joined with non-NULL server IDs after join. Aborting parsing.")
        con.execute("DROP TABLE IF EXISTS pbp_joined;")
        return

    all_points_data = []
    processed_matches = 0
    skipped_matches = 0

    for _, row in tqdm(pbp_df.iterrows(), total=len(pbp_df), desc="Parsing matches"):
        match_id = row['match_id']
        pbp_string = row['pbp']
        winner_name = row['winner_name']
        loser_name = row['loser_name']
        tourney_name = row['tourney_name']
        season = row['season']
        try:
            s1_id = int(row['server1_id'])
            s2_id = int(row['server2_id'])
        except (ValueError, TypeError):
             skipped_matches += 1
             continue

        if not pbp_string or not isinstance(pbp_string, str):
            skipped_matches += 1
            continue

        try:
            points_for_match = _parse_pbp_string(pbp_string, s1_id, s2_id)
            for point_data in points_for_match:
                point_data['match_id'] = match_id
                point_data['winner_name'] = winner_name
                point_data['loser_name'] = loser_name
                point_data['tourney_name'] = tourney_name
                point_data['season'] = season
                all_points_data.append(point_data)
            processed_matches += 1
        except Exception as e:
            print(f"Error parsing PBP for match {match_id}: {e}. PBP: '{pbp_string[:50]}...'")
            skipped_matches += 1

    print(f"Parsed {processed_matches} matches, skipped {skipped_matches}. Generated {len(all_points_data)} points.")

    if not all_points_data:
        print("No point data generated. Aborting load.")
        con.execute("DROP TABLE IF EXISTS pbp_joined;")
        return

    temp_csv_path = C.PBP_RAW_DIR / "_temp_points.csv" # Use constant
    print(f"Writing {len(all_points_data)} points to temporary CSV: {temp_csv_path}")
    try:
        fieldnames = [
            'match_id', 'point_idx', 'set_no', 'game_no', 'point_in_game',
            'server_id', 'returner_id', 'server_won', 'is_ace', 'is_df',
            'is_tb', 'score_server', 'score_returner', 'rally_len',
            'winner_name', 'loser_name', 'tourney_name', 'season'
        ]
        with open(temp_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(all_points_data)

        print(f"Loading points into {table_name}...")
        con.execute(f"COPY {table_name} FROM '{temp_csv_path}' (HEADER, FORMAT CSV);")
        final_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✔ Loaded {final_count} points into {table_name}.")

    except Exception as e:
        print(f"Error during CSV write or COPY: {e}")
    finally:
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        con.execute("DROP TABLE IF EXISTS pbp_joined;")
        con.execute("DROP TABLE IF EXISTS pbp_raw;") 


# --- Rally Length Update ---

def update_rally_lengths(
    con: duckdb.DuckDBPyConnection,
    slam_pbp_raw_dir: Path = C.SLAM_PBP_RAW_DIR, # Now contains both points and matches files
) -> None:
    """Updates the rally_len column in points.points_fact using Slam PBP data and robust joining."""
    print("Attempting to update rally lengths using Slam PBP data (robust join)...")

    slam_points_files = list(slam_pbp_raw_dir.glob("*-points.csv"))
    slam_matches_files = list(slam_pbp_raw_dir.glob("*-matches.csv"))

    if not slam_points_files or not slam_matches_files:
        print(f"Missing Slam PBP points or matches files in {U._display_path(slam_pbp_raw_dir)}")
        print("Skipping rally length update.")
        return

    # Load Slam Points Data
    print(f"Loading {len(slam_points_files)} Slam PBP points files...")
    try:
        slam_points_glob = str(slam_pbp_raw_dir / "*-points.csv")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE slam_pbp_raw AS
            SELECT *,
                   TRY_CAST(regexp_extract(filename, '(\\d{{4}})', 1) AS INTEGER) AS slam_year,
                   regexp_extract(filename, '\\d{{4}}-([a-z]+)-points\\.csv', 1) AS slam_name_extracted
            FROM read_csv_auto('{slam_points_glob}', header=True, filename=true, all_varchar=True);
        """)
        raw_slam_points_count = con.execute("SELECT COUNT(*) FROM slam_pbp_raw").fetchone()[0]
        print(f"Loaded {raw_slam_points_count} raw Slam points.")
        if raw_slam_points_count == 0: raise ValueError("No points loaded")
    except Exception as e:
        print(f"Error loading Slam PBP points CSVs: {e}")
        return

    # Load Slam Matches Data
    print(f"Loading {len(slam_matches_files)} Slam Match files...")
    try:
        slam_matches_glob = str(slam_pbp_raw_dir / "*-matches.csv")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE slam_matches_raw AS
            SELECT "match_id", "player1" as slam_p1_name, "player2" as slam_p2_name
            FROM read_csv_auto('{slam_matches_glob}', header=True, all_varchar=True);
        """)
        raw_slam_matches_count = con.execute("SELECT COUNT(*) FROM slam_matches_raw").fetchone()[0]
        print(f"Loaded {raw_slam_matches_count} raw Slam matches metadata.")
        if raw_slam_matches_count == 0: raise ValueError("No matches loaded")
    except Exception as e:
        print(f"Error loading Slam Matches CSVs: {e}")
        con.execute("DROP TABLE IF EXISTS slam_pbp_raw;")
        return

    # Create enriched Slam details table
    print("Creating temporary Slam match details table...")
    try:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE slam_match_details AS
            SELECT
                sm.slam_p1_name,
                sm.slam_p2_name,
                sp.slam_year,
                sp.slam_name_extracted,
                TRY_CAST(sp."PointNumber" AS INTEGER) AS slam_point_number_int,
                sp."Rally"
            FROM slam_pbp_raw sp
            JOIN slam_matches_raw sm ON sp.match_id = sm.match_id
            WHERE sp.slam_year IS NOT NULL; -- Ensure we have a year
        """)
        details_count = con.execute("SELECT COUNT(*) FROM slam_match_details").fetchone()[0]
        print(f"Created slam_match_details with {details_count} rows.")
    except Exception as e:
        print(f"Error creating slam_match_details: {e}")
        con.execute("DROP TABLE IF EXISTS slam_pbp_raw;")
        con.execute("DROP TABLE IF EXISTS slam_matches_raw;")
        return

    # Update points_fact using the robust join
    print("Updating rally_len in points.points_fact using robust join...")
    try:
        update_sql = """
            UPDATE points.points_fact pf
            SET rally_len = TRY_CAST(smd."Rally" AS INTEGER) -- Use TRY_CAST for rally too
            FROM slam_match_details smd
            WHERE
                -- Match Year
                pf.season = smd.slam_year
                -- Match Players (order insensitive)
                AND (
                    (lower(pf.winner_name) = lower(smd.slam_p1_name) AND lower(pf.loser_name) = lower(smd.slam_p2_name)) OR
                    (lower(pf.winner_name) = lower(smd.slam_p2_name) AND lower(pf.loser_name) = lower(smd.slam_p1_name))
                )
                -- Match Tournament (using mapping)
                AND (
                    (smd.slam_name_extracted = 'ausopen' AND lower(pf.tourney_name) LIKE '%australian open%') OR
                    (smd.slam_name_extracted = 'frenchopen' AND (lower(pf.tourney_name) LIKE '%french open%' OR lower(pf.tourney_name) LIKE '%roland garros%')) OR
                    (smd.slam_name_extracted = 'wimbledon' AND lower(pf.tourney_name) LIKE '%wimbledon%') OR
                    (smd.slam_name_extracted = 'usopen' AND lower(pf.tourney_name) LIKE '%us open%')
                )
                -- Match Point Number
                AND smd.slam_point_number_int IS NOT NULL
                AND pf.point_idx = (smd.slam_point_number_int - 1)
                -- Only update if rally_len is currently NULL to avoid redundant updates?
                -- AND pf.rally_len IS NULL; -- Optional: maybe remove if re-runs are needed
                ;
        """
        result = con.execute(update_sql)
        updated_rows = result.fetchone()[0]
        print(f"✔ Updated rally_len for {updated_rows} points using robust join.")
        if updated_rows == 0:
             print("Warning: No rows in points.points_fact were updated. Check join conditions and data.")

    except Exception as e:
        print(f"Error updating rally_len with robust join: {e}")
    finally:
        # Clean up temporary tables
        con.execute("DROP TABLE IF EXISTS slam_pbp_raw;")
        con.execute("DROP TABLE IF EXISTS slam_matches_raw;")
        con.execute("DROP TABLE IF EXISTS slam_match_details;") 


# --- New Slam Point Loading --- 

def print_distinct_match_ids(con, table_or_view_name, context_msg):
    try:
        total_distinct = con.execute(f"SELECT COUNT(DISTINCT match_id) FROM {table_or_view_name}").fetchone()[0]
        print(f"DEBUG ({context_msg}): Found {total_distinct} distinct match_ids in {table_or_view_name}.")

        # Specifically for _temp_slam_joined, check canonical vs generated
        if table_or_view_name == '_temp_slam_joined':
            distinct_canonical = con.execute(f"SELECT COUNT(DISTINCT canonical_match_id) FROM {table_or_view_name} WHERE canonical_match_id IS NOT NULL").fetchone()[0]
            generated_id_check_sql = f"SELECT COUNT(DISTINCT (slam_year || '-' || slam_name_extracted || '-' || slam_match_id)) FROM {table_or_view_name} WHERE canonical_match_id IS NULL"
            distinct_generated = con.execute(generated_id_check_sql).fetchone()[0]
            print(f"DEBUG ({context_msg}):   - Canonical IDs: {distinct_canonical}")
            print(f"DEBUG ({context_msg}):   - Generated IDs: {distinct_generated}")
            if total_distinct != (distinct_canonical + distinct_generated):
                 # This check might fail if a generated ID happens to look like a canonical one, but it's a good sanity check
                 print(f"DEBUG WARNING ({context_msg}): Sum of canonical ({distinct_canonical}) + generated ({distinct_generated}) != total distinct ({total_distinct})? Investigate ID generation/overlap.")

    except Exception as e:
        print(f"DEBUG ERROR ({context_msg}): Could not query distinct match_ids from {table_or_view_name}: {e}")

def load_slam_points(
    con: duckdb.DuckDBPyConnection,
    slam_pbp_raw_dir: Path = C.SLAM_PBP_RAW_DIR,
    force_reload: bool = False,
) -> None:
    """Loads Slam PBP data into points.points_slam, attempting to join to canonical IDs."""
    print("Loading Slam PBP data into points.points_slam...")
    table_name = "points.points_slam"
    con.execute("CREATE SCHEMA IF NOT EXISTS points;")

    if force_reload:
        con.execute(f"DROP TABLE IF EXISTS {table_name};")

    try:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if count > 0 and not force_reload:
            print(f"✔ {table_name} already populated with {count} points. Skipping.")
            return
    except duckdb.CatalogException:
        pass # Table doesn't exist yet, proceed

    slam_points_files = list(slam_pbp_raw_dir.glob("*-points.csv"))
    slam_matches_files = list(slam_pbp_raw_dir.glob("*-matches.csv"))

    if not slam_points_files or not slam_matches_files:
        print(f"Missing Slam PBP points or matches files... Skipping {table_name} load.")
        return

    # Load Slam Points Data (Temporary)
    try:
        slam_points_glob = str(slam_pbp_raw_dir / "*-points.csv")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _temp_slam_pbp_raw AS
            SELECT *,
                   TRY_CAST(regexp_extract(filename, '(\\d{{4}})', 1) AS INTEGER) AS slam_year,
                   regexp_extract(filename, '\\d{{4}}-([a-z]+)-points\\.csv', 1) AS slam_name_extracted
            FROM read_csv_auto('{slam_points_glob}', header=True, filename=true, all_varchar=True);
        """)
    except Exception as e:
        print(f"Error loading Slam PBP points CSVs: {e}")
        return

    # Load Slam Matches Data (Temporary)
    try:
        slam_matches_glob = str(slam_pbp_raw_dir / "*-matches.csv")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _temp_slam_matches_raw AS
            SELECT "match_id" as slam_match_id, "player1" as slam_p1_name, "player2" as slam_p2_name
            FROM read_csv_auto('{slam_matches_glob}', header=True, all_varchar=True);
        """)
    except Exception as e:
        print(f"Error loading Slam Matches CSVs: {e}")
        con.execute("DROP TABLE IF EXISTS _temp_slam_pbp_raw;")
        return

    # Combine raw slam data and LEFT JOIN to matches_all to find canonical IDs
    print("Combining Slam data and attempting LEFT JOIN to matches_all...")
    try:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE _temp_slam_joined AS
            SELECT
                -- Raw Slam Data
                sp.*, -- Select all columns from slam points
                sm.slam_p1_name,
                sm.slam_p2_name,
                sm.slam_match_id,
                -- Canonical Data (if join succeeds)
                m.match_id as canonical_match_id,
                m.winner_id as canonical_winner_id,
                m.loser_id as canonical_loser_id,
                m.winner_name as canonical_winner_name, -- Keep for reference
                m.loser_name as canonical_loser_name,   -- Keep for reference
                m.tourney_name as canonical_tourney_name -- Keep for reference
            FROM _temp_slam_pbp_raw sp
            JOIN _temp_slam_matches_raw sm ON sp.match_id = sm.slam_match_id
            LEFT JOIN matches_all m
                -- Match Year
                ON m.season = sp.slam_year
                -- Match Players (order insensitive)
                AND (
                    (lower(m.winner_name) = lower(sm.slam_p1_name) AND lower(m.loser_name) = lower(sm.slam_p2_name)) OR
                    (lower(m.winner_name) = lower(sm.slam_p2_name) AND lower(m.loser_name) = lower(sm.slam_p1_name))
                    )
                -- Match Tournament (using mapping)
                AND (
                    (sp.slam_name_extracted = 'ausopen' AND lower(m.tourney_name) LIKE '%australian open%') OR
                    (sp.slam_name_extracted = 'frenchopen' AND (lower(m.tourney_name) LIKE '%french open%' OR lower(m.tourney_name) LIKE '%roland garros%')) OR
                    (sp.slam_name_extracted = 'wimbledon' AND lower(m.tourney_name) LIKE '%wimbledon%') OR
                    (sp.slam_name_extracted = 'usopen' AND lower(m.tourney_name) LIKE '%us open%')
                    )
            WHERE sp.slam_year IS NOT NULL;
        """)
        joined_count = con.execute("SELECT COUNT(*) FROM _temp_slam_joined").fetchone()[0]
        linked_count = con.execute("SELECT COUNT(*) FROM _temp_slam_joined WHERE canonical_match_id IS NOT NULL").fetchone()[0]
        print(f"Combined Slam data for {joined_count} points. Found canonical match for {linked_count} points.")
        print_distinct_match_ids(con, '_temp_slam_joined', 'Post Slam Join')

    except Exception as e:
        print(f"Error joining Slam data: {e}")
        con.execute("DROP TABLE IF EXISTS _temp_slam_pbp_raw;")
        con.execute("DROP TABLE IF EXISTS _temp_slam_matches_raw;")
        return

    # Create and populate points.points_slam
    print(f"Creating and populating {table_name}...")
    try:
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT
                -- Use canonical match_id if available, otherwise generate slam-specific one
                COALESCE(canonical_match_id, slam_year || '-' || slam_name_extracted || '-' || slam_match_id) AS match_id,
                (TRY_CAST("PointNumber" AS INTEGER) - 1) AS point_idx,
                TRY_CAST("SetNo" AS INTEGER) AS set_no,
                TRY_CAST("GameNo" AS INTEGER) AS game_no,
                NULL::INTEGER AS point_in_game,
                -- Determine server/returner ID based on canonical IDs if available
                -- This logic assumes P1 in slam files corresponds to winner if linked, P2 to loser
                -- Needs refinement if P1/P2 isn't consistently winner/loser in source OR if Player IDs exist in slam files
                CASE
                    WHEN canonical_winner_id IS NOT NULL AND "PointServer" = '1' THEN canonical_winner_id
                    WHEN canonical_loser_id IS NOT NULL AND "PointServer" = '2' THEN canonical_loser_id
                    ELSE NULL -- Cannot determine ID if not linked
                END AS server_id,
                 CASE
                    WHEN canonical_loser_id IS NOT NULL AND "PointServer" = '1' THEN canonical_loser_id
                    WHEN canonical_winner_id IS NOT NULL AND "PointServer" = '2' THEN canonical_winner_id
                    ELSE NULL
                END AS returner_id,
                ("PointWinner" = "PointServer") AS server_won,
                TRY_CAST(CASE
                    WHEN "PointServer" = '1' THEN "P1Ace"
                    WHEN "PointServer" = '2' THEN "P2Ace"
                    ELSE '0'
                END AS BOOLEAN) AS is_ace,
                TRY_CAST(CASE
                    WHEN "PointServer" = '1' THEN "P1DoubleFault"
                    WHEN "PointServer" = '2' THEN "P2DoubleFault"
                    ELSE '0'
                END AS BOOLEAN) AS is_df,
                NULL::BOOLEAN AS is_tb,
                NULL::TINYINT AS score_server,
                NULL::TINYINT AS score_returner,
                TRY_CAST("Rally" AS INTEGER) AS rally_len,
                -- Use Slam names directly, canonical if available is just for linking
                slam_p1_name as winner_name, -- Placeholder: Revisit if P1 isn't always winner
                slam_p2_name as loser_name,  -- Placeholder: Revisit if P2 isn't always loser
                slam_name_extracted as tourney_name, -- Use slam name directly
                slam_year as season
            FROM _temp_slam_joined
            WHERE TRY_CAST("PointNumber" AS INTEGER) IS NOT NULL;
        """)
        inserted_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✔ Populated {table_name} with {inserted_count} points.")
        print_distinct_match_ids(con, table_name, 'Post points.points_slam Creation')
        # Add primary key or unique constraint? MatchID+PointIDX should be unique
        # con.execute(f"ALTER TABLE {table_name} ADD CONSTRAINT pk_{table_name.split('.')[-1]} PRIMARY KEY (match_id, point_idx);")

    except Exception as e:
        print(f"Error creating/populating {table_name}: {e}")
    finally:
         # Clean up temporary tables
        con.execute("DROP TABLE IF EXISTS _temp_slam_pbp_raw;")
        con.execute("DROP TABLE IF EXISTS _temp_slam_matches_raw;")
        con.execute("DROP TABLE IF EXISTS _temp_slam_joined;")

# --- View Creation --- 
def create_points_view(con: duckdb.DuckDBPyConnection) -> None:
    """Creates the points.points_all view combining ATP and Slam points, adding rally_bin."""
    print("Creating points.points_all view...")
    # Define rally bins
    short_rally_max = 4
    medium_rally_max = 8

    try:
        con.execute(f"""
            CREATE OR REPLACE VIEW points.points_all AS
            WITH CombinedPoints AS (
                SELECT * FROM points.points_atp
                UNION ALL
                SELECT * FROM points.points_slam
            )
            SELECT
                *,
                CASE
                    WHEN rally_len IS NULL THEN NULL
                    WHEN rally_len <= {short_rally_max} THEN 'short'
                    WHEN rally_len <= {medium_rally_max} THEN 'medium'
                    ELSE 'long'
                END AS rally_bin,
                -- Add serve_plus_one feature
                (rally_len <= 2 AND server_won) AS serve_plus_one
            FROM CombinedPoints;
        """)
        print("✔ Created points.points_all view (with rally_bin and serve_plus_one).")
    except Exception as e:
        print(f"Error creating points.points_all view: {e}")