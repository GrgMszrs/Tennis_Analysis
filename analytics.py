import duckdb
import constants as C

def compute_yearly_summary(con: duckdb.DuckDBPyConnection) -> None:
    """Aggregate raw KPIs into `analytics.yearly_summary`."""
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    con.execute("CREATE OR REPLACE TABLE analytics.yearly_summary AS " + C.SUMMARY_SQL)
    print("✔ Loaded analytics.yearly_summary")


def add_z_scores(con: duckdb.DuckDBPyConnection) -> None:
    """Calculates and adds within‑season z‑scores for key metrics."""
    print("Adding within‑season z‑scores...")
    con.execute(C.Z_SCORE_SQL)
    print("✔ Added within‑season z‑scores")


def compute_rally_stats(
    con: duckdb.DuckDBPyConnection,
    short_rally_max: int = 4,
    medium_rally_max: int = 8,
    force_reload: bool = False,
) -> None:
    """Computes rally length statistics per match and stores them."""
    print("Computing rally statistics per match...")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")

    if force_reload:
        con.execute("DROP TABLE IF EXISTS analytics.match_rally_stats;")

    con.execute("""
        CREATE TABLE IF NOT EXISTS analytics.match_rally_stats (
            match_id VARCHAR PRIMARY KEY,
            avg_rally_len FLOAT,
            pct_rally_short FLOAT,
            pct_rally_medium FLOAT,
            pct_rally_long FLOAT,
            total_rallies_counted INTEGER
        );
    """)

    # Check if table is already populated
    try:
        count = con.execute("SELECT COUNT(*) FROM analytics.match_rally_stats").fetchone()[0]
        if count > 0:
            print(f"✔ analytics.match_rally_stats already populated with {count} matches. Skipping.")
            return
    except duckdb.CatalogException:
        pass # Table doesn't exist yet, proceed

    # <<< ADDED DEBUG PRINT >>>
    try:
        pre_count = con.execute("SELECT COUNT(DISTINCT match_id) FROM points.points_all WHERE rally_len IS NOT NULL").fetchone()[0]
        print(f"DEBUG (Pre-Rally Stats): Found {pre_count} distinct match_ids in points.points_all with non-NULL rally_len.")
    except Exception as e:
        print(f"DEBUG ERROR (Pre-Rally Stats): Could not query points.points_all: {e}")

    # Calculate stats for matches with rally data
    rally_sql = f"""
        INSERT INTO analytics.match_rally_stats
        WITH RallyCounts AS (
            SELECT
                match_id,
                AVG(rally_len) AS avg_rally_len,
                COUNT(*) AS total_rallies,
                SUM(CASE WHEN rally_len BETWEEN 0 AND {short_rally_max} THEN 1 ELSE 0 END) AS count_short,
                SUM(CASE WHEN rally_len BETWEEN {short_rally_max + 1} AND {medium_rally_max} THEN 1 ELSE 0 END) AS count_medium,
                SUM(CASE WHEN rally_len > {medium_rally_max} THEN 1 ELSE 0 END) AS count_long
            FROM points.points_all
            WHERE rally_len IS NOT NULL AND rally_len >= 0 -- Ensure rally_len is valid
            GROUP BY match_id
        )
        SELECT
            match_id,
            avg_rally_len,
            count_short * 100.0 / NULLIF(total_rallies, 0) AS pct_rally_short,
            count_medium * 100.0 / NULLIF(total_rallies, 0) AS pct_rally_medium,
            count_long * 100.0 / NULLIF(total_rallies, 0) AS pct_rally_long,
            total_rallies AS total_rallies_counted
        FROM RallyCounts
        WHERE total_rallies > 0; -- Only insert matches with valid rallies counted
    """
    try:
        con.execute(rally_sql)
        inserted_count = con.execute("SELECT COUNT(*) FROM analytics.match_rally_stats").fetchone()[0]
        print(f"✔ Computed and stored rally stats for {inserted_count} matches in analytics.match_rally_stats.")
    except Exception as e:
        print(f"Error computing rally stats: {e}")


def compute_technical_stats(
    con: duckdb.DuckDBPyConnection,
    force_reload: bool = False,
) -> None:
    """Computes basic serve statistics per player per match."""
    print("Computing technical serve stats per player per match...")
    table_name = "analytics.player_match_serve_stats"
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")

    if force_reload:
        con.execute(f"DROP TABLE IF EXISTS {table_name};")

    # Create table
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            match_id VARCHAR,
            player_id INTEGER,
            total_points_served INTEGER,
            aces INTEGER,
            dfs INTEGER,
            points_won_on_serve INTEGER,
            ace_pct FLOAT,
            df_pct FLOAT,
            serve_pts_won_pct FLOAT,
            PRIMARY KEY (match_id, player_id)
        );
    """)

    # Check if table is already populated
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if count > 0 and not force_reload:
            print(f"✔ {table_name} already populated with stats for {count} player-matches. Skipping.")
            return
    except duckdb.CatalogException:
        pass # Table doesn't exist yet, proceed

    # Calculate stats
    serve_sql = f"""
        INSERT INTO {table_name}
        WITH ServeStatsRaw AS (
            SELECT
                match_id,
                server_id AS player_id,
                COUNT(*) AS total_points_served,
                SUM(CASE WHEN is_ace THEN 1 ELSE 0 END) AS aces,
                SUM(CASE WHEN is_df THEN 1 ELSE 0 END) AS dfs,
                SUM(CASE WHEN server_won THEN 1 ELSE 0 END) AS points_won_on_serve
            FROM points.points_all
            WHERE server_id IS NOT NULL -- Ensure we have a server ID
            GROUP BY match_id, server_id
        )
        SELECT
            match_id,
            player_id,
            total_points_served,
            aces,
            dfs,
            points_won_on_serve,
            aces * 100.0 / NULLIF(total_points_served, 0) AS ace_pct,
            dfs * 100.0 / NULLIF(total_points_served, 0) AS df_pct,
            points_won_on_serve * 100.0 / NULLIF(total_points_served, 0) AS serve_pts_won_pct
        FROM ServeStatsRaw
        WHERE total_points_served > 0;
    """
    try:
        con.execute(serve_sql)
        inserted_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✔ Computed and stored serve stats for {inserted_count} player-matches in {table_name}.")
    except Exception as e:
        print(f"Error computing serve stats: {e}")


def compute_return_stats(
    con: duckdb.DuckDBPyConnection,
    force_reload: bool = False,
) -> None:
    """Computes basic return statistics per player per match."""
    print("Computing technical return stats per player per match...")
    table_name = "analytics.player_match_return_stats"
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")

    if force_reload:
        con.execute(f"DROP TABLE IF EXISTS {table_name};")

    # Create table
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            match_id VARCHAR,
            player_id INTEGER,
            total_points_returned INTEGER,
            return_points_won INTEGER,
            return_pts_won_pct FLOAT,
            PRIMARY KEY (match_id, player_id)
        );
    """)

    # Check if table is already populated
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if count > 0 and not force_reload:
            print(f"✔ {table_name} already populated with stats for {count} player-matches. Skipping.")
            return
    except duckdb.CatalogException:
        pass # Table doesn't exist yet, proceed

    # Calculate stats
    return_sql = f"""
        INSERT INTO {table_name}
        WITH ReturnStatsRaw AS (
            SELECT
                match_id,
                returner_id AS player_id,
                COUNT(*) AS total_points_returned,
                SUM(CASE WHEN NOT server_won THEN 1 ELSE 0 END) AS return_points_won
            FROM points.points_all
            WHERE returner_id IS NOT NULL -- Ensure we have a returner ID
            GROUP BY match_id, returner_id
        )
        SELECT
            match_id,
            player_id,
            total_points_returned,
            return_points_won,
            return_points_won * 100.0 / NULLIF(total_points_returned, 0) AS return_pts_won_pct
        FROM ReturnStatsRaw
        WHERE total_points_returned > 0;
    """
    try:
        con.execute(return_sql)
        inserted_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✔ Computed and stored return stats for {inserted_count} player-matches in {table_name}.")
    except Exception as e:
        print(f"Error computing return stats: {e}") 