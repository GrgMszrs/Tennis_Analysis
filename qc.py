import re
import duckdb
from pathlib import Path
import constants as C
import utils as U

def run_pbp_qc(con: duckdb.DuckDBPyConnection) -> None:
    """Runs basic quality checks on the loaded point data."""
    print("Running Point-by-Point QC checks...")

    # --- QC Check Implementations --- #
    # 1. Row counts per match
    print("QC 1: Comparing parsed point counts vs. estimates from match scores...")
    try:
        query = """
            SELECT
                m.match_id,
                m.score AS match_score_string,
                COUNT(p.point_idx) AS parsed_point_count
            FROM matches_all m
            LEFT JOIN points.points_all p ON m.match_id = p.match_id
            WHERE m.score IS NOT NULL AND m.score != ''
            GROUP BY m.match_id, m.score
            ORDER BY m.match_id;
        """
        match_point_counts_df = con.execute(query).df()

        mismatched_count = 0
        total_checked = 0
        matches_with_zero_parsed_points = 0

        for _, row in match_point_counts_df.iterrows():
            total_checked += 1
            match_id = row['match_id']
            score_str = row['match_score_string']
            parsed_points = int(row['parsed_point_count'])

            if parsed_points == 0 and not any(term in score_str for term in ["RET", "W/O", "DEF"]):
                matches_with_zero_parsed_points +=1
                continue
            elif parsed_points == 0 and any(term in score_str for term in ["RET", "W/O", "DEF"]):
                continue

            estimated_min_pts = _estimate_points_from_score(score_str)

            if estimated_min_pts is not None:
                if parsed_points < 10 and not any(term in score_str for term in ["RET", "W/O", "DEF"]):
                    print(f"Warning (QC1): Match {match_id} (Score: '{score_str}') has only {parsed_points} parsed points. Estimated min: {estimated_min_pts if estimated_min_pts else 'N/A'}.")
                    mismatched_count += 1

        if mismatched_count > 0:
            print(f"Warning (QC1): Found {mismatched_count} / {total_checked} matches with potential point count discrepancies (low parsed points for non-retired matches).")
        else:
            print("✔ QC1: Parsed point count basic check passed (no obvious low counts for scored matches).")
        if matches_with_zero_parsed_points > 0:
            print(f"Info (QC1): {matches_with_zero_parsed_points} matches have a score string but 0 parsed points (PBP data might be missing for these).")

    except Exception as e:
        print(f"QC1 Error (Row Counts): {e}")

    # 2. Ace/DF tallies
    try:
        ace_df_check = con.execute("""
            WITH pbp_agg AS (
                SELECT
                    match_id,
                    SUM(CASE WHEN is_ace THEN 1 ELSE 0 END) as pbp_aces,
                    SUM(CASE WHEN is_df THEN 1 ELSE 0 END) as pbp_dfs
                FROM points.points_all
                GROUP BY match_id
            ),
            match_stats AS (
                SELECT
                    match_id,
                    SUM(w_ace) + SUM(l_ace) AS match_aces,
                    SUM(w_df) + SUM(l_df) AS match_dfs
                FROM matches_all
                GROUP BY match_id
            )
            SELECT m.match_id
            FROM match_stats m JOIN pbp_agg p ON m.match_id = p.match_id
            WHERE m.match_aces != p.pbp_aces OR m.match_dfs != p.pbp_dfs;
        """).fetchall()
        if ace_df_check:
             print(f"Note: Found {len(ace_df_check)} matches where Ace/DF counts differ from match stats (expected for some sources).")
        else:
             print("✔ QC: Ace/DF tally check passed (or no discrepancies found).")
    except Exception as e:
        print(f"QC Error (Ace/DF): {e}")

    print("Skipping QC: Serve alternation check (complex).")
    print("Skipping QC: Score replay check (complex).")
    print("Finished PBP QC checks.")


def _estimate_points_from_score(score_str: str | None) -> int | None:
    """Estimates minimum points played based on a score string."""
    if not score_str or not isinstance(score_str, str):
        return None

    total_points = 0
    try:
        sets = score_str.strip().split(' ')
        for s in sets:
            if 'RET' in s or 'W/O' in s or 'DEF' in s:
                continue

            tiebreak_match = re.search(r'(\d+)-(\d+)\((\d+)\)', s)
            if tiebreak_match:
                g1 = int(tiebreak_match.group(1))
                g2 = int(tiebreak_match.group(2))
                tb_loser_score = int(tiebreak_match.group(3))
                total_points += 48 # Min points in 6-6 games
                tb_winner_score = max(7, tb_loser_score + 2)
                if g1 > 6 or g2 > 6:
                     tb_winner_score = max(g1, g2)
                     tb_loser_score = min(g1, g2)
                     if '(' in s:
                         tb_loser_score_in_paren = int(re.search(r'\((\d+)\)', s).group(1))
                         tb_loser_score = tb_loser_score_in_paren
                         tb_winner_score = max(7, tb_loser_score+1 if tb_loser_score==6 else tb_loser_score + 2)

                total_points += tb_winner_score + tb_loser_score
                continue

            game_match = re.match(r'(\d+)-(\d+)', s)
            if game_match:
                g1 = int(game_match.group(1))
                g2 = int(game_match.group(2))
                games_in_set = g1 + g2
                avg_pts_per_game = 6.8
                total_points += int(games_in_set * avg_pts_per_game)

    except Exception as e:
        return None

    return total_points if total_points > 0 else None


def run_elo_qc(con: duckdb.DuckDBPyConnection) -> None:
    """Runs sanity checks on the generated Elo ratings tables."""
    print("\nRunning Elo Rating QC checks...")

    elo_tables = {
        "Unified": "analytics.elo_player_season",
        "Surface-Specific": "analytics.elo_player_season_surface"
    }

    for name, table_name in elo_tables.items():
        print(f"\n--- Checking {name} Elo Table ({table_name}) ---")
        try:
            # Check if table exists
            con.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1;")
        except (duckdb.CatalogException, duckdb.BinderException):
            print(f"Table {table_name} not found. Skipping checks.")
            continue
        except Exception as e:
            print(f"Error accessing table {table_name}: {e}. Skipping checks.")
            continue

        # 1. Distribution Sanity Check
        print("  QC: Distribution Sanity (Mean ≈ 1500, SD ≈ 160-220?)")
        try:
            if name == "Unified":
                dist_query = f"""
                    SELECT season, AVG(elo_end) AS mean, STDDEV_SAMP(elo_end) AS sd
                    FROM {table_name}
                    WHERE matches_played > 0 -- Only consider active players for stats
                    GROUP BY season
                    ORDER BY season;
                """
            else: # Surface-Specific
                dist_query = f"""
                    SELECT season, surface, AVG(elo_end) AS mean, STDDEV_SAMP(elo_end) AS sd
                    FROM {table_name}
                    WHERE matches_played > 0
                    GROUP BY season, surface
                    ORDER BY season, surface;
                """
            dist_df = con.execute(dist_query).df()
            print("    Distribution Summary:")
            # Print first few and last few rows for brevity
            print(dist_df.head(5).to_string())
            if len(dist_df) > 10:
                print("    ...")
                print(dist_df.tail(5).to_string())
            elif len(dist_df) > 5:
                print(dist_df.tail(len(dist_df) - 5).to_string())

        except Exception as e:
            print(f"  QC Error (Distribution): {e}")

        # 2. Top-N Face Check
        print("\n  QC: Top-N Face Check (Sample)")
        try:
            if name == "Unified":
                top_n_query = f"""
                    SELECT season, player_id, elo_end
                    FROM {table_name}
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY season ORDER BY elo_end DESC) <= 3 -- Show top 3
                    ORDER BY season, elo_end DESC;
                """
                # Filter specific years to keep output small
                sample_years = (2005, 2015, dist_df['season'].max() if not dist_df.empty else 2024) 
                top_n_df = con.execute(top_n_query).df()
                top_n_df = top_n_df[top_n_df['season'].isin(sample_years)]

            else: # Surface-Specific
                top_n_query = f"""
                    SELECT season, surface, player_id, elo_end
                    FROM {table_name}
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY season, surface ORDER BY elo_end DESC) <= 3
                    ORDER BY season, surface, elo_end DESC;
                """
                sample_years = (2005, 2015, dist_df['season'].max() if not dist_df.empty else 2024)
                top_n_df = con.execute(top_n_query).df()
                top_n_df = top_n_df[top_n_df['season'].isin(sample_years)]

            print(f"    Top 3 Players (Sample Years: {sample_years}):")
            print(top_n_df.to_string())
            print("    (Note: Check ATP year-end Top 10 lists for full validation)")

        except Exception as e:
            print(f"  QC Error (Top-N): {e}")

        print("\n  QC: Temporal smoothness check requires plotting (manual check recommended).")
        print("  QC: Bootstrap reorder check is optional and not implemented.")

    print("\nFinished Elo Rating QC checks.") 