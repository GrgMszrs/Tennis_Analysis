from pathlib import Path
from typing import Iterable

import requests
import duckdb
from tqdm import tqdm

import constants as C # Assuming constants are in constants.py
import utils as U     # Assuming utils are in utils.py


def download_data(
    years: Iterable[int] = C.YEARS,
    raw_dir: Path = C.RAW_DIR,
    force: bool = False,
) -> None:
    """Stream‑download CSVs for the given *years* into *raw_dir*."""
    U._ensure_dir(raw_dir)
    for y in tqdm(years, desc="Downloading CSVs"):
        fname = raw_dir / f"atp_matches_{y}.csv"
        if fname.exists() and not force:
            continue
        url = C.DATA_URL.format(year=y)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        fname.write_bytes(resp.content)
    print(
        f"✔ Downloaded {len(list(raw_dir.glob('*.csv')))} files → {U._display_path(raw_dir)}"
    )


def download_pbp_data(
    pbp_filenames: Iterable[str] = C.PBP_FILENAMES,
    pbp_raw_dir: Path = C.PBP_RAW_DIR,
    force: bool = False,
) -> None:
    """Stream‑download point-by-point CSVs into *pbp_raw_dir*."""
    U._ensure_dir(pbp_raw_dir)
    downloaded_count = 0
    for fname_only in tqdm(pbp_filenames, desc="Downloading PBP CSVs"):
        fpath = pbp_raw_dir / fname_only
        if fpath.exists() and not force:
            continue
        url = C.PBP_DATA_URL_TEMPLATE.format(filename=fname_only)
        try:
            resp = requests.get(url, timeout=120) # Larger timeout for potentially bigger files
            resp.raise_for_status()
            fpath.write_bytes(resp.content)
            downloaded_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to download {fname_only}: {e}")

    total_files = len(list(pbp_raw_dir.glob("*.csv")))
    print(
        f"✔ Downloaded {downloaded_count} new PBP files. Total: {total_files} → {U._display_path(pbp_raw_dir)}"
    )


def load_duckdb(
    db_path: Path = C.DB_PATH,
    raw_dir: Path = C.RAW_DIR,
    years: Iterable[int] = C.YEARS,
) -> duckdb.DuckDBPyConnection:  # noqa: D401,E501
    """Return a DuckDB connection and populate *matches_* tables.

    Keeps the **Path → str** fix for `duckdb.connect()`.
    """
    U._ensure_dir(db_path.parent)
    con = duckdb.connect(str(db_path))

    # Skip heavy table creation when *years* is empty (unit tests)
    if not years:
        return con

    first_year = next(iter(years), None) # Get the first year to describe columns
    if first_year is None:
        print("Warning: No years provided to load_duckdb.")
        return con

    all_cols = []
    for y in years:
        csv_path = raw_dir / f"atp_matches_{y}.csv"
        con.execute(
            f"""CREATE OR REPLACE TABLE matches_{y} AS
                SELECT *,
                       ' {csv_path.stem} ' AS source_file, /* Add source file */
                       {y} AS season,
                       /* Generate match_id from stable fields in match-level data */
                       tourney_id || '-' || match_num AS match_id
                FROM read_csv_auto('{csv_path}', header=True);
            """
        )
        # Describe the first created table to get columns for the view
        if y == first_year:
             try:
                 all_cols = con.execute(f"DESCRIBE matches_{y};").df()['column_name'].to_list()
             except Exception as e:
                 print(f"Error describing table matches_{y}: {e}")
                 # Fallback or handle error appropriately
                 return con # Cannot proceed without columns

    if not all_cols:
        print("Error: Could not determine columns for matches_all view.")
        return con

    # Regenerate view to include new columns (season, match_id, source_file)
    select_cols_str = ", ".join([f'"{c}"' for c in all_cols]) # Quote names just in case
    view_sql = " UNION ALL ".join([f"SELECT {select_cols_str} FROM matches_{y}" for y in years])
    con.execute(f"CREATE OR REPLACE VIEW matches_all AS {view_sql};")
    print(f"✔ Loaded {len(list(years))} years into tables/view (matches_*, matches_all)") # Use list() for length
    return con


def download_slam_pbp_data(
    slam_names: Iterable[str] = C.SLAM_NAMES,
    years: Iterable[int] = C.SLAM_YEARS,
    slam_pbp_raw_dir: Path = C.SLAM_PBP_RAW_DIR,
    force: bool = False,
) -> None:
    """Stream‑download Slam point-by-point CSVs into *slam_pbp_raw_dir*."""
    U._ensure_dir(slam_pbp_raw_dir)
    downloaded_count = 0
    for year in tqdm(years, desc="Slam PBP Years"):
        for slam_name in tqdm(slam_names, desc=f"Downloading {year} Slams", leave=False):
            # Construct filename based on the repository structure
            fname_only = f"{year}-{slam_name}-points.csv"
            fpath = slam_pbp_raw_dir / fname_only # Save directly in slam_pbp_raw_dir
            
            if fpath.exists() and not force:
                continue
            
            url = C.SLAM_PBP_DATA_URL_TEMPLATE.format(year=year, slam=slam_name)
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 404:
                    # This is expected for some year/slam combinations (e.g. AO/RG post-2022)
                    # print(f"Info: Slam PBP file not found (404): {url}") 
                    continue 
                resp.raise_for_status()
                fpath.write_bytes(resp.content)
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to download {fname_only} from {url}: {e}")
    
    current_files = list(slam_pbp_raw_dir.glob("*-points.csv")) # Glob for files directly in the dir
    print(
        f"✔ Slam PBP: Downloaded {downloaded_count} new files. Total on disk: {len(current_files)} → {U._display_path(slam_pbp_raw_dir)}"
    ) 

def download_slam_match_data(
    slam_names: Iterable[str] = C.SLAM_NAMES,
    years: Iterable[int] = C.SLAM_YEARS,
    slam_match_raw_dir: Path = C.SLAM_PBP_RAW_DIR, # Store in the same place for simplicity
    force: bool = False,
) -> None:
    """Stream‑download Slam *-matches.csv files."""
    U._ensure_dir(slam_match_raw_dir)
    downloaded_count = 0
    for year in tqdm(years, desc="Slam Match File Years"):
        for slam_name in tqdm(slam_names, desc=f"Downloading {year} Slam Matches", leave=False):
            fname_only = f"{year}-{slam_name}-matches.csv"
            fpath = slam_match_raw_dir / fname_only
            
            if fpath.exists() and not force:
                continue
            
            # Construct URL - Assuming same base path as points files
            # URL is SLAM_PBP_DATA_URL_TEMPLATE but replace -points with -matches
            # Corrected URL construction:
            matches_file_template = C.SLAM_PBP_DATA_URL_TEMPLATE.replace("-points.csv", "-matches.csv")
            url = matches_file_template.format(year=year, slam=slam_name)
            
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 404:
                    continue # Expected for some year/slam combinations
                resp.raise_for_status()
                fpath.write_bytes(resp.content)
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to download {fname_only} from {url}: {e}")
    
    current_files = list(slam_match_raw_dir.glob("*-matches.csv"))
    print(
        f"✔ Slam Matches: Downloaded {downloaded_count} new files. Total on disk: {len(current_files)} → {U._display_path(slam_match_raw_dir)}"
    ) 