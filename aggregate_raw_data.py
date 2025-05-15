import pandas as pd
import glob
import os

def aggregate_data(folder_path, file_pattern, aggregated_file_name, low_memory=False):
    """
    Aggregates CSV files in a given folder that match a specific pattern.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        file_pattern (str): The glob pattern to match the CSV files (e.g., "atp_matches_*.csv").
        aggregated_file_name (str): The name for the output aggregated CSV file.
        low_memory (bool): Passed to pd.read_csv low_memory option.
    """
    full_pattern = os.path.join(folder_path, file_pattern)
    # Ensure we only pick up the original, non-aggregated CSVs
    source_files = [
        f for f in glob.glob(full_pattern) 
        if not os.path.basename(f).startswith("aggregated_") # General check for aggregated files
    ]

    # Check if the aggregated file already exists and if source files are present
    output_path = os.path.join(folder_path, aggregated_file_name)
    if os.path.exists(output_path) and not source_files:
        print(f"Aggregated file {output_path} already exists and no new source files found. Skipping aggregation for {folder_path}.")
        # Optionally, read and print info about existing aggregated file
        try:
            existing_df = pd.read_csv(output_path, low_memory=low_memory)
            print(f"Existing aggregated file info: {output_path}")
            print(f"Shape: {existing_df.shape}")
            print(f"Columns: {existing_df.columns.tolist()}")
        except Exception as e:
            print(f"Could not read existing aggregated file {output_path}: {e}")
        return

    if not source_files:
        if not os.path.exists(output_path): # only print if aggregated doesn't exist
             print(f"No non-aggregated '{file_pattern}' files found in {folder_path} to create {aggregated_file_name}.")
        return

    print(f"Found the following files to aggregate in {folder_path}:")
    for f in source_files:
        print(f)
    
    data_frames = []
    for f in source_files:
        try:
            df = pd.read_csv(f, low_memory=low_memory)
            data_frames.append(df)
            print(f"Successfully read {f} (shape: {df.shape})")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if data_frames:
        print(f"Successfully read {len(data_frames)} files from {folder_path}. Concatenating...")
        try:
            aggregated_df = pd.concat(data_frames, ignore_index=True)
            aggregated_df.to_csv(output_path, index=False)
            print(f"Successfully aggregated {len(data_frames)} files into {output_path}")
            print(f"Shape of {aggregated_file_name}: {aggregated_df.shape}")
            print(f"Columns: {aggregated_df.columns.tolist()}")
            # print(f"First 5 rows of {aggregated_file_name}:\n{aggregated_df.head()}") # Keep output concise
        except Exception as e:
            print(f"Error during concatenation or writing {aggregated_file_name} to CSV for {folder_path}: {e}")
    else:
        print(f"No dataframes were read from {folder_path}. Aggregation skipped.")

def main():
    # Aggregate files in data/atp_matches/
    print("--- Aggregating data/atp_matches/ ---")
    aggregate_data(
        folder_path="data/atp_matches/",
        file_pattern="atp_matches_*.csv",
        aggregated_file_name="data/atp_matches/aggregated_atp_matches.csv",
        low_memory=False # Matches original script logic for these files
    )

    print("\n--- Aggregating data/atp_point_by_point/ ---")
    aggregate_data(
        folder_path="data/atp_point_by_point/",
        file_pattern="pbp_matches_atp_main_*.csv", # Catches current and archive
        aggregated_file_name="data/atp_point_by_point/aggregated_pbp_matches.csv",
        low_memory=False # Default, can be overridden if needed
    )

    print("\n--- Aggregating data/slam_point_by_point/ (matches) ---")
    aggregate_data(
        folder_path="data/slam_point_by_point/",
        file_pattern="*-matches.csv", # e.g., 2023-usopen-matches.csv
        aggregated_file_name="data/slam_point_by_point/aggregated_slam_matches.csv",
        low_memory=False 
    )

    print("\n--- Aggregating data/slam_point_by_point/ (points) ---")
    aggregate_data(
        folder_path="data/slam_point_by_point/",
        file_pattern="*-points.csv", # e.g., 2023-usopen-points.csv
        aggregated_file_name="data/slam_point_by_point/aggregated_slam_points.csv",
        low_memory=False # Given the PBP nature, expect many columns, some potentially sparse.
    )
    # Future aggregations (e.g., data/slam_point_by_point/) can be added here

if __name__ == "__main__":
    main()
    print("\nAll aggregation tasks complete.") 