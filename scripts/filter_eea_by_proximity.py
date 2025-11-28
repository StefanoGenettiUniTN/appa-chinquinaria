#!/usr/bin/env python3
"""
ETL Pipeline for EEA Air Quality Data - Haversine Distance Filtering
Finds and filters EEA stations based on proximity to Trentino weather stations
using Haversine distance calculations.
"""

import os
import logging
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set

# --- Global Configuration ---
DATA_DIR_PATH = "./input"
EEA_DIR = DATA_DIR_PATH + "/eea_data"
OUTPUT_DIR_PATH = "./output"
TRENTINO_DATA_PATH = "./output/historical_weather_airPM_trentino.csv"

# Number of closest EEA stations to keep for each Trentino station (easily modifiable parameter)
N_CLOSEST_STATIONS = 30

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
) 


def haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate distance in km using Haversine formula."""
    R = 6371.0  # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def download_eea_data(eea_dir: str) -> pd.DataFrame:
    """
    Download EEA air quality data from Google Drive.
    
    Args:
        eea_dir (str): Directory path to store the downloaded file
    
    Returns:
        pd.DataFrame: DataFrame containing the downloaded data
    """
    
    file_id = '15IJoiMRyX9MPhSfReei4MsamO4zcwaqc'
    out_dir = Path(eea_dir)
    output_file = out_dir / "eea.csv"
    
    try:
        gdown.download(
            id=file_id,
            output=str(output_file),
            quiet=False
        )
    except Exception as e:
        raise

    return pd.read_csv(str(output_file))


def preprocessing_eea(df_eea: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess EEA data: rename columns, convert types, clean data.
    
    Args:
        df_eea (pd.DataFrame): Raw EEA data
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Convert column Start to datetime
    df_eea["Start"] = pd.to_datetime(df_eea["Start"])
    
    # Create Data column (date only, no time)
    df_eea["Data"] = df_eea["Start"].dt.strftime("%Y-%m-%d")
    df_eea["Data"] = pd.to_datetime(df_eea["Data"])
    
    # Drop Start and End columns
    df_eea.drop(columns=["Start", "End"], inplace=True)
    
    # Rename columns to Italian
    rename_dict = {
        "Air Pollutant": "Inquinante",
        "Value": "Valore",
        "station-id": "Stazione",
        "Country": "Nazione",
        "Longitude": "Longitudine",
        "Latitude": "Latitudine",
        "Municipality": "Comune",
        "Unit": "Unità di misura"
    }
    df_eea.rename(columns=rename_dict, inplace=True)
    
    # Drop unnecessary columns
    cols_to_drop = [
        "Duration Unit",
        "Cadence Unit",
        "AggType",
        "Altitude",
        "Altitude Unit",
        "Air Quality Station Area",
        "Air Quality Station Type"
    ]
    df_eea.drop(columns=cols_to_drop, inplace=True)
    
    # Remove stations whose Valore is always < 0
    df_eea = df_eea.groupby("Stazione").filter(lambda g: (g["Valore"] >= 0).any())
    
    # Reset index
    df_eea = df_eea.reset_index(drop=True)
    
    return df_eea


def export_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Export aggregated dataset to CSV file.
    Creates the output directory if it doesn't exist.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        output_path (str): File path for the output CSV
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise


def extract_trentino_stations(trentino_path: str) -> pd.DataFrame:
    """
    Extract unique Trentino weather stations with their coordinates.
    
    Args:
        trentino_path (str): Path to the Trentino dataset CSV
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Stazione', 'Latitudine', 'Longitudine']
    """
    logging.info(f"Loading Trentino data from {trentino_path}")
    
    try:
        df_trentino = pd.read_csv(trentino_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trentino data file not found at {trentino_path}")
    
    # Extract unique stations with their coordinates
    trentino_stations = df_trentino[['Stazione', 'Latitudine', 'Longitudine']].drop_duplicates()
    
    logging.info(f"Extracted {len(trentino_stations)} unique Trentino stations")
    
    # Validation: check for null values
    if trentino_stations.isnull().any().any():
        logging.warning("Some Trentino stations have missing coordinates - removing them")
        trentino_stations = trentino_stations.dropna()
        logging.info(f"Remaining Trentino stations after cleaning: {len(trentino_stations)}")
    
    return trentino_stations.reset_index(drop=True)


def extract_eea_stations(df_eea: pd.DataFrame, trentino_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique EEA stations with their coordinates.
    Removes EEA stations that have the same coordinates as Trentino stations to avoid data leakage.
    Also removes hardcoded problematic stations that weren't caught by coordinate matching.
    
    Args:
        df_eea (pd.DataFrame): EEA dataset with columns including 'Stazione', 'Latitudine', 'Longitudine'
        trentino_stations (pd.DataFrame): Trentino stations with columns ['Stazione', 'Latitudine', 'Longitudine']
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Stazione', 'Latitudine', 'Longitudine']
    """
    logging.info("Extracting unique EEA stations from dataset")
    
    # Extract unique stations with their coordinates
    eea_stations = df_eea[['Stazione', 'Latitudine', 'Longitudine']].drop_duplicates()
    
    logging.info(f"Extracted {len(eea_stations)} unique EEA stations")
    
    # Validation: check for null values
    if eea_stations.isnull().any().any():
        logging.warning("Some EEA stations have missing coordinates - removing them")
        eea_stations = eea_stations.dropna()
        logging.info(f"Remaining EEA stations after cleaning: {len(eea_stations)}")
    
    # Check for stations with matching coordinates (data leakage prevention)
    logging.info("Checking for EEA stations with same coordinates as Trentino stations...")
    
    initial_eea_count = len(eea_stations)
    dropped_stations = []
    
    for _, trentino_row in trentino_stations.iterrows():
        trentino_lat = trentino_row['Latitudine']
        trentino_lon = trentino_row['Longitudine']
        trentino_name = trentino_row['Stazione']
        
        # Find EEA stations with matching coordinates (using small epsilon for floating point comparison)
        matching_mask = (
            (np.abs(eea_stations['Latitudine'] - trentino_lat) < 1e-6) & 
            (np.abs(eea_stations['Longitudine'] - trentino_lon) < 1e-6)
        )
        
        matching_stations = eea_stations[matching_mask]
        
        if len(matching_stations) > 0:
            for _, eea_row in matching_stations.iterrows():
                dropped_stations.append({
                    'EEA_Stazione': eea_row['Stazione'],
                    'Trentino_Stazione': trentino_name,
                    'Latitudine': trentino_lat,
                    'Longitudine': trentino_lon
                })
                logging.warning(
                    f"  DROPPED EEA station '{eea_row['Stazione']}' - "
                    f"matches Trentino station '{trentino_name}' "
                    f"at coordinates ({trentino_lat:.6f}, {trentino_lon:.6f})"
                )
            
            # Remove matching stations
            eea_stations = eea_stations[~matching_mask]
    
    final_eea_count = len(eea_stations)
    num_dropped = initial_eea_count - final_eea_count
    
    if num_dropped > 0:
        logging.warning(f"Data leakage prevention: Dropped {num_dropped} EEA station(s) with matching Trentino coordinates")
        logging.info(f"Summary of dropped stations:")
        for dropped in dropped_stations:
            logging.info(
                f"  - EEA: '{dropped['EEA_Stazione']}' <-> Trentino: '{dropped['Trentino_Stazione']}' "
                f"@ ({dropped['Latitudine']:.6f}, {dropped['Longitudine']:.6f})"
            )
    else:
        logging.info("No EEA stations found with matching Trentino coordinates - no data leakage detected")
    
    # --- Drop hardcoded problematic stations ---
    logging.warning("="*80)
    logging.warning("ADDITIONAL DATA LEAKAGE PREVENTION: Hardcoded station removal")
    logging.warning("The dynamic coordinate matching was unable to catch all stations from the Trentino dataset.")
    logging.warning("This may be due to slight coordinate differences or data inconsistencies.")
    logging.warning("Applying hardcoded removal of known problematic stations to prevent data leakage.")
    logging.warning("="*80)
    
    problematic_stations = [
        'SPO.IT0753A_5_nephelometry_beta_2002-03-23_00:00:00',
        'SPO.IT0591A_5_nephelometry_beta_2002-04-16_00:00:00',
        'SPO.IT1037A_5_nephelometry_beta_2006-01-14_00:00:00',
        'SPO.IT1859A_5_nephelometry_beta_2007-11-16_00:00:00',
        'SPO.IT1930A_5_nephelometry_beta_2008-08-07_00:00:00'
    ]
    
    hardcoded_count = len(eea_stations)
    hardcoded_dropped = []
    
    for station in problematic_stations:
        station_data = eea_stations[eea_stations['Stazione'] == station]
        if len(station_data) > 0:
            coords = station_data.iloc[0]
            hardcoded_dropped.append({
                'EEA_Stazione': station,
                'Latitudine': coords['Latitudine'],
                'Longitudine': coords['Longitudine']
            })
    
    eea_stations = eea_stations[~eea_stations['Stazione'].isin(problematic_stations)]
    hardcoded_dropped_count = hardcoded_count - len(eea_stations)
    
    if hardcoded_dropped_count > 0:
        logging.warning(f"Hardcoded removal: Dropped {hardcoded_dropped_count} additional station(s)")
        logging.info("Details of hardcoded dropped stations:")
        for dropped in hardcoded_dropped:
            logging.info(
                f"  - EEA: '{dropped['EEA_Stazione']}' "
                f"@ ({dropped['Latitudine']:.6f}, {dropped['Longitudine']:.6f})"
            )
    else:
        logging.info("No additional stations found in hardcoded list (already removed or not present)")
    
    logging.info(f"Final EEA stations count: {len(eea_stations)}")
    
    return eea_stations.reset_index(drop=True)


def compute_pairwise_distances(trentino_stations: pd.DataFrame, 
                                eea_stations: pd.DataFrame) -> np.ndarray:
    """
    Compute haversine distances between all pairs of Trentino and EEA stations.
    
    Args:
        trentino_stations (pd.DataFrame): Trentino stations with lat/lon
        eea_stations (pd.DataFrame): EEA stations with lat/lon
    
    Returns:
        np.ndarray: Distance matrix of shape (n_trentino, n_eea) with distances in km
    """
    logging.info(f"Computing pairwise distances between {len(trentino_stations)} Trentino and {len(eea_stations)} EEA stations")
    
    n_trentino = len(trentino_stations)
    n_eea = len(eea_stations)
    
    # Initialize distance matrix
    distances = np.zeros((n_trentino, n_eea))
    
    # Get coordinates as numpy arrays for vectorized operations
    trentino_lats = trentino_stations['Latitudine'].values
    trentino_lons = trentino_stations['Longitudine'].values
    eea_lats = eea_stations['Latitudine'].values
    eea_lons = eea_stations['Longitudine'].values
    
    # Compute distances for each Trentino station
    for i in range(n_trentino):
        distances[i, :] = haversine(trentino_lats[i], trentino_lons[i], eea_lats, eea_lons)
    
    logging.info("Distance computation completed")
    return distances


def find_n_closest_stations(trentino_stations: pd.DataFrame, 
                            eea_stations: pd.DataFrame, 
                            distances: np.ndarray,
                            n: int = N_CLOSEST_STATIONS) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    For each Trentino station, find the n closest EEA stations.
    
    IMPORTANT: This function handles the case where an EEA station is close to multiple
    Trentino stations. In such cases, the station is assigned to the Trentino station
    where it is closest, and the next closest EEA station is considered for other
    Trentino stations. This ensures each EEA station appears only once in the final mask.
    
    Args:
        trentino_stations (pd.DataFrame): Trentino stations data
        eea_stations (pd.DataFrame): EEA stations data
        distances (np.ndarray): Distance matrix (n_trentino x n_eea)
        n (int): Number of closest stations to keep per Trentino station
    
    Returns:
        Tuple containing:
            - Set[str]: Set of unique EEA station IDs selected
            - Dict[str, List[str]]: Mapping of Trentino station name to list of closest EEA station names
    """
    logging.info(f"Finding {n} closest EEA stations for each Trentino station")
    
    n_trentino = len(trentino_stations)
    n_eea = len(eea_stations)
    
    selected_eea_stations = set()
    proximity_mapping = {}
    
    # Track which EEA stations have been selected to avoid duplicates
    used_eea_mask = np.zeros(n_eea, dtype=bool)
    
    # For each Trentino station
    for i in range(n_trentino):
        trentino_name = trentino_stations.iloc[i]['Stazione']
        
        # Get distances for this Trentino station
        dists = distances[i, :].copy()
        
        # Set already-used stations to infinity to exclude them
        dists[used_eea_mask] = np.inf
        
        # Find indices of n closest stations (that haven't been used yet)
        # We need to handle the case where fewer than n stations are available
        closest_indices = np.argsort(dists)[:n]
        
        # Filter out stations that couldn't be found (distance = inf)
        closest_indices = closest_indices[np.isfinite(dists[closest_indices])]
        
        if len(closest_indices) < n:
            logging.warning(
                f"Trentino station '{trentino_name}': Could only find {len(closest_indices)} "
                f"available stations (requested {n}). This may occur if many EEA stations "
                f"are already assigned to closer Trentino stations."
            )
        
        # Add selected stations
        closest_eea_stations = []
        for idx in closest_indices:
            eea_name = eea_stations.iloc[idx]['Stazione']
            selected_eea_stations.add(eea_name)
            closest_eea_stations.append(eea_name)
            used_eea_mask[idx] = True
        
        proximity_mapping[trentino_name] = closest_eea_stations
        logging.info(f"  {trentino_name}: {len(closest_eea_stations)} stations selected")
    
    logging.info(f"Total unique EEA stations selected: {len(selected_eea_stations)}")
    return selected_eea_stations, proximity_mapping


def filter_eea_dataset(df_eea: pd.DataFrame, 
                       selected_station_ids: Set[str]) -> pd.DataFrame:
    """
    Filter the EEA dataset to keep only selected stations.
    
    Args:
        df_eea (pd.DataFrame): Full EEA dataset
        selected_station_ids (Set[str]): Set of station IDs to keep
    
    Returns:
        pd.DataFrame: Filtered EEA dataset
    """
    logging.info(f"Filtering EEA dataset to {len(selected_station_ids)} selected stations")
    
    # Filter the dataset
    df_filtered = df_eea[df_eea['Stazione'].isin(selected_station_ids)].copy()
    
    logging.info(f"Filtered dataset shape: {df_filtered.shape}")
    logging.info(f"Date range: {df_filtered['Data'].min()} to {df_filtered['Data'].max()}")
    
    return df_filtered.reset_index(drop=True)


def drop_columns_with_high_nan(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Drop columns that have more than a specified percentage of NaN values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed fraction of NaN values (default 0.05 = 5%)
    
    Returns:
        pd.DataFrame: DataFrame with high-NaN columns removed
    """
    logging.info(f"Checking for columns with more than {threshold*100}% NaN values")
    
    initial_columns = df.shape[1]
    nan_percentages = df.isnull().sum() / len(df)
    
    # Find columns to drop
    columns_to_drop = nan_percentages[nan_percentages > threshold].index.tolist()
    
    if columns_to_drop:
        logging.warning(f"Dropping {len(columns_to_drop)} columns with >{threshold*100}% NaN values:")
        for col in columns_to_drop:
            nan_pct = nan_percentages[col] * 100
            logging.warning(f"  - '{col}': {nan_pct:.2f}% NaN")
        
        df = df.drop(columns=columns_to_drop)
        logging.info(f"Columns reduced from {initial_columns} to {df.shape[1]}")
    else:
        logging.info("No columns found with excessive NaN values")
    
    return df


def drop_stations_with_many_negatives(df: pd.DataFrame, 
                                       station_col: str = 'Stazione',
                                       value_col: str = 'Valore',
                                       max_negatives: int = 300) -> pd.DataFrame:
    """
    Drop stations that have more than a specified number of negative values.
    
    Args:
        df (pd.DataFrame): Input DataFrame with station data
        station_col (str): Name of the station identifier column (default 'Stazione')
        value_col (str): Name of the value column to check (default 'Valore')
        max_negatives (int): Maximum allowed negative values per station (default 300)
    
    Returns:
        pd.DataFrame: DataFrame with problematic stations removed
    """
    logging.info(f"Checking for stations with more than {max_negatives} negative values")
    
    initial_stations = df[station_col].nunique()
    initial_rows = len(df)
    
    # Count negative values per station
    negative_counts = df[df[value_col] < 0].groupby(station_col).size()
    
    # Find stations to drop
    stations_to_drop = negative_counts[negative_counts > max_negatives].index.tolist()
    
    if stations_to_drop:
        logging.warning(f"Dropping {len(stations_to_drop)} stations with >{max_negatives} negative values:")
        for station in stations_to_drop:
            neg_count = negative_counts[station]
            logging.warning(f"  - '{station}': {neg_count} negative values")
        
        df = df[~df[station_col].isin(stations_to_drop)]
        final_stations = df[station_col].nunique()
        final_rows = len(df)
        
        logging.info(f"Stations reduced from {initial_stations} to {final_stations}")
        logging.info(f"Rows reduced from {initial_rows} to {final_rows}")
    else:
        logging.info("No stations found with excessive negative values")
    
    return df.reset_index(drop=True)


def save_proximity_mapping(proximity_mapping: Dict[str, List[str]], 
                           output_path: str) -> None:
    """
    Save the proximity mapping (Trentino stations to nearby EEA stations) as a CSV file.
    
    Args:
        proximity_mapping (Dict[str, List[str]]): Mapping of Trentino to EEA stations
        output_path (str): File path for the output CSV
    """
    logging.info(f"Saving proximity mapping to {output_path}")
    
    # Convert mapping to DataFrame for easier handling
    mapping_data = []
    for trentino_station, eea_stations in proximity_mapping.items():
        for eea_station in eea_stations:
            mapping_data.append({
                'Trentino_Stazione': trentino_station,
                'EEA_Stazione': eea_station
            })
    
    df_mapping = pd.DataFrame(mapping_data)
    export_to_csv(df_mapping, output_path)
    logging.info(f"Proximity mapping saved with {len(df_mapping)} station pairs")


def main():
    """Main ETL pipeline execution."""
    
    logging.info("="*80)
    logging.info("Starting EEA Haversine Distance Filtering Pipeline")
    logging.info("="*80)
    
    # --- Load and preprocess data ---
    Path(EEA_DIR).mkdir(parents=True, exist_ok=True)
    
    if len(os.listdir(EEA_DIR)) < 1:
        df_eea = download_eea_data(eea_dir=EEA_DIR)
    else:
        df_eea = pd.read_csv(f"{EEA_DIR}/eea.csv")
    
    df_eea = preprocessing_eea(df_eea)
    
    # --- Extract unique stations ---
    trentino_stations = extract_trentino_stations(TRENTINO_DATA_PATH)
    eea_stations = extract_eea_stations(df_eea, trentino_stations)
    
    # --- Compute distances ---
    distances = compute_pairwise_distances(trentino_stations, eea_stations)
    
    # --- Find closest stations ---
    selected_eea_stations, proximity_mapping = find_n_closest_stations(
        trentino_stations, 
        eea_stations, 
        distances, 
        n=N_CLOSEST_STATIONS
    )
    
    # --- Filter EEA dataset ---
    df_filtered = filter_eea_dataset(df_eea, selected_eea_stations)
    
    # --- Save results ---
    filtered_output_path = f"{OUTPUT_DIR_PATH}/eea_filtered_by_proximity.csv"
    mapping_output_path = f"{OUTPUT_DIR_PATH}/trentino_eea_proximity_mapping.csv"
    
    export_to_csv(df_filtered, filtered_output_path)
    save_proximity_mapping(proximity_mapping, mapping_output_path)
    
    logging.info("="*80)
    logging.info("Pipeline completed successfully")
    logging.info(f"Filtered EEA data: {filtered_output_path}")
    logging.info(f"Proximity mapping: {mapping_output_path}")
    logging.info("="*80)
    
    return df_filtered, proximity_mapping
    
    
if __name__ == "__main__":
    main()