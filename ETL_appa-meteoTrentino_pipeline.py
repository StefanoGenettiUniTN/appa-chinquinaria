#!/usr/bin/env python3
"""
ETL Pipeline for APPA Air Quality and MeteoTrentino Weather Data
Merges APPA pollution measurements with MeteoTrentino weather data for Trentino region.
"""

import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

import pandas as pd
import numpy as np
import gdown
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import requests
from lxml import etree


# --- Global Configuration ---
DATA_DIR_PATH = "./input"
APPA_DIR = DATA_DIR_PATH + "/appa_data"
METEO_DIR = DATA_DIR_PATH + "/meteo-trentino-data"
OUTPUT_DIR_PATH = "./output"


class WeatherPlotter:
    """
    Interactive weather data plotter for Jupyter/Colab notebooks.

    Attributes:
    -----------
    cleaned_dfs : dict
        Dictionary of cleaned dataframes {station_name: df}
    variable_map : dict
        Dictionary mapping stations to their available variables
    """

    def __init__(
        self,
        cleaned_dfs: Dict[str, pd.DataFrame],
        variable_map: Dict[str, List[str]]
    ) -> None:
        """
        Initialize the WeatherPlotter.

        Parameters:
        -----------
        cleaned_dfs : dict
            Dictionary of cleaned dataframes {station_name: df}
        variable_map : dict
            Dictionary mapping stations to their available variables
        """
        self.cleaned_dfs: Dict[str, pd.DataFrame] = cleaned_dfs
        self.variable_map: Dict[str, List[str]] = variable_map
        self.stations: List[str] = list(cleaned_dfs.keys())
        
        # Create widgets
        self.station_dropdown: widgets.Dropdown = widgets.Dropdown(
            options=self.stations,
            value=self.stations[0],
            description='Station:',
            style={'description_width': 'initial'}
        )
        self.variable_dropdown: widgets.Dropdown = widgets.Dropdown(
            options=[],
            description='Variable:',
            style={'description_width': 'initial'}
        )
        self.output: widgets.Output = widgets.Output()
        
        # Set up observers
        self.station_dropdown.observe(self.on_station_change, names='value')
        self.variable_dropdown.observe(self.on_variable_change, names='value')

        # Initialize variables for first station
        self.update_variables()

    def update_variables(self) -> None:
        """Update available variables based on selected station."""
        selected_station: str = self.station_dropdown.value
        available_vars: List[str] = self.variable_map.get(selected_station, [])
        self.variable_dropdown.options = available_vars
        if available_vars:
            self.variable_dropdown.value = available_vars[0]

    def on_station_change(self, change: Dict[str, Any]) -> None:
        """Callback when station dropdown changes."""
        self.update_variables()
        self.plot_data()

    def on_variable_change(self, change: Dict[str, Any]) -> None:
        """Callback when variable dropdown changes."""
        self.plot_data()

    def plot_data(self) -> None:
        """Generate plot based on selected station and variable."""
        with self.output:
            self.output.clear_output(wait=True)
            station: str = self.station_dropdown.value
            variable: str = self.variable_dropdown.value
            
            if not variable:
                print("No variables available for this station")
                return
            
            # Get the dataframe
            df: pd.DataFrame = self.cleaned_dfs[station]
            
            # Check if variable exists in dataframe
            if variable not in df.columns:
                print(f"Variable '{variable}' not found in station {station}")
                return
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot the data
            ax.plot(df['Date'], df[variable], linewidth=1.5, color='#2E86AB')
            
            # Styling
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel(variable, fontsize=12, fontweight='bold')
            ax.set_title(
                f'{variable} - Station {station}\n{df["Date"].min().strftime("%d/%m/%Y")} to {df["Date"].max().strftime("%d/%m/%Y")}',
                fontsize=14, fontweight='bold', pad=20
            )
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add statistics
            stats_text: str = (
                f'Mean: {df[variable].mean():.2f}\n'
                f'Min: {df[variable].min():.2f}\n'
                f'Max: {df[variable].max():.2f}'
            )
            ax.text(
                0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            plt.tight_layout()
            plt.show()

    def display(self) -> None:
        """Display the interactive plot widget."""
        # Display widgets
        display(widgets.VBox([
            widgets.HBox([self.station_dropdown, self.variable_dropdown]),
            self.output
        ]))
        # Generate initial plot
        self.plot_data()


def prepare_data_for_plotter(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Prepares the merged dataframe for the WeatherPlotter class.

    Args:
        df: The merged dataframe containing data from all stations.

    Returns:
        A tuple containing:
            - cleaned_dfs: Dictionary of dataframes {station_name: df}
            - variable_map: Dictionary mapping stations to their available variables
    """
    cleaned_dfs = {}
    variable_map = {}

    for station_id, station_df in df.groupby('Station_ID'):
        # Ensure 'Date' column is datetime
        station_df['Date'] = pd.to_datetime(station_df['Date'])
        # Drop the Station_ID column as it's now the key
        station_df = station_df.drop(columns=['Station_ID'])
        cleaned_dfs[station_id] = station_df

        # Get variable columns, excluding 'Date'
        variable_cols = [col for col in station_df.columns if col != 'Date']
        variable_map[station_id] = variable_cols

    return cleaned_dfs, variable_map


def meteo_data_visualization(meteo_df):
    """Visualize weather data using interactive plotter."""
    # Prepare the data using the new function
    cleaned_dfs, variable_map = prepare_data_for_plotter(meteo_df)

    # Initialize and display the plotter
    weather_plotter = WeatherPlotter(cleaned_dfs, variable_map)
    weather_plotter.display()


def download_meteo_trentino(out_dir) -> Path:
    """
    Download MeteoTrentino data from Google Drive folder and extract CSVs.

    Steps:
    1. Build output folder in data/meteo-trentino-data
    2. Download zips from Google Drive folder using gdown
    3. Extract them to CSVs
    4. Return the path where CSVs are stored

    Args:
        out_dir: Output directory path

    Returns:
        Path to the folder containing extracted CSV files
    """
    # Google Drive folder ID
    folder_id = "1nUAp3t_XL5kazrE1L5fqAHl56PvV9aN6"

    # Create a temporary directory for downloading
    temp_dir = Path(tempfile.mkdtemp())

    try:
        print(f"  Downloading ZIP files from Google Drive folder {folder_id}...")

        # Download all files from the Google Drive folder
        gdown.download_folder(
            id=folder_id,
            output=str(temp_dir),
            quiet=False,
            use_cookies=False
        )

        # Get all ZIP files from the temporary directory
        zip_files = list(temp_dir.glob("**/*.zip"))
        print(f"  Found {len(zip_files)} ZIP files to extract")

        if not zip_files:
            print("  Warning: No ZIP files found in Google Drive folder")
            return out_dir

        # Extract all ZIP files to output directory
        extracted_csvs = []
        for zip_file in zip_files:
            print(f"  Extracting {zip_file.name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract all CSV files to the output directory
                    for file_info in zip_ref.filelist:
                        if file_info.filename.endswith('.csv'):
                            # Extract with original filename
                            extracted_path = Path(out_dir) / Path(file_info.filename).name
                            with zip_ref.open(file_info) as source:
                                with open(extracted_path, 'wb') as target:
                                    target.write(source.read())
                            extracted_csvs.append(extracted_path)
                            print(f"    Extracted: {extracted_path.name}")
            except Exception as e:
                print(f"    Error extracting {zip_file.name}: {e}")

        print(f"  Successfully extracted {len(extracted_csvs)} CSV files")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    return out_dir


def data_loading(df_name: str, path) -> pd.DataFrame:
    """
    Data loading function to dynamically load raw weather data to pd.DataFrame Object

    Parameters:
    -----------
    df_name: Name/ID of the station
    path: file path

    Returns:
    --------
    df loaded
    """
    file_path = path + '/' + df_name
    
    # Read the header of the file to determine where to trim the columns header
    with open(file_path, 'r', encoding='latin-1') as f:
        for i in range(3):
            line = f.readline()
            if i == 2:
                usecols = len(line.strip().split(','))
    
    # Pandas csv parser, dynamic
    df = pd.read_csv(
        file_path,
        skiprows=2,
        header=0,
        sep=",",
        decimal=".",  # Symbol to assign decimal point
        encoding='latin-1',
        usecols=range(usecols)  # Select the correct columns
    )
    
    # Skipping first row
    df = df.iloc[1:].copy()
    
    # Converting each numeric value to numeric type
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df


def standardize_weather_data(
    df,
    station_name,
    variable_map=None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Standardize weather station data by cleaning columns and parsing dates.

    Parameters:
    -----------
    df : Raw dataframe from weather station
    station_name : ID of the station (e.g., 'T0135')
    variable_map : Dictionary to update with station variables.
                   If None, creates new dict.

    Returns:
    --------
    cleaned_df, updated_variable_map
    """
    if variable_map is None:
        variable_map = {}
    
    # We make a copy to avoid damage
    df_clean = df.copy()
    
    # Drop "Unnamed" columns
    unnamed_columns = [col for col in df_clean.columns if 'Unnamed' in str(col)]
    df_clean = df_clean.drop(columns=unnamed_columns)
    
    # Fix the 'Date' column
    df_clean['Date'] = pd.to_datetime(
        df_clean['Date'],
        format='%H:%M:%S %d/%m/%Y'
    )
    
    # Add Station Identifier
    df_clean['Station_ID'] = station_name
    
    # Update Variable map
    variable_cols = [
        col for col in df_clean.columns
        if col not in ['Date', 'Station_ID']
    ]
    variable_map[station_name] = variable_cols
    
    # Final data
    df_clean = df_clean[['Date'] + ['Station_ID'] + variable_cols]
    return df_clean, variable_map


def preprocessing_meteo_data(meteo_dir):
    """Preprocess MeteoTrentino weather data."""
    # Dict comprehension to build python dict for each dfs
    dfs = {
        df_name.replace('.csv', ''): data_loading(df_name, meteo_dir)
        for df_name in os.listdir(meteo_dir)
        if df_name.endswith('.csv')
    }
    
    # - cleaned_dfs: dict collection of dfs
    # - variable_map: map of the variables for each df
    variable_map = {}
    cleaned_dfs = {}
    
    for df_name, df in dfs.items():
        cleaned_df, variable_map = standardize_weather_data(df, df_name, variable_map)
        cleaned_dfs[df_name] = cleaned_df
    
    return pd.concat(cleaned_dfs.values(), ignore_index=True)


def appa_download(out_dir) -> list:
    """
    Download APPA data CSV from Google Drive.

    Returns:
        List of paths to downloaded CSV files
    """
    # Google Drive file IDs
    file_ids = {
        "data": "1PmK8wVWC-sfWQJ9e6zYb9sdx3IgHYIsO",
        "metadata": "17OkT0e9QNh2AuWrcMt8IgOEd4r9jQaMI"
    }

    output_files = list()

    for id, file_id in file_ids.items():
        # Output file path
        out_dir = Path(out_dir)
        output_file = out_dir / f"appa_{id}.csv"

        print(f"  Downloading APPA CSV from Google Drive...")
        print(f"  Output folder: {out_dir}")

        try:
            # Download the file using gdown
            gdown.download(
                id=file_id,
                output=str(output_file),
                quiet=False
            )

            print(f"  Successfully downloaded APPA data to: {output_file}")
            output_files.append(output_file)

        except Exception as e:
            print(f"  Error downloading APPA data: {e}")
            raise

    return output_files


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


def find_nearest_station(lat: float, lon: float, stations_df: pd.DataFrame) -> str:
    """Find nearest weather station code based on coordinates."""
    distances = haversine(lat, lon, stations_df['latitudine'].astype(float), stations_df['longitudine'].astype(float))
    return stations_df.iloc[distances.argmin()]['codice']


def load_appa_metadata(output_folder: str) -> pd.DataFrame:
    """Load and prepare APPA metadata."""
    metadata_path = f"{output_folder}/appa_metadata.csv"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_metadata = pd.read_csv(metadata_path)
    print(f"✓ Loaded APPA metadata: {len(df_metadata)} records")

    # Station name mapping
    station_mapping = {
        "TRENTO PSC": "Parco S. Chiara",
        "TRENTO VBZ": "Via Bolzano",
        "PIANA ROTALIANA": "Piana Rotaliana",
        "ROVERETO LGP": "Rovereto",
        "BORGO VAL": "Borgo Valsugana",
        "RIVA GAR": "Riva del Garda",
        "AVIO A22": "A22 (Avio)",
        "MONTE GAZA": "Monte Gaza"
    }

    df_metadata.rename(columns={"Nome stazione": "Stazione"}, inplace=True)
    df_metadata["Stazione"] = df_metadata["Stazione"].replace(station_mapping)

    return df_metadata


def fetch_weather_stations(appa_dir) -> pd.DataFrame:
    """Download and load weather station data."""
    # Google Drive file ID
    file_id = '1bTaAPq6Q65mKKb0NtZy30kqUSILeod7t'
    out_dir = Path(appa_dir)
    output_file = out_dir / f"statJson.csv"

    try:
        # Download the file using gdown
        gdown.download(
            id=file_id,
            output=str(output_file),
            quiet=False
        )

        print(f"  Successfully downloaded weather stations data to: {output_file}")

    except Exception as e:
        print(f"  Error downloading weather stations data: {e}")
        raise

    return pd.read_csv(str(output_file))


def process_appa_measurements(df: pd.DataFrame, df_metadata: pd.DataFrame) -> pd.DataFrame:
    """Process and clean APPA measurement data."""
    # Merge with metadata
    df = pd.merge(df, df_metadata, on="Stazione", how="left", indicator=True)

    unmatched = df[df["_merge"] == "left_only"]
    if len(unmatched) > 0:
        print(f"⚠ Warning: {len(unmatched)} unmatched measurements")

    # Filter PM10 only
    df = df[df['Inquinante'] == 'PM10']
    print(f"✓ Filtered to PM10: {len(df)} records")

    # Clean and transform data
    df["Ora"] = df["Ora"] - 1
    df['Valore'] = pd.to_numeric(df['Valore'], errors='coerce').astype(float)
    df["Unità di misura"] = "ug.m-3"

    # Drop unnecessary columns
    drop_cols = ["_merge", "EU - codice europeo", "Località", "Zona",
                 "Tipologia", "IT - codice italiano", "Dati stazione", "Indirizzo"]
    df.drop(columns=drop_cols, inplace=True)

    # Parse coordinates
    df[['Latitudine', 'Longitudine']] = df['Posizione'].str.split(',', expand=True)
    df['Latitudine'] = df['Latitudine'].str.strip().astype(float)
    df['Longitudine'] = df['Longitudine'].str.strip().astype(float)
    df.drop(columns=['Posizione'], inplace=True)

    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly measurements to daily averages."""
    df = df.drop(columns=['Ora'])

    # Build aggregation dictionary
    agg_dict = {col: 'first' for col in df.columns if col not in ['Stazione', 'Data', 'Valore']}
    agg_dict['Valore'] = 'mean'

    df = df.groupby(['Stazione', 'Data']).agg(agg_dict).reset_index()
    df['Data'] = pd.to_datetime(df['Data'])

    print(f"✓ Aggregated to daily: {len(df)} records")

    return df


def preprocessing_appa_data(appa_dir):
    """Main preprocessing pipeline for APPA air quality data."""
    output_folder = appa_dir

    # Load main dataset
    df = pd.read_csv(f"{output_folder}/appa_data.csv")
    print(f"✓ Loaded APPA data: {len(df)} records")

    # Load metadata
    df_metadata = load_appa_metadata(output_folder)

    # Process measurements
    df = process_appa_measurements(df, df_metadata)

    # Aggregate to daily
    df = aggregate_to_daily(df)

    # Add country and region columns
    df['Nazione'] = 'Italy'
    df['Comune'] = 'APPA'

    # Fetch weather stations and find nearest
    weather_stations = fetch_weather_stations(appa_dir)
    df['Station_ID'] = df.apply(
        lambda row: find_nearest_station(
            row['Latitudine'],
            row['Longitudine'],
            weather_stations),
        axis=1
    )

    print(f"✓ Processing complete: {len(df)} final records")

    return df


def merge_datasets(meteo_df: pd.DataFrame, appa_df: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
    """Merge MeteoTrentino and APPA datasets."""
    print("Merging...")

    # 1) Date parsing robusto
    if "Date" in meteo_df.columns and "Data" not in meteo_df.columns:
        meteo_df = meteo_df.rename(columns={"Date": "Data"})
    meteo_df["Data"] = pd.to_datetime(meteo_df["Data"], dayfirst=True, errors="coerce").dt.date
    appa_df["Data"] = pd.to_datetime(appa_df["Data"], dayfirst=True, errors="coerce").dt.date

    # 2) Rimuovi colonne duplicate nel meteo
    meteo_df = meteo_df.loc[:, ~meteo_df.columns.duplicated()]

    # 3) Merge su Station_ID + Data
    final_df = pd.merge(appa_df, meteo_df, on=["Station_ID", "Data"], how="inner")

    # 4) Rinomina per chiarezza
    final_df = final_df.rename(columns={"Station_ID": "StazioneMeteo"})

    # 5) Salvataggio
    out_dir = Path(OUTPUT_DIR_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_data_path = out_dir / "historical_weather_airPM_trentino.csv"
    final_df.to_csv(final_data_path, index=False)
    
    print(f"✓ Merged dataset saved: {len(final_df)} records")
    
    return final_df, final_data_path


def main():
    """Main ETL pipeline execution."""
    print("="*80)
    print("ETL PIPELINE: APPA AIR QUALITY + METEOTRENTINO WEATHER DATA")
    print("="*80)

    # Create directories
    Path(APPA_DIR).mkdir(parents=True, exist_ok=True)
    Path(METEO_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR_PATH).mkdir(parents=True, exist_ok=True)

    # PART 1: Download data from APPA
    print("\n[1/3] Downloading APPA Air Quality data...")
    if len(os.listdir(APPA_DIR)) < 1:
        csv_files = appa_download(APPA_DIR)
        print(f"  Downloaded APPA DATA to: {csv_files[0]}")
        print(f"  Downloaded APPA METADATA to: {csv_files[1]}")
    else:
        print("  APPA data already present!")

    # PART 2: Download data from MeteoTrentino
    print("\n[2/3] Downloading MeteoTrentino Weather data...")
    if len(os.listdir(METEO_DIR)) < 1:
        meteo_trentino_dir = download_meteo_trentino(METEO_DIR)
        print(f"  Downloaded METEO DATA to: {meteo_trentino_dir}")
    else:
        print("  MeteoTrentino data already present!")

    # PART 3: Data preprocessing and merge
    print("\n[3/3] Data preprocessing and merging...")
    meteo_df = preprocessing_meteo_data(METEO_DIR)
    appa_df = preprocessing_appa_data(APPA_DIR)
    
    df, output_file_path = merge_datasets(meteo_df, appa_df)
    
    print(f"\n✓ Final merged dataset saved to: {output_file_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    
    print("\n" + "="*80)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return df, output_file_path


if __name__ == "__main__":
    df, output_path = main()
