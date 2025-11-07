import pandas as pd
import numpy as np
from typing import Tuple

def load_datasets(appa_path: str, eea_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both APPA and EEA datasets from CSV files.
    
    Args:
        appa_path: Path to the APPA dataset CSV file
        eea_path: Path to the EEA dataset CSV file (filtered by proximity)
    
    Returns:
        Tuple containing (appa_df, eea_df) DataFrames
    """
    print("Loading datasets...")
    appa = pd.read_csv(appa_path)
    eea = pd.read_csv(eea_path)
    
    print(f"APPA dataset shape: {appa.shape}")
    print("APPA dataset head:")
    print(appa.head())
    
    print(f"\nEEA dataset shape: {eea.shape}")
    print("EEA dataset head:")
    print(eea.head())
    
    return appa, eea


def normalize_appa_df_schema(appa: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize APPA weather feature column names to a standardized schema.
    
    Handles special cases:
    - Pioggia: keeps 'Pioggia (mm).1' and renames to 'Pioggia_value'
    - Temperatura: averages 'Temp. aria (°C).1' and 'Temp. aria (°C).2' into 'Temperatura_aria_value'
    
    Args:
        appa: APPA DataFrame with original weather feature column names
    
    Returns:
        APPA DataFrame with normalized column names
    """
    print("\n--- NORMALIZING APPA SCHEMA ---")
    print("Reasoning: APPA data contains both validated and non-validated measurements.")
    print("We keep historically validated data: validated rainfall instead of raw readings,")
    print("and average validated daily max/min temperatures instead of unvalidated readings.\n")
    
    # Handle rainfall: keep validated rainfall data
    # Reasoning: 'Pioggia (mm).1' is the historically validated measurement
    if 'Pioggia (mm).1' in appa.columns:
        print("  - Keeping validated rainfall: 'Pioggia (mm).1'")
        appa = appa.rename(columns={'Pioggia (mm).1': 'Pioggia_value'})
        # Drop non-validated rainfall columns if they exist
        cols_to_drop = [col for col in ['Pioggia (mm)', 'Pioggia (mm).2'] if col in appa.columns]
        if cols_to_drop:
            appa = appa.drop(columns=cols_to_drop)
    
    # Handle temperature: average validated daily max and min
    # Reasoning: 'Temp. aria (°C).1' (max) and 'Temp. aria (°C).2' (min) are validated daily extremes
    # Averaging reduces bias and provides a representative daily temperature
    if 'Temp. aria (°C).1' in appa.columns and 'Temp. aria (°C).2' in appa.columns:
        print("  - Averaging validated max/min daily temperatures for robustness")
        appa['Temperatura_aria_value'] = appa[['Temp. aria (°C).1', 'Temp. aria (°C).2']].mean(axis=1)
        appa = appa.drop(columns=['Temp. aria (°C).1', 'Temp. aria (°C).2'])
    
    # Standardize remaining column names to match global schema (Ettore & Nicolò)
    print("  - Renaming weather features to standardized global schema")
    rename_mapping = {
        'Dir. Vento (°)': 'Direzione_vento_media_value',
        'Pressione atm. (hPa)': 'Pressione_atmosferica_value',
        'Rad.Sol.Tot. (kJ/m2)': 'Radiazione_solare_totale_value',
        'Umid.relat. aria (%)': 'Umid.relativa_aria_value',
        'Vel. Vento (m/s)': 'Veloc._vento_media_value'
    }
    appa = appa.rename(columns=rename_mapping)
    
    print(f"APPA schema normalized. New columns: {appa.columns.tolist()}\n")
    
    return appa


def preprocess_datasets(appa: pd.DataFrame, eea: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert date columns to datetime format for both datasets.
    
    Args:
        appa: APPA DataFrame (with columns: Stazione, Data, Inquinante, etc.)
        eea: EEA DataFrame (with columns: Stazione, Data, Inquinante, etc. - filtered by proximity)
    
    Returns:
        Tuple containing preprocessed (appa_df, eea_df) DataFrames
    """
    print("\n--- PREPROCESSING DATASETS ---")
    
    # Convert 'Data' column to datetime for both datasets
    appa['Data'] = pd.to_datetime(appa['Data'])
    eea['Data'] = pd.to_datetime(eea['Data'])
    
    print(f"APPA preprocessed shape: {appa.shape}")
    print("APPA preprocessed head:")
    print(appa.head())
    
    print(f"\nEEA preprocessed shape: {eea.shape}")
    print("EEA preprocessed head:")
    print(eea.head())
    
    # Normalize APPA schema
    appa = normalize_appa_df_schema(appa)
    
    return appa, eea


def pivot_eea_data_by_stations(eea: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot EEA data from long to wide format by individual stations.
    
    Creates columns for each station's attributes (Valore, Latitudine, Longitudine, etc.)
    This allows merging with APPA on the Data column.
    
    Args:
        eea: EEA DataFrame with columns: Stazione, Data, Inquinante, Valore, 
             Latitudine, Longitudine, etc.
    
    Returns:
        EEA DataFrame in wide format pivoted by Stazione (station names)
    """
    print("\n--- PIVOTING EEA DATASET BY INDIVIDUAL STATIONS ---")
    
    # Get all unique stations
    unique_stations = eea['Stazione'].unique()
    print(f"Unique EEA stations to pivot: {len(unique_stations)}")
    print(f"Stations: {unique_stations[:10]}..." if len(unique_stations) > 10 else f"Stations: {unique_stations}")
    
    # Initialize pivot dataframe with unique dates
    eea_pivot = pd.DataFrame()
    eea_pivot['Data'] = eea['Data'].unique()
    eea_pivot = eea_pivot.sort_values('Data').reset_index(drop=True)
    
    print(f"Pivot dataframe initialized with {len(eea_pivot)} unique dates")
    print(f"Date range: {eea_pivot['Data'].min()} to {eea_pivot['Data'].max()}\n")
    
    # Iterate through each station and create wide columns
    for station in unique_stations:
        # Filter data for current station
        station_data = eea[eea['Stazione'] == station].copy()
        
        # Select relevant columns
        station_data = station_data[['Data', 'Valore', 'Latitudine', 'Longitudine', 'Inquinante', 'Unità di misura']]
        
        # Rename columns to include station identifier
        station_data.columns = [
            'Data',
            f'{station}_Valore',
            f'{station}_Latitudine',
            f'{station}_Longitudine',
            f'{station}_Inquinante',
            f'{station}_Unità_misura'
        ]
        
        # Merge with pivot dataframe on Data column
        eea_pivot = eea_pivot.merge(station_data, on='Data', how='left')
        
        print(f"  ✓ Added station: {station}")
    
    print(f"\nEEA pivoted shape: {eea_pivot.shape}")
    print(f"EEA pivoted columns ({len(eea_pivot.columns)}): {eea_pivot.columns.tolist()[:20]}...")
    print("EEA pivoted head:")
    print(eea_pivot.head())
    
    return eea_pivot


def merge_datasets(appa: pd.DataFrame, eea_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Merge APPA dataset with pivoted EEA dataset on the Data column.
    
    Uses a left join to preserve all APPA records.
    
    Args:
        appa: APPA DataFrame
        eea_pivot: Pivoted EEA DataFrame (with individual stations as columns)
    
    Returns:
        Merged DataFrame containing both datasets
    """
    print("\n--- MERGING DATASETS ---")
    
    # Merge on Data column using left join
    result = appa.merge(eea_pivot, on='Data', how='left')
    
    print(f"Final merged dataset shape: {result.shape}")
    print(f"Final merged dataset columns ({len(result.columns)}): {result.columns.tolist()[:30]}...")
    print("Merged dataset head:")
    print(result.head())
    
    return result


def save_and_report(result: pd.DataFrame, output_path: str) -> None:
    """
    Save the merged dataset to CSV and display summary statistics.
    
    Args:
        result: Merged DataFrame to save
        output_path: Path where the CSV file will be saved
    """
    print("\n--- SAVING AND REPORTING ---")
    
    # Save the result
    result.to_csv(output_path, index=False)
    print(f"Merged dataset saved to: {output_path}")
    
    # Display basic statistics
    print(f"\nMerged dataset info:")
    print(f"Total rows: {len(result)}")
    print(f"Total columns: {len(result.columns)}")
    print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display missing values summary
    print(f"\nMissing values summary:")
    missing_summary = result.isnull().sum()
    missing_pct = (missing_summary / len(result) * 100).round(2)
    
    print(f"  Total missing values: {missing_summary.sum()}")
    print(f"  Columns with missing values: {(missing_summary > 0).sum()}")
    print(f"  Average % missing per column: {missing_pct.mean():.2f}%")


def main() -> None:
    """
    Main orchestration function that coordinates the entire merge process.
    
    Calls functions in sequence: load -> preprocess -> pivot -> merge -> save
    
    This version works with the filtered EEA dataset (individual stations by proximity)
    instead of aggregated microareas.
    """
    print("=" * 80)
    print("APPA-EEA DATASET MERGE PIPELINE (By Proximity - Individual Stations)")
    print("=" * 80)
    
    # Define file paths
    appa_path = './output/historical_weather_airPM_trentino.csv'
    eea_path = './output/eea_filtered_by_proximity.csv'  # New filtered EEA dataset
    output_path = './output/merged_appa_eea_by_proximity.csv'
    
    # Execute pipeline
    appa, eea = load_datasets(appa_path, eea_path)
    appa, eea = preprocess_datasets(appa, eea)
    eea_pivot = pivot_eea_data_by_stations(eea)
    result = merge_datasets(appa, eea_pivot)
    save_and_report(result, output_path)
    
    print("\n" + "=" * 80)
    print("APPA-EEA MERGE PIPELINE (By Proximity) COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
