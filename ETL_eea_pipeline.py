#!/usr/bin/env python3
"""
ETL Pipeline for EEA Air Quality Data
Aggregates pollution measurements by geographic microareas to reduce station count
while preserving the dataset schema.
"""

import os
import math
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# --- Global Configuration ---
DATA_DIR_PATH = "./input"
EEA_DIR = DATA_DIR_PATH + "/eea_data"
OUTPUT_DIR_PATH = "./output"

# Global Bounding Boxes (lat/lon coordinates)
LOMBARDIA_BBOX = (8.4, 44.6, 11.5, 46.7)
VENETO_BBOX = (10.6, 44.8, 13.1, 46.7)

NUM_MICROAREAS = 20


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
        print(f"  Successfully downloaded EEA data to: {output_file}")
    except Exception as e:
        print(f"  Error downloading EEA data: {e}")
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


def divide_into_microareas(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    num_microareas: int
) -> List[Dict[str, float]]:
    """
    Divide a geographic bounding box into smaller rectangular microareas.
    
    Args:
        min_lon (float): Minimum longitude (west boundary)
        min_lat (float): Minimum latitude (south boundary)
        max_lon (float): Maximum longitude (east boundary)
        max_lat (float): Maximum latitude (north boundary)
        num_microareas (int): Target number of microareas to create
    
    Returns:
        List[Dict[str, float]]: List of dictionaries with microarea boundaries
    """
    # Calculate dimensions of the bounding box
    height = max_lat - min_lat
    width = max_lon - min_lon
    total_area = width * height
    
    # Calculate the size of each microarea
    area_per_microarea = total_area / num_microareas
    step_size = math.sqrt(area_per_microarea)
    
    # Calculate number of rows and columns
    num_rows = math.ceil(height / step_size)
    num_cols = math.ceil(width / step_size)
    
    print(f"Debug: step_size = {step_size:.4f}")
    print(f"Debug: height = {height:.4f}, width = {width:.4f}")
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    
    list_of_microareas = []
    curr_micro_num = 0
    
    # Generate microareas in row-major order
    for row in range(num_rows):
        for col in range(num_cols):
            if curr_micro_num >= num_microareas:
                break
            
            microarea = {
                'min_lat': min_lat + (row * step_size),
                'min_lon': min_lon + (col * step_size),
                'max_lat': min_lat + ((row + 1) * step_size),
                'max_lon': min_lon + ((col + 1) * step_size)
            }
            
            list_of_microareas.append(microarea)
            curr_micro_num += 1
        
        if curr_micro_num >= num_microareas:
            break
    
    print(f"Debug: Total number of microareas created: {curr_micro_num}")
    
    return list_of_microareas


def aggregate_by_microareas(
    df: pd.DataFrame,
    microareas: List[Dict[str, float]],
    region_name: str
) -> pd.DataFrame:
    """
    Aggregate pollution measurements by microareas, preserving the dataset schema.
    
    For each microarea, measurements from all stations within that area are aggregated:
    - Valore (pollution values) are averaged
    - Dates, pollutants, and other attributes are preserved
    - A synthetic "microarea station" ID is created for each microarea
    
    Args:
        df (pd.DataFrame): Input DataFrame with pollution data
        microareas (List[Dict]): List of microareas with min/max lat/lon
        region_name (str): Name of the region for labeling
    
    Returns:
        pd.DataFrame: Aggregated DataFrame with same schema as input
    """
    
    # Configure pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    def find_microarea(lat, lon):
        """Find which microarea a point belongs to."""
        for idx, micro in enumerate(microareas):
            if (micro['min_lat'] <= lat <= micro['max_lat'] and 
                micro['min_lon'] <= lon <= micro['max_lon']):
                return idx
        return None
    
    # Apply microarea assignment
    df_copy = df.copy()
    print(f"\n\n=== AGGREGATE_BY_MICROAREAS DEBUG ({region_name}) ===\n")
    print(f"Initial DataFrame shape: {df_copy.shape}")
    print(f"Initial DataFrame columns: {list(df_copy.columns)}\n")
    print("Initial DataFrame (first 10 rows):")
    print(df_copy.head(10))
    
    df_copy['_microarea_idx'] = df_copy.apply(
        lambda row: find_microarea(row['Latitudine'], row['Longitudine']), axis=1
    )
    print(f"\nAfter microarea assignment:")
    print(f"  Rows with assigned microarea: {df_copy['_microarea_idx'].notna().sum()}")
    print(f"  Rows without microarea: {df_copy['_microarea_idx'].isna().sum()}")
    print("\nDataFrame after microarea indexing (first 10 rows):")
    print(df_copy.head(10))
    
    # Filter out rows that don't belong to any microarea
    df_copy = df_copy[df_copy['_microarea_idx'].notna()]
    print(f"\nAfter filtering unassigned rows:")
    print(f"  DataFrame shape: {df_copy.shape}")
    print("Filtered DataFrame (first 10 rows):")
    print(df_copy.head(10))
    
    # Rename Stazione to NamesOfAggregatedStats
    df_copy.rename(columns={'Stazione': 'NamesOfAggregatedStats'}, inplace=True)
    
    # Aggregate by microarea index, Data (date), and Inquinante (pollutant)
    agg_dict = {
        'Valore': 'mean',
        'Latitudine': 'mean',
        'Longitudine': 'mean',
        'Nazione': 'first',
        'Unità di misura': 'first',
        'Inquinante': 'first',
        'Data': 'first',
        'NamesOfAggregatedStats': lambda x: '|'.join(x.unique())
    }
    
    df_agg = df_copy.groupby(['_microarea_idx', 'Data', 'Inquinante'], as_index=False).agg(agg_dict)
    df_agg.rename(columns={'Valore': 'Valore_medio_microarea'}, inplace=True)
    df_agg.rename(columns={'Latitudine': 'Latitudine_media_microarea'}, inplace=True)
    df_agg.rename(columns={'Longitudine': 'Longitudine_media_microarea'}, inplace=True)
    print(f"\nAfter groupby aggregation:")
    print(f"  DataFrame shape: {df_agg.shape}")
    print(f"  Unique microareas: {df_agg['_microarea_idx'].nunique()}")
    print(f"  Unique dates: {df_agg['Data'].nunique()}")
    print(f"  Unique pollutants: {df_agg['Inquinante'].nunique()}")
    print("Aggregated DataFrame (first 10 rows):")
    print(df_agg.head(10))
    
    # Add microarea boundary columns
    df_agg['min_lat'] = df_agg['_microarea_idx'].apply(lambda x: microareas[int(x)]['min_lat'])
    df_agg['max_lat'] = df_agg['_microarea_idx'].apply(lambda x: microareas[int(x)]['max_lat'])
    df_agg['min_lon'] = df_agg['_microarea_idx'].apply(lambda x: microareas[int(x)]['min_lon'])
    df_agg['max_lon'] = df_agg['_microarea_idx'].apply(lambda x: microareas[int(x)]['max_lon'])
    print(f"\nAfter adding boundary columns:")
    print(f"  Columns: {df_agg.columns.tolist()}")
    print("DataFrame with boundaries (first 10 rows):")
    print(df_agg.head(10))
    
    # Create Region_microarea_id as primary key
    df_agg['Region_microarea_id'] = df_agg['_microarea_idx'].apply(
        lambda x: f"{region_name}_MA_{int(x)}"
    )
    print(f"\nAfter creating Region_microarea_id:")
    print(f"  Sample IDs: {df_agg['Region_microarea_id'].head(3).tolist()}")
    print("DataFrame with Region_microarea_id (first 10 rows):")
    print(df_agg.head(10))
    
    # Drop temporary column
    df_agg.drop(columns=['_microarea_idx'], inplace=True)
    print(f"\nAfter dropping temporary column:")
    print(f"  DataFrame columns: {list(df_agg.columns)}")
    
    # Reorder columns to match original schema (excluding Comune and Microarea_ID)
    columns_order = [
        'Region_microarea_id', 'Data', 'Inquinante', 'Valore_medio_microarea', 'min_lat', 'min_lon',
        'max_lat', 'max_lon', 'Latitudine_media_microarea', 'Longitudine_media_microarea', 'Nazione',
        'Unità di misura', 'NamesOfAggregatedStats'
    ]
    
    # Keep only columns that exist
    cols_to_keep = [col for col in columns_order if col in df_agg.columns]
    df_agg = df_agg[cols_to_keep]
    print(f"\nAfter column reordering:")
    print(f"  Final columns: {list(df_agg.columns)}")
    print("DataFrame after reordering (first 10 rows):")
    print(df_agg.head(10))
    
    # Set Region_microarea_id as index
    df_agg = df_agg.set_index('Region_microarea_id')
    df_agg = df_agg.reset_index(drop=False)
    
    print(f"\nFinal aggregated DataFrame:")
    print(f"  Shape: {df_agg.shape}")
    print(f"  Total aggregated records: {len(df_agg)}")
    print("Final DataFrame (first 10 rows):")
    print(df_agg.head(10))
    print(f"\n=== END DEBUG ({region_name}) ===\n\n")
    
    return df_agg


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
        print(f"  Successfully exported data to: {output_path}")
        print(f"  Total rows exported: {len(df)}")
    except Exception as e:
        print(f"  Error exporting data to CSV: {e}")
        raise


def main():
    """Main ETL pipeline execution."""
    
    print("="*80)
    print("EEA AIR QUALITY DATA - MICROAREA AGGREGATION PIPELINE")
    print("="*80)
    
    # --- Load and preprocess data ---
    print("\n[1/4] Loading and preprocessing data...")
    Path(EEA_DIR).mkdir(parents=True, exist_ok=True)
    
    if len(os.listdir(EEA_DIR)) < 1:
        print("  Downloading EEA data...")
        df_eea = download_eea_data(eea_dir=EEA_DIR)
    else:
        print("  Loading cached EEA data...")
        df_eea = pd.read_csv(f"{EEA_DIR}/eea.csv")
    
    df_eea = preprocessing_eea(df_eea)
    
    # --- Divide regions into microareas ---
    print("\n[2/4] Dividing regions into microareas...")
    print("\n  LOMBARDIA:")
    lombardia_microareas = divide_into_microareas(
        min_lon=LOMBARDIA_BBOX[0],
        min_lat=LOMBARDIA_BBOX[1],
        max_lon=LOMBARDIA_BBOX[2],
        max_lat=LOMBARDIA_BBOX[3],
        num_microareas=NUM_MICROAREAS
    )
    
    print("\n  VENETO:")
    veneto_microareas = divide_into_microareas(
        min_lon=VENETO_BBOX[0],
        min_lat=VENETO_BBOX[1],
        max_lon=VENETO_BBOX[2],
        max_lat=VENETO_BBOX[3],
        num_microareas=NUM_MICROAREAS
    )
    
    # --- Aggregate by microareas ---
    print("\n[3/4] Aggregating data by microareas...")
    df_lombardia_agg = aggregate_by_microareas(df_eea, lombardia_microareas, "Lombardia")
    df_veneto_agg = aggregate_by_microareas(df_eea, veneto_microareas, "Veneto")
    
    # Combine aggregated data
    df_eea_aggregated = pd.concat([df_lombardia_agg, df_veneto_agg], ignore_index=True)
    
    # --- Display comparison ---
    print("\n[4/4] Displaying results...\n")
    print("BEFORE aggregation:")
    print(f"  Total rows: {len(df_eea)}")
    print(f"  Unique stations: {df_eea['Stazione'].nunique()}")
    print(f"  Schema: {list(df_eea.columns)}")
    
    print("\nAFTER aggregation:")
    print(f"  Total rows: {len(df_eea_aggregated)}")
    print(f"  Unique microareas: {df_eea_aggregated['Region_microarea_id'].nunique()}")
    print(f"  Schema: {list(df_eea_aggregated.columns)}")
    print(f"\n  Reduction factor: {len(df_eea) / len(df_eea_aggregated):.2f}x fewer rows")
    
    print("\n" + "="*100)
    print("ORIGINAL DATA (first 5 rows):")
    print("="*100)
    print(df_eea.head())
    
    print("\n" + "="*100)
    print("AGGREGATED DATA BY MICROAREAS (first 5 rows):")
    print("="*100)
    print(df_eea_aggregated.head())
    
    # --- Export to CSV ---
    print("\n[5/5] Exporting aggregated data to CSV...")
    output_csv_path = f"{OUTPUT_DIR_PATH}/eea_data_aggregated.csv"
    export_to_csv(df_eea_aggregated, output_csv_path)
    
    print("\n" + "="*100)
    print("ETL EEA Pipeline completed successfully!")
    print("="*100)
    
    return df_eea, df_eea_aggregated


if __name__ == "__main__":
    df_eea, df_eea_aggregated = main()
