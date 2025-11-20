#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to match APPA air quality stations with Meteo Trentino weather stations
and analyze data coverage for the period 2014-2025.

This script:
1. Loads APPA stations with coordinates
2. Loads Meteo Trentino stations from XML
3. Calculates distance matrix between all stations
4. Finds closest weather stations for each APPA station
5. Analyzes data coverage (missing and invalid values) for all weather variables
6. Generates comprehensive coverage reports
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import zipfile
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def load_appa_stations(stations_file: Path) -> pd.DataFrame:
    """
    Load APPA monitoring stations with coordinates.
    
    Args:
        stations_file: Path to APPA stations CSV file
    
    Returns:
        DataFrame with columns: station_code, station_name, latitude, longitude
    """
    df = pd.read_csv(stations_file)
    
    # Parse coordinates from Posizione column
    def parse_coordinates(pos_str):
        """Parse lat,lon from Posizione string."""
        try:
            pos_str = str(pos_str).strip().strip('"').strip("'")
            parts = [p.strip() for p in pos_str.split(',')]
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
        except:
            return None, None
    
    df['latitude'] = df['Posizione'].apply(lambda x: parse_coordinates(x)[0])
    df['longitude'] = df['Posizione'].apply(lambda x: parse_coordinates(x)[1])
    
    # Create station_code
    df['station_code'] = df['IT - codice italiano'].fillna(
        df['Nome stazione'].apply(lambda x: x.replace(' ', '_').upper())
    )
    
    # Select relevant columns
    stations_df = df[['station_code', 'Nome stazione', 'latitude', 'longitude']].copy()
    stations_df.columns = ['station_code', 'station_name', 'latitude', 'longitude']
    
    # Filter out stations without coordinates
    stations_df = stations_df[stations_df['latitude'].notna() & stations_df['longitude'].notna()]
    
    return stations_df.reset_index(drop=True)


def load_meteo_trentino_stations(xml_file: Path) -> pd.DataFrame:
    """
    Load Meteo Trentino weather stations from XML file.
    
    Args:
        xml_file: Path to stations.xml file
    
    Returns:
        DataFrame with columns: code, name, latitude, longitude, elevation, startdate, enddate
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Define namespace
    ns = {'mt': 'http://www.meteotrentino.it/'}
    
    stations = []
    for point in root.findall('.//mt:pointOfMeasureInfo', ns):
        code = point.find('mt:code', ns)
        name = point.find('mt:name', ns)
        lat = point.find('mt:latitude', ns)
        lon = point.find('mt:longitude', ns)
        elev = point.find('mt:elevation', ns)
        startdate = point.find('mt:startdate', ns)
        enddate = point.find('mt:enddate', ns)
        
        # Parse dates
        start_date = None
        end_date = None
        if startdate is not None and startdate.text:
            try:
                start_date = pd.to_datetime(startdate.text)
            except:
                pass
        
        if enddate is not None and enddate.text:
            try:
                end_date = pd.to_datetime(enddate.text)
            except:
                pass
        
        stations.append({
            'code': code.text if code is not None else None,
            'name': name.text if name is not None else None,
            'latitude': float(lat.text) if lat is not None and lat.text else None,
            'longitude': float(lon.text) if lon is not None and lon.text else None,
            'elevation': float(elev.text) if elev is not None and elev.text else None,
            'startdate': start_date,
            'enddate': end_date
        })
    
    stations_df = pd.DataFrame(stations)
    
    # Filter out stations without coordinates
    stations_df = stations_df[
        stations_df['latitude'].notna() & 
        stations_df['longitude'].notna() &
        stations_df['code'].notna()
    ]
    
    return stations_df.reset_index(drop=True)


def calculate_distance_matrix(appa_stations: pd.DataFrame, 
                              meteo_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance matrix between APPA and Meteo Trentino stations.
    
    Args:
        appa_stations: DataFrame with APPA stations
        meteo_stations: DataFrame with Meteo Trentino stations
    
    Returns:
        Distance matrix DataFrame (rows: APPA stations, columns: Meteo stations)
    """
    distances = []
    appa_codes = []
    meteo_codes = []
    
    # Use tqdm for progress
    for _, appa_row in tqdm(appa_stations.iterrows(), 
                           total=len(appa_stations),
                           desc="Calculating distances",
                           unit="station"):
        appa_codes.append(appa_row['station_code'])
        row_distances = []
        for _, meteo_row in meteo_stations.iterrows():
            dist = haversine_distance(
                appa_row['latitude'], appa_row['longitude'],
                meteo_row['latitude'], meteo_row['longitude']
            )
            row_distances.append(dist)
        distances.append(row_distances)
    
    # Create DataFrame
    distance_matrix = pd.DataFrame(
        distances,
        index=appa_codes,
        columns=meteo_stations['code'].values
    )
    
    return distance_matrix


def get_variable_definitions():
    """Get variable definitions for matching."""
    return {
        'temperature': {
            'keywords': ['temperatura', 'temperature'],
            'zip_patterns': ['*Temperatura*.zip', '*temperatura*.zip']
        },
        'rain': {
            'keywords': ['pioggia', 'rain'],
            'zip_patterns': ['*Pioggia*.zip', '*pioggia*.zip']
        },
        'wind_speed': {
            'keywords': ['veloc', 'vento', 'wind'],
            'zip_patterns': ['*Veloc*.zip', '*vento*.zip', '*wind*.zip']
        },
        'wind_direction': {
            'keywords': ['direzione', 'direction'],
            'zip_patterns': ['*Direzione*.zip', '*direzione*.zip', '*direction*.zip']
        },
        'pressure': {
            'keywords': ['pressione', 'pressure'],
            'zip_patterns': ['*Pressione*.zip', '*pressione*.zip', '*pressure*.zip']
        },
        'radiation': {
            'keywords': ['radiazione', 'radiation'],
            'zip_patterns': ['*Radiazione*.zip', '*radiazione*.zip', '*radiation*.zip']
        },
        'humidity': {
            'keywords': ['umid', 'humidity'],
            'zip_patterns': ['*Umid*.zip', '*umid*.zip', '*humidity*.zip']
        }
    }


def load_or_compute_distance_matrix(appa_stations: pd.DataFrame,
                                   meteo_stations: pd.DataFrame,
                                   cache_path: Path,
                                   force_recompute: bool = False) -> pd.DataFrame:
    """
    Load a cached distance matrix if available, otherwise compute and cache it.
    """
    if cache_path.exists() and not force_recompute:
        print(f"  Using cached distance matrix from {cache_path}")
        dm = pd.read_csv(cache_path, index_col=0)
        return dm
    
    print("  Computing distance matrix (no cache found or recompute forced)...")
    dm = calculate_distance_matrix(appa_stations, meteo_stations)
    dm.to_csv(cache_path)
    print(f"  ✓ Saved distance matrix to {cache_path}")
    return dm


def create_merged_csvs_per_variable(var_dirs: Dict[str, Path],
                                    output_dir: Path,
                                    force_recreate: bool = False) -> Dict[str, Path]:
    """
    Create merged CSV files per variable (one CSV with all stations for each variable).
    This significantly speeds up data access for analysis.
    
    Args:
        var_dirs: Dictionary mapping variable names to extraction directories
        output_dir: Directory to save merged CSVs
        force_recreate: If True, recreate even if merged CSV exists
    
    Returns:
        Dictionary mapping variable names to merged CSV paths
    """
    print("\n" + "="*80)
    print("Creating merged CSV files per variable for faster access")
    print("="*80)
    
    merged_dir = output_dir / "merged_data"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    merged_csvs = {}
    
    for var_name, var_dir in var_dirs.items():
        merged_csv = merged_dir / f"{var_name}_all_stations.csv"
        merged_csvs[var_name] = merged_csv
        
        if merged_csv.exists() and not force_recreate:
            print(f"  {var_name}: Using existing merged CSV ({merged_csv.stat().st_size / 1024**2:.1f} MB)")
            continue
        
        print(f"\n  Merging {var_name} CSV files...")
        csv_files = sorted(var_dir.glob("*.csv"))
        
        if len(csv_files) == 0:
            print(f"    No CSV files found in {var_dir}")
            continue
        
        print(f"    Found {len(csv_files)} station files")
        
        # Read and merge all CSVs (sequential for now - parallel would require pickling issues)
        all_data = []
        failed_files = 0
        for csv_file in tqdm(csv_files, desc=f"    Reading {var_name}", unit="file", leave=False):
            try:
                df = load_weather_csv(csv_file)
                if df is not None and len(df) > 0:
                    station_code = csv_file.stem
                    df['station_code'] = station_code
                    all_data.append(df)
                else:
                    failed_files += 1
            except Exception as e:
                failed_files += 1
                # Only warn for first few failures to avoid spam
                if failed_files <= 5:
                    warnings.warn(f"Error reading {csv_file}: {e}")
                continue
        
        if len(all_data) == 0:
            print(f"    ⚠ No valid data found for {var_name} (failed to read {failed_files}/{len(csv_files)} files)")
            # Still create an entry in merged_csvs dict, but mark as empty
            merged_csvs[var_name] = None
            continue
        
        if failed_files > 0:
            print(f"    ⚠ Successfully read {len(all_data)}/{len(csv_files)} files ({failed_files} failed)")
        
        # Combine all dataframes
        print(f"    Combining {len(all_data)} dataframes...")
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by station_code and datetime for better access patterns
        merged_df = merged_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
        
        # Save merged CSV
        print(f"    Saving merged CSV...")
        merged_df.to_csv(merged_csv, index=False)
        print(f"    ✓ Created merged CSV: {len(merged_df):,} rows, {merged_csv.stat().st_size / 1024**2:.1f} MB")
    
    print("="*80 + "\n")
    return merged_csvs


def extract_all_zip_files(temp_rain_dir: Path, 
                          wind_pressure_dir: Path,
                          extract_dir: Optional[Path] = None,
                          force_reextract: bool = False) -> Dict[str, Path]:
    """
    Extract all ZIP files and organize them by variable type.
    
    Creates subdirectories for each variable type and extracts CSV files there.
    
    Args:
        temp_rain_dir: Directory containing temperature and rain ZIP files
        wind_pressure_dir: Directory containing wind, pressure, radiation, humidity ZIP files
        extract_dir: Directory to extract to (default: creates subdirectories in data dirs)
        force_reextract: If True, re-extract even if CSV already exists
    
    Returns:
        Dictionary mapping variable names to their extraction directories
    """
    print("\n" + "="*80)
    print("Extracting ZIP files and organizing by variable")
    print("="*80)
    
    var_definitions = get_variable_definitions()
    
    # Create extraction directories
    if extract_dir is None:
        # Extract to subdirectories in the original directories
        temp_rain_extract = temp_rain_dir / "extracted"
        wind_pressure_extract = wind_pressure_dir / "extracted"
    else:
        temp_rain_extract = extract_dir / "temp_rain"
        wind_pressure_extract = extract_dir / "wind_pressure"
    
    temp_rain_extract.mkdir(parents=True, exist_ok=True)
    wind_pressure_extract.mkdir(parents=True, exist_ok=True)
    
    # Map variables to their directories
    var_dirs = {
        'temperature': temp_rain_extract / 'temperature',
        'rain': temp_rain_extract / 'rain',
        'wind_speed': wind_pressure_extract / 'wind_speed',
        'wind_direction': wind_pressure_extract / 'wind_direction',
        'pressure': wind_pressure_extract / 'pressure',
        'radiation': wind_pressure_extract / 'radiation',
        'humidity': wind_pressure_extract / 'humidity'
    }
    
    # Create subdirectories for each variable
    for var_dir in var_dirs.values():
        var_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract ZIP files for each variable
    all_dirs = {
        'temperature': temp_rain_dir,
        'rain': temp_rain_dir,
        'wind_speed': wind_pressure_dir,
        'wind_direction': wind_pressure_dir,
        'pressure': wind_pressure_dir,
        'radiation': wind_pressure_dir,
        'humidity': wind_pressure_dir
    }
    
    total_extracted = 0
    
    for var_name, var_info in var_definitions.items():
        source_dir = all_dirs[var_name]
        target_dir = var_dirs[var_name]
        
        # Find ZIP files matching this variable
        zip_files = []
        for pattern in var_info['zip_patterns']:
            zip_files.extend(list(source_dir.glob(pattern)))
        
        # Remove duplicates
        zip_files = list(set(zip_files))
        
        print(f"\n  Processing {var_name}:")
        print(f"    Found {len(zip_files)} ZIP files")
        
        extracted_count = 0
        for zip_file in tqdm(zip_files, desc=f"    Extracting {var_name}", unit="file", leave=False):
            # Extract CSV from ZIP
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        csv_name = csv_files[0]
                        # Extract to variable-specific directory with station code preserved
                        # CSV files are named like T0129.csv, so we keep that name
                        target_csv = target_dir / csv_name
                        
                        # Skip if already extracted (unless force_reextract)
                        if not force_reextract and target_csv.exists():
                            continue
                        
                        with zip_ref.open(csv_name) as source:
                            target_csv.write_bytes(source.read())
                        extracted_count += 1
            except Exception as e:
                warnings.warn(f"Error extracting {zip_file}: {e}")
                continue
        
        print(f"    ✓ Extracted {extracted_count} CSV files to {target_dir}")
        total_extracted += extracted_count
    
    print(f"\n  ✓ Total: Extracted {total_extracted} CSV files")
    print("="*80 + "\n")
    
    return var_dirs


def find_variable_file(station_code: str, var_name: str, var_dirs: Dict[str, Path], 
                      merged_csvs: Optional[Dict[str, Path]] = None) -> Optional[Path]:
    """
    Find the CSV file for a specific variable and station.
    If merged CSV exists, returns path to merged CSV (will filter by station_code when loading).
    
    Args:
        station_code: Weather station code
        var_name: Variable name (e.g., 'temperature', 'rain')
        var_dirs: Dictionary mapping variable names to their extraction directories
        merged_csvs: Optional dictionary mapping variable names to merged CSV paths
    
    Returns:
        Path to CSV file, or None if not found
    """
    if var_name not in var_dirs:
        return None
    
    # If merged CSV exists, prefer it (much faster for repeated access)
    if merged_csvs and var_name in merged_csvs:
        merged_csv = merged_csvs[var_name]
        if merged_csv is not None and merged_csv.exists():
            return merged_csv  # Return merged CSV path (will filter by station_code when loading)
    
    # Fallback to individual CSV file
    var_dir = var_dirs[var_name]
    csv_file = var_dir / f"{station_code}.csv"
    
    if csv_file.exists():
        return csv_file
    
    return None


def find_closest_stations_per_variable(distance_matrix: pd.DataFrame,
                                       var_dirs: Dict[str, Path],
                                       start_date: str = '2014-01-01',
                                       end_date: str = '2025-12-31',
                                       prefer_one_to_one: bool = True,
                                       merged_csvs: Optional[Dict[str, Path]] = None,
                                       coverage_cache: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Find weather stations for each APPA station, ensuring no sharing between APPA stations.
    
    Algorithm:
    1. Process APPA stations in order (by distance to nearest weather station)
    2. For each APPA station, find weather stations for each variable:
       - Prefer weather stations NOT already used by other APPA stations
       - Each APPA station can use different weather stations for different variables
       - Only share weather stations if no unused station is available
    3. Prefers stations with high data quality (>80% valid data)
    4. Tracks which weather stations are assigned to which APPA stations
    
    Key constraint: No weather station should be used by multiple APPA stations.
    However, one APPA station can use multiple weather stations (one per variable).
    
    Args:
        distance_matrix: Distance matrix DataFrame
        var_dirs: Dictionary mapping variable names to their extraction directories
        start_date: Start date for quality checking
        end_date: End date for quality checking
        prefer_one_to_one: If True, ensure no weather station is shared between APPA stations
    
    Returns:
        DataFrame with columns: appa_station, variable, meteo_station, distance_km
    """
    print("Finding weather stations for each APPA station...")
    if prefer_one_to_one:
        print("  Strategy: 1-to-1 mapping (no weather station shared between APPA stations)")
        print("  Note: Each APPA station can use different weather stations for different variables")
    else:
        print("  Strategy: Variable-specific matching (weather stations can be shared)")
    
    variables = get_variable_definitions()
    
    def check_variable_available(station_code: str, var_name: str) -> bool:
        """Check if a station has data for a specific variable.
        
        If a coverage cache is provided, use it instead of touching the filesystem.
        """
        if coverage_cache is not None:
            row = coverage_cache[
                (coverage_cache['meteo_station'] == station_code) &
                (coverage_cache['variable'] == var_name)
            ]
            if len(row) == 0:
                return False
            return bool(row['available'].iloc[0])
        
        file_path = find_variable_file(station_code, var_name, var_dirs, merged_csvs)
        if file_path is None:
            return False
        if not file_path.exists():
            return False
        try:
            if file_path.stat().st_size == 0:
                return False
        except Exception:
            return False
        return True
    
    def get_variable_data_quality(station_code: str, var_name: str, var_dirs: Dict[str, Path], 
                                  start_date: str = '2014-01-01', end_date: str = '2025-12-31',
                                  merged_csvs: Optional[Dict[str, Path]] = None) -> Optional[float]:
        """
        Get the valid data percentage for a variable at a station.
        If a coverage cache is provided, read from it instead of loading CSVs.
        """
        if coverage_cache is not None:
            row = coverage_cache[
                (coverage_cache['meteo_station'] == station_code) &
                (coverage_cache['variable'] == var_name)
            ]
            if len(row) == 0:
                return None
            return float(row['valid_percent'].iloc[0])
        
        file_path = find_variable_file(station_code, var_name, var_dirs, merged_csvs)
        if file_path is None or not file_path.exists():
            return None
        
        try:
            # Fallback: compute quality directly from CSV
            var_names = {
                'temperature': 'Temperatura aria',
                'rain': 'Pioggia',
                'wind_speed': 'Veloc. vento media',
                'wind_direction': 'Direzione vento media',
                'pressure': 'Pressione atmosferica',
                'radiation': 'Radiazione solare totale',
                'humidity': 'Umid.relativa aria'
            }
            var_display_name = var_names.get(var_name)
            is_merged = (merged_csvs and var_name in merged_csvs and 
                        merged_csvs[var_name] is not None and 
                        file_path == merged_csvs[var_name])
            df = load_weather_csv(file_path, variable_name=var_display_name, 
                                 station_code=station_code if is_merged else None)
            
            if df is None or len(df) == 0:
                return None
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
            
            if len(df) == 0:
                return None
            
            valid_mask = df['quality'].isin([1]) & df['value'].notna()
            valid_records = valid_mask.sum()
            total_records = len(df)
            
            if total_records == 0:
                return None
            
            return (valid_records / total_records * 100) if total_records > 0 else 0.0
        except Exception:
            return None
    
    results = []
    unmatched_vars = {}  # Track which variables couldn't be matched
    # Track which weather stations are assigned to which APPA stations
    # Key: weather_station_code, Value: appa_station_code (only one APPA station per weather station)
    weather_station_assignments = {}  # weather_station -> appa_station
    
    # Sort APPA stations by distance to nearest weather station (process closest first)
    appa_station_order = []
    for appa_station in distance_matrix.index:
        min_dist = distance_matrix.loc[appa_station].min()
        appa_station_order.append((appa_station, min_dist))
    appa_station_order.sort(key=lambda x: x[1])  # Sort by minimum distance
    
    # For each APPA station (process in order)
    for appa_station, _ in tqdm(appa_station_order, desc="Matching stations", unit="station"):
        # Get distances sorted
        distances = distance_matrix.loc[appa_station].sort_values()
        
        # For each variable, find closest station that has it WITH GOOD DATA QUALITY
        # Prefer stations NOT already assigned to other APPA stations
        for var_name in variables.keys():
            found = False
            checked_stations = []
            best_station = None
            best_distance = None
            best_quality = 0.0
            
            # Iterate through stations sorted by distance (closest first)
            # But prefer stations with better data quality and not already assigned to other APPA stations
            for meteo_station in distances.index:
                checked_stations.append(meteo_station)
                
                # Skip if this weather station is already assigned to a different APPA station
                if prefer_one_to_one and meteo_station in weather_station_assignments:
                    assigned_to = weather_station_assignments[meteo_station]
                    if assigned_to != appa_station:
                        continue  # Skip - already assigned to another APPA station
                
                if check_variable_available(meteo_station, var_name):
                    # Check data quality for this station
                    quality = get_variable_data_quality(meteo_station, var_name, var_dirs, start_date, end_date, merged_csvs)
                    
                    # Debug: log first few quality checks
                    if len(checked_stations) <= 3 and quality is not None:
                        print(f"      Debug: {meteo_station} - {var_name}: quality={quality:.1f}%")
                    
                    if quality is not None and quality > 0:
                        distance = distances.loc[meteo_station]
                        
                        # Prefer unassigned stations (for 1-to-1 mapping)
                        is_unassigned = meteo_station not in weather_station_assignments
                        unassigned_bonus = 30.0 if (prefer_one_to_one and is_unassigned) else 0.0
                        adjusted_quality = quality + unassigned_bonus
                        
                        # If quality is high enough (>=80%), select immediately
                        if quality >= 80.0:
                            results.append({
                                'appa_station': appa_station,
                                'variable': var_name,
                                'meteo_station': meteo_station,
                                'distance_km': distance
                            })
                            # Track assignment (only one APPA station per weather station)
                            if prefer_one_to_one:
                                weather_station_assignments[meteo_station] = appa_station
                            found = True
                            break
                        # Otherwise, track the best one so far (use adjusted quality for comparison)
                        elif quality >= 50.0 and adjusted_quality > best_quality:
                            best_station = meteo_station
                            best_distance = distance
                            best_quality = quality  # Store actual quality, not adjusted
            
            # If we didn't find a high-quality station, use the best one found (if any)
            # But only if it meets minimum quality threshold (>= 50%)
            if not found and best_station is not None and best_quality >= 50.0:
                # Check if we can use this station (not assigned to another APPA station)
                can_use = True
                if prefer_one_to_one and best_station in weather_station_assignments:
                    assigned_to = weather_station_assignments[best_station]
                    if assigned_to != appa_station:
                        can_use = False  # Already assigned to another APPA station
                
                if can_use:
                    results.append({
                        'appa_station': appa_station,
                        'variable': var_name,
                        'meteo_station': best_station,
                        'distance_km': best_distance
                    })
                    # Track assignment (only one APPA station per weather station)
                    if prefer_one_to_one:
                        weather_station_assignments[best_station] = appa_station
                    found = True
            
            if not found:
                # No station found for this variable
                key = f"{appa_station}_{var_name}"
                unmatched_vars[key] = {
                    'appa_station': appa_station,
                    'variable': var_name,
                    'checked_stations': checked_stations[:10]  # First 10 checked
                }
                results.append({
                    'appa_station': appa_station,
                    'variable': var_name,
                    'meteo_station': None,
                    'distance_km': np.nan
                })
    
    result_df = pd.DataFrame(results)
    
    # Print summary of matches
    print(f"\n  Matching summary:")
    print(f"    Quality thresholds: Prefer stations with >80% valid data, minimum >50%")
    
    # Count 1-to-1 mappings (no weather station shared between APPA stations)
    if prefer_one_to_one:
        # Check for shared weather stations
        station_usage = {}
        for _, row in result_df[result_df['meteo_station'].notna()].iterrows():
            station = row['meteo_station']
            appa = row['appa_station']
            if station not in station_usage:
                station_usage[station] = set()
            station_usage[station].add(appa)
        
        shared_stations = {s: apps for s, apps in station_usage.items() if len(apps) > 1}
        unique_assignments = len(station_usage)
        total_appa_stations = len(distance_matrix.index)
        
        if shared_stations:
            print(f"    ⚠️  Shared weather stations: {len(shared_stations)} weather stations shared between APPA stations")
            print(f"       (This violates 1-to-1 constraint - may need to relax quality thresholds)")
            for station, apps in list(shared_stations.items())[:3]:
                print(f"         {station}: used by {len(apps)} APPA stations ({', '.join(map(str, list(apps)[:3]))})")
        else:
            print(f"    ✓ Perfect 1-to-1 mapping: No weather station shared between APPA stations")
            print(f"      {unique_assignments} unique weather stations assigned to {total_appa_stations} APPA stations")
        
        # Count how many variables each APPA station uses from different weather stations
        for appa_station in distance_matrix.index[:3]:  # Show first 3 as examples
            station_matches = result_df[result_df['appa_station'] == appa_station]
            unique_stations = station_matches['meteo_station'].nunique()
            if unique_stations > 1:
                print(f"      Example: APPA {appa_station} uses {unique_stations} different weather stations")
    
    for var_name in variables.keys():
        var_matches = result_df[result_df['variable'] == var_name]
        matched_count = var_matches['meteo_station'].notna().sum()
        total_count = len(var_matches)
        unmatched_count = total_count - matched_count
        print(f"    {var_name}: {matched_count}/{total_count} APPA stations matched", end="")
        if unmatched_count > 0:
            print(f" ({unmatched_count} unmatched)")
        else:
            print()
    
    # Debug: Show some examples of unmatched variables and statistics
    if unmatched_vars:
        print(f"\n  Debug: Examples of unmatched variables (showing first 5):")
        for i, (key, info) in enumerate(list(unmatched_vars.items())[:5]):
            var_dir = var_dirs.get(info['variable'])
            print(f"    {info['appa_station']} - {info['variable']}:")
            print(f"      Checked {len(info['checked_stations'])} stations (first 5: {info['checked_stations'][:5]})")
            if var_dir:
                # Check if directory exists and has files
                if var_dir.exists():
                    files_in_dir = list(var_dir.glob("*.csv"))
                    print(f"      Directory exists: {var_dir} ({len(files_in_dir)} CSV files)")
                    # Check if any of the checked stations have files
                    checked_station_codes = set(info['checked_stations'][:10])
                    found_files = [f for f in files_in_dir if f.stem in checked_station_codes]
                    print(f"      Files found for checked stations: {len(found_files)}")
                    if len(found_files) == 0:
                        # Show which stations DO have this variable
                        available_stations = [f.stem for f in files_in_dir[:5]]
                        print(f"      Example stations WITH this variable: {available_stations}")
                else:
                    print(f"      Directory does NOT exist: {var_dir}")
        
        # Summary: count unmatched by variable
        unmatched_by_var = {}
        for key, info in unmatched_vars.items():
            var = info['variable']
            unmatched_by_var[var] = unmatched_by_var.get(var, 0) + 1
        print(f"\n  Unmatched variables summary:")
        for var, count in sorted(unmatched_by_var.items(), key=lambda x: x[1], reverse=True):
            print(f"    {var}: {count} APPA stations")
    
    print(f"  ✓ Matched variables for {len(distance_matrix.index)} APPA stations")
    
    return result_df


def load_weather_csv_from_zip(zip_file: Path, variable_keywords: List[str]) -> Optional[pd.DataFrame]:
    """
    Load a weather CSV file from a ZIP archive.
    
    Args:
        zip_file: Path to ZIP file
        variable_keywords: Keywords to verify the variable matches
    
    Returns:
        DataFrame with columns: datetime, value, quality, or None if file doesn't exist
    """
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
            
            # Use first CSV file in ZIP
            csv_name = csv_files[0]
            
            # Extract to temporary file
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.csv') as tmp_file:
                tmp_path = tmp_file.name
                with zip_ref.open(csv_name) as source:
                    tmp_file.write(source.read())
            
            try:
                # Read CSV, skipping first 2 header rows
                df = pd.read_csv(
                    tmp_path,
                    skiprows=2,
                    header=0,
                    encoding='latin-1',
                    on_bad_lines='skip'
                )
                
                # Verify this is the right variable by checking header
                with open(tmp_path, 'r', encoding='latin-1') as f:
                    header_lines = [f.readline().strip() for _ in range(3)]
                header_text = ' '.join(header_lines).lower()
                
                if not any(keyword.lower() in header_text for keyword in variable_keywords):
                    os.unlink(tmp_path)
                    return None
                
                # Skip first data row if it contains metadata
                if len(df) > 0:
                    first_val = str(df.iloc[0, 0]).strip()
                    if 'Sites:' in first_val or 'Variables:' in first_val or 'Qualities:' in first_val:
                        df = df.iloc[1:].copy()
                
                # Get column names
                if len(df.columns) < 3:
                    os.unlink(tmp_path)
                    return None
                
                time_col = df.columns[0]
                value_col = df.columns[1]
                quality_col = df.columns[2]
                
                # Parse datetime
                df['datetime'] = pd.to_datetime(df[time_col], format='%H:%M:%S %d/%m/%Y', errors='coerce')
                
                # Parse value
                df['value'] = pd.to_numeric(df[value_col], errors='coerce')
                
                # Parse quality
                df['quality'] = pd.to_numeric(df[quality_col], errors='coerce')
                
                # Select only valid rows
                df = df[df['datetime'].notna()].copy()
                
                if len(df) == 0:
                    os.unlink(tmp_path)
                    return None
                
                # Select relevant columns
                result = df[['datetime', 'value', 'quality']].copy()
                
                os.unlink(tmp_path)
                return result
                
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e
                
    except Exception as e:
        return None


@lru_cache(maxsize=None)
def _load_weather_csv_cached(path_str: str) -> Optional[pd.DataFrame]:
    """
    Cached reader for weather CSV files to avoid repeated disk IO.
    Returns a dataframe with datetime/value/quality columns or None.
    """
    csv_file = Path(path_str)
    if not csv_file.exists():
        return None
    
    try:
        df = pd.read_csv(
            csv_file,
            skiprows=2,
            header=0,
            encoding='latin-1',
            on_bad_lines='skip',
            dtype=str,
            low_memory=False
        )
        
        if len(df) > 0:
            first_val = str(df.iloc[0, 0]).strip()
            if 'Sites:' in first_val or 'Variables:' in first_val or 'Qualities:' in first_val:
                df = df.iloc[1:].copy()
        
        if len(df.columns) < 3:
            return None
        
        time_col = df.columns[0]
        value_col = df.columns[1]
        quality_col = df.columns[2]
        
        df['datetime'] = pd.to_datetime(df[time_col], format='%H:%M:%S %d/%m/%Y', errors='coerce')
        df['value'] = pd.to_numeric(df[value_col], errors='coerce')
        df['quality'] = pd.to_numeric(df[quality_col], errors='coerce')
        df = df[df['datetime'].notna()].copy()
        
        if len(df) == 0:
            return None
        
        return df[['datetime', 'value', 'quality']]
    
    except Exception as e:
        warnings.warn(f"Error loading {csv_file}: {e}")
        return None


def load_weather_csv(csv_file: Path, variable_name: Optional[str] = None, 
                    station_code: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a weather CSV file and parse it correctly.
    Supports both individual station CSVs and merged CSVs (with station_code filtering).
    
    Args:
        csv_file: Path to CSV file (can be individual station CSV or merged CSV)
        variable_name: Optional variable name to filter (for combined CSV files)
        station_code: Optional station code to filter (for merged CSV files)
    
    Returns:
        DataFrame with columns: datetime, value, quality, or None if file doesn't exist
    """
    if not csv_file.exists():
        return None
    
    try:
        # Check if this is a merged CSV (has station_code column)
        # Read first few rows to check structure (use python engine to avoid C engine tokenizing errors)
        sample_df = pd.read_csv(
            csv_file,
            nrows=5,
            encoding='latin-1',
            dtype=str,
            engine='python',
            on_bad_lines='skip'
        )
        
        if 'station_code' in sample_df.columns:
            # This is a merged CSV - filter by station_code if provided
            if station_code:
                # Read full CSV and filter (use chunking for large files)
                chunk_list = []
                for chunk in pd.read_csv(
                    csv_file,
                    encoding='latin-1',
                    dtype={'station_code': str},
                    low_memory=False,
                    chunksize=100000  # Process in chunks
                ):
                    filtered_chunk = chunk[chunk['station_code'] == station_code].copy()
                    if len(filtered_chunk) > 0:
                        chunk_list.append(filtered_chunk)
                
                if len(chunk_list) == 0:
                    return None
                df = pd.concat(chunk_list, ignore_index=True)
            else:
                # Read full merged CSV
                df = pd.read_csv(
                    csv_file,
                    encoding='latin-1',
                    dtype={'station_code': str},
                    low_memory=False
                )
            
            # Columns should already be parsed (datetime, value, quality)
            if 'datetime' not in df.columns:
                return None
            
            # Ensure datetime is parsed
            if df['datetime'].dtype == 'object':
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Ensure value and quality are numeric
            if 'value' in df.columns and df['value'].dtype == 'object':
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            if 'quality' in df.columns and df['quality'].dtype == 'object':
                df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
        else:
            # Individual station CSV - use original logic
            # Try reading with error handling for malformed lines
            try:
                df = pd.read_csv(
                    csv_file,
                    skiprows=2,
                    header=0,
                    encoding='latin-1',
                    on_bad_lines='skip',
                    dtype=str,
                    sep=',',
                    engine='python'  # Use Python engine for better error handling
                )
            except Exception as e:
                # If that fails, try with more lenient settings
                try:
                    df = pd.read_csv(
                        csv_file,
                        skiprows=2,
                        header=0,
                        encoding='latin-1',
                        on_bad_lines='skip',
                        dtype=str,
                        sep=',',
                        engine='python',
                        quoting=1,  # QUOTE_ALL
                        skipinitialspace=True
                    )
                except Exception as e2:
                    warnings.warn(f"Could not parse {csv_file}: {e2}")
                    return None
            
            # Skip first data row if it contains metadata
            if len(df) > 0:
                first_val = str(df.iloc[0, 0]).strip()
                if 'Sites:' in first_val or 'Variables:' in first_val or 'Qualities:' in first_val:
                    df = df.iloc[1:].copy()
            
            # Get column names
            if len(df.columns) < 3:
                return None
            
            time_col = df.columns[0]
            value_col = df.columns[1]
            quality_col = df.columns[2]
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df[time_col], format='%H:%M:%S %d/%m/%Y', errors='coerce')
            
            # Parse value
            df['value'] = pd.to_numeric(df[value_col], errors='coerce')
            
            # Parse quality
            df['quality'] = pd.to_numeric(df[quality_col], errors='coerce')
        
        # Select only valid rows
        df = df[df['datetime'].notna()].copy()
        
        if len(df) == 0:
            return None
        
        # Select relevant columns (exclude station_code if present, we'll add it back if needed)
        result_cols = ['datetime', 'value', 'quality']
        if 'station_code' in df.columns:
            result_cols.append('station_code')
        result = df[result_cols].copy()
        
        return result
        
    except Exception as e:
        warnings.warn(f"Error loading {csv_file}: {e}")
        return None


def analyze_station_coverage(meteo_station_code: str,
                            var_dirs: Dict[str, Path],
                            start_date: str = '2014-01-01',
                            end_date: str = '2025-12-31',
                            merged_csvs: Optional[Dict[str, Path]] = None) -> Dict:
    """
    Analyze data coverage for a single weather station.
    
    Args:
        meteo_station_code: Station code (e.g., 'T0129')
        var_dirs: Dictionary mapping variable names to their extraction directories
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
    
    Returns:
        Dictionary with coverage statistics
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Expected hourly time series
    expected_times = pd.date_range(start=start_dt, end=end_dt, freq='h')
    expected_count = len(expected_times)
    
    # Variable names and display names
    var_names = {
        'temperature': 'Temperatura aria',
        'rain': 'Pioggia',
        'wind_speed': 'Veloc. vento media',
        'wind_direction': 'Direzione vento media',
        'pressure': 'Pressione atmosferica',
        'radiation': 'Radiazione solare totale',
        'humidity': 'Umid.relativa aria'
    }
    
    coverage_stats = {
        'station_code': meteo_station_code,
        'expected_hours': expected_count,
        'variables': {}
    }
    
    for var_name, var_display_name in var_names.items():
        # Find CSV file for this variable
        csv_file = find_variable_file(meteo_station_code, var_name, var_dirs, merged_csvs)
        
        if csv_file is None:
            # Variable not available for this station
            df = None
        else:
            # Verify file exists and is readable
            if not csv_file.exists():
                df = None
            else:
                # Check if this is a merged CSV
                is_merged = (merged_csvs and var_name in merged_csvs and 
                            merged_csvs[var_name] is not None and
                            csv_file == merged_csvs[var_name])
                # Load from CSV file (pass station_code for merged CSV filtering)
                df = load_weather_csv(csv_file, variable_name=var_display_name,
                                     station_code=meteo_station_code if is_merged else None)
                # Debug: if df is None or empty, log it
                if df is None or len(df) == 0:
                    # File exists but couldn't be loaded or is empty
                    pass
        
        if df is None or len(df) == 0:
            coverage_stats['variables'][var_name] = {
                'available': False,
                'total_records': 0,
                'valid_records': 0,
                'missing_records': expected_count,
                'invalid_records': 0,
                'coverage_percent': 0.0,
                'valid_percent': 0.0
            }
            continue
        
        # Filter by date range
        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
        
        # Quality codes:
        # 1 = good data
        # 140, 145 = uncertain/unvalidated (consider as invalid for our purposes)
        # 151 = missing
        # 255 = no data
        
        valid_quality = [1]  # Only quality code 1 is considered valid
        invalid_quality = [140, 145]  # Uncertain/unvalidated
        missing_quality = [151, 255]  # Missing/no data
        
        # Count records
        total_records = len(df)
        
        # Count valid records (quality = 1)
        valid_mask = df['quality'].isin(valid_quality) & df['value'].notna()
        valid_records = valid_mask.sum()
        
        # Count invalid records (quality = 140, 145)
        invalid_mask = df['quality'].isin(invalid_quality)
        invalid_records = invalid_mask.sum()
        
        # Count missing records (quality = 151, 255 or NaN)
        missing_mask = df['quality'].isin(missing_quality) | df['quality'].isna() | df['value'].isna()
        missing_in_data = missing_mask.sum()
        
        # Calculate missing records: expected - valid - invalid
        # Missing = records that should exist but don't have valid data
        missing_records = expected_count - valid_records - invalid_records
        
        # Ensure missing_records is non-negative (in case of duplicates or data outside range)
        missing_records = max(0, missing_records)
        
        # Calculate coverage percentages (cap at 100%)
        coverage_percent = min(100.0, (total_records / expected_count * 100) if expected_count > 0 else 0.0)
        valid_percent = min(100.0, (valid_records / expected_count * 100) if expected_count > 0 else 0.0)
        
        coverage_stats['variables'][var_name] = {
            'available': True,
            'total_records': total_records,
            'valid_records': valid_records,
            'invalid_records': invalid_records,
            'missing_records': missing_records,
            'coverage_percent': coverage_percent,
            'valid_percent': valid_percent
        }
    
    return coverage_stats


def analyze_station_coverage_parallel(args_tuple):
    """Wrapper function for parallel processing."""
    meteo_station_code, var_dirs_dict, merged_csvs_dict, start_date, end_date = args_tuple
    # Convert dict keys/values from strings back to Path objects
    var_dirs = {k: Path(v) for k, v in var_dirs_dict.items()}
    merged_csvs = {k: Path(v) for k, v in merged_csvs_dict.items()} if merged_csvs_dict else None
    return analyze_station_coverage(
        meteo_station_code, var_dirs, start_date, end_date, merged_csvs
    )


def build_or_load_coverage_cache(var_dirs: Dict[str, Path],
                                 meteo_stations: pd.DataFrame,
                                 start_date: str,
                                 end_date: str,
                                 output_dir: Path,
                                 n_jobs: int = -1,
                                 merged_csvs: Optional[Dict[str, Path]] = None,
                                 force_recompute: bool = False) -> pd.DataFrame:
    """
    Build (or load) a cache of coverage data for each measurement for each station.
    This is the heavy step (reads CSVs) but is done once and then reused.
    """
    cache_path = output_dir / "station_variable_coverage.csv"
    availability_path = output_dir / "station_variable_availability.csv"
    
    if cache_path.exists() and not force_recompute:
        print(f"\nUsing cached coverage data from {cache_path}")
        coverage_cache = pd.read_csv(cache_path)
        # Ensure correct dtypes
        coverage_cache['meteo_station'] = coverage_cache['meteo_station'].astype(str)
        coverage_cache['variable'] = coverage_cache['variable'].astype(str)
        # Availability cache (lightweight) – load or recreate from coverage_cache
        if availability_path.exists():
            print(f"Using cached availability data from {availability_path}")
        else:
            availability_df = coverage_cache[['meteo_station', 'variable', 'available']].drop_duplicates()
            availability_df.to_csv(availability_path, index=False)
            print(f"  ✓ Saved availability cache to {availability_path}")
        return coverage_cache
    
    print("\n" + "=" * 80)
    print("Building global coverage cache for all Meteo Trentino stations")
    print("=" * 80)
    print(f"  Period: {start_date} to {end_date}")
    
    station_codes = meteo_stations['code'].astype(str).unique()
    print(f"  Total stations to analyze: {len(station_codes)}")
    
    # Prepare arguments for parallel processing
    var_dirs_str = {k: str(v) for k, v in var_dirs.items()}
    merged_csvs_str = {k: str(v) for k, v in merged_csvs.items()} if merged_csvs else {}
    args_list = [
        (station_code, var_dirs_str, merged_csvs_str, start_date, end_date)
        for station_code in station_codes
    ]
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    results = []
    print(f"  Using parallel processing with {n_jobs} workers")
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_station = {
            executor.submit(analyze_station_coverage_parallel, args): args[0]
            for args in args_list
        }
        
        for future in tqdm(as_completed(future_to_station),
                          total=len(future_to_station),
                          desc="  Computing coverage cache",
                          unit="station"):
            station_code = future_to_station[future]
            try:
                coverage = future.result()
                for var_name, var_stats in coverage['variables'].items():
                    results.append({
                        'meteo_station': station_code,
                        'variable': var_name,
                        'available': var_stats['available'],
                        'expected_hours': coverage['expected_hours'],
                        'total_records': var_stats['total_records'],
                        'valid_records': var_stats['valid_records'],
                        'invalid_records': var_stats['invalid_records'],
                        'missing_records': var_stats['missing_records'],
                        'coverage_percent': var_stats['coverage_percent'],
                        'valid_percent': var_stats['valid_percent']
                    })
            except Exception as e:
                warnings.warn(f"Error computing coverage for station {station_code}: {e}")
    
    coverage_cache = pd.DataFrame(results)
    print(f"\n  ✓ Built coverage cache for {coverage_cache['meteo_station'].nunique()} stations")
    
    # Save coverage cache
    coverage_cache.to_csv(cache_path, index=False)
    print(f"  ✓ Saved coverage cache to {cache_path}")
    
    # Build and save availability cache (lightweight)
    availability_df = coverage_cache[['meteo_station', 'variable', 'available']].drop_duplicates()
    availability_df.to_csv(availability_path, index=False)
    print(f"  ✓ Saved availability cache to {availability_path}")
    
    print("=" * 80 + "\n")
    return coverage_cache


def analyze_candidate_stations(variable_matching_df: pd.DataFrame,
                               var_dirs: Dict[str, Path],
                               start_date: str = '2014-01-01',
                               end_date: str = '2025-12-31',
                               output_file: Optional[Path] = None,
                               n_jobs: int = -1,
                               merged_csvs: Optional[Dict[str, Path]] = None,
                               coverage_cache: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Analyze coverage for weather stations matched to variables.
    
    Args:
        variable_matching_df: DataFrame with variable-specific station matching
        var_dirs: Dictionary mapping variable names to their extraction directories
        start_date: Start date for analysis
        end_date: End date for analysis
        output_file: Optional path to save results CSV
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    
    Returns:
        DataFrame with coverage analysis results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing coverage for matched weather stations")
    print(f"{'='*80}")
    print(f"  Period: {start_date} to {end_date}")
    
    # If a global coverage cache is provided, just slice it to the matched stations/variables.
    if coverage_cache is not None:
        print("  Using cached coverage data (no per-run CSV scanning).")
        # Filter cache to matched meteo stations and variables
        valid_matches = variable_matching_df[variable_matching_df['meteo_station'].notna()].copy()
        key = valid_matches[['meteo_station', 'variable']].drop_duplicates()
        results_df = coverage_cache.merge(
            key,
            how='inner',
            left_on=['meteo_station', 'variable'],
            right_on=['meteo_station', 'variable']
        )
    else:
        # Fallback: compute coverage only for matched stations (legacy behaviour)
        valid_matches = variable_matching_df[variable_matching_df['meteo_station'].notna()].copy()
        unique_stations = valid_matches['meteo_station'].unique()
        
        print(f"  Total unique stations to analyze: {len(unique_stations)}")
        print(f"  Using parallel processing with {n_jobs if n_jobs > 0 else mp.cpu_count()} workers")
        print()
        
        # Prepare arguments for parallel processing (convert Path to str for pickling)
        var_dirs_str = {k: str(v) for k, v in var_dirs.items()}
        merged_csvs_str = {k: str(v) for k, v in merged_csvs.items()} if merged_csvs else {}
        args_list = [
            (station_code, var_dirs_str, merged_csvs_str, start_date, end_date)
            for station_code in unique_stations
        ]
        
        results = []
        
        # Parallel processing
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_station = {
                executor.submit(analyze_station_coverage_parallel, args): args[0]
                for args in args_list
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_station), 
                              total=len(future_to_station),
                              desc="Analyzing stations",
                              unit="station"):
                station_code = future_to_station[future]
                try:
                    coverage = future.result()
                    
                    # Get all variables coverage
                    for var_name, var_stats in coverage['variables'].items():
                        results.append({
                            'meteo_station': station_code,
                            'variable': var_name,
                            'available': var_stats['available'],
                            'total_records': var_stats['total_records'],
                            'valid_records': var_stats['valid_records'],
                            'invalid_records': var_stats['invalid_records'],
                            'missing_records': var_stats['missing_records'],
                            'coverage_percent': var_stats['coverage_percent'],
                            'valid_percent': var_stats['valid_percent']
                        })
                except Exception as e:
                    warnings.warn(f"Error analyzing station {station_code}: {e}")
        
        print(f"\n  ✓ Analyzed {len(unique_stations)} weather stations")
        results_df = pd.DataFrame(results)
    
    # Handle empty results case
    if len(results_df) == 0:
        print("  ⚠ Warning: No coverage data found. This may indicate matching issues.")
        # Create empty dataframe with expected columns
        results_df = pd.DataFrame(columns=[
            'meteo_station', 'variable', 'available', 'expected_hours',
            'total_records', 'valid_records', 'invalid_records',
            'missing_records', 'coverage_percent', 'valid_percent',
            'appa_station', 'distance_km'
        ])
    else:
        # Merge with variable matching info
        if 'meteo_station' in results_df.columns and 'variable' in results_df.columns:
            results_df = results_df.merge(
                variable_matching_df[['appa_station', 'variable', 'meteo_station', 'distance_km']],
                on=['meteo_station', 'variable'],
                how='left'
            )
        else:
            print("  ⚠ Warning: results_df missing expected columns, skipping merge")
    
    # Save if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved detailed results to {output_file}")
    
    return results_df


def generate_summary_report(coverage_df: pd.DataFrame,
                           output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Generate a summary report aggregated by APPA station and variable.
    
    Args:
        coverage_df: DataFrame with coverage analysis results
        output_file: Optional path to save summary CSV
    
    Returns:
        Summary DataFrame
    """
    print("\nGenerating summary report...")

    # Handle completely empty coverage_df (e.g., no stations matched)
    if coverage_df is None or len(coverage_df) == 0:
        print("  ⚠ coverage_df is empty. No summary can be generated.")
        empty_summary = pd.DataFrame(columns=[
            'appa_station', 'variable', 'meteo_station', 'distance_km',
            'available', 'valid_percent', 'coverage_percent',
            'valid_records', 'missing_records', 'invalid_records'
        ])
        if output_file:
            empty_summary.to_csv(output_file, index=False)
            print(f"  ✓ Saved empty summary to {output_file}")
        return empty_summary
    
    # Ensure required columns exist
    required_cols = {'appa_station', 'variable', 'meteo_station', 'distance_km',
                     'available', 'valid_percent', 'coverage_percent',
                     'valid_records', 'missing_records', 'invalid_records'}
    missing = required_cols - set(coverage_df.columns)
    if missing:
        print(f"  ⚠ coverage_df is missing expected columns: {missing}. Returning empty summary.")
        empty_summary = pd.DataFrame(columns=list(required_cols))
        if output_file:
            empty_summary.to_csv(output_file, index=False)
            print(f"  ✓ Saved empty summary to {output_file}")
        return empty_summary
    
    # Aggregate by APPA station and variable (since we now match per variable)
    summary = coverage_df.groupby(['appa_station', 'variable', 'meteo_station', 'distance_km']).agg({
        'available': 'first',  # Should be same for all rows
        'valid_percent': 'first',  # Should be same for all rows
        'coverage_percent': 'first',  # Should be same for all rows
        'valid_records': 'first',  # Should be same for all rows
        'missing_records': 'first',  # Should be same for all rows
        'invalid_records': 'first'  # Should be same for all rows
    }).reset_index()
    
    # Rename columns
    summary.columns = [
        'appa_station', 'variable', 'meteo_station', 'distance_km',
        'available', 'valid_percent', 'coverage_percent',
        'valid_records', 'missing_records', 'invalid_records'
    ]
    
    # Sort by APPA station and variable
    summary = summary.sort_values(['appa_station', 'variable'])
    
    if output_file:
        summary.to_csv(output_file, index=False)
        print(f"  ✓ Saved summary to {output_file}")
    
    return summary


def create_station_map(appa_stations: pd.DataFrame,
                      meteo_stations: pd.DataFrame,
                      variable_matching: pd.DataFrame,
                      output_dir: Path):
    """
    Create a map visualization showing all APPA and Meteo Trentino stations.
    
    Args:
        appa_stations: DataFrame with APPA station info
        meteo_stations: DataFrame with Meteo Trentino station info
        variable_matching: DataFrame with station matching results
        output_dir: Output directory for plots
    """
    print("Creating station map visualization...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot Meteo Trentino stations
    ax.scatter(meteo_stations['longitude'], meteo_stations['latitude'],
              c='green', marker='^', s=100, alpha=0.6, label='Meteo Trentino Stations', zorder=3)
    
    # Add labels for Meteo Trentino stations
    for _, row in meteo_stations.iterrows():
        ax.annotate(row['code'], (row['longitude'], row['latitude']),
                   fontsize=6, alpha=0.7, ha='center', va='bottom')
    
    # Plot APPA stations
    ax.scatter(appa_stations['longitude'], appa_stations['latitude'],
              c='blue', marker='o', s=150, alpha=0.8, label='APPA Stations', zorder=4, edgecolors='black', linewidths=1.5)
    
    # Add labels for APPA stations
    for _, row in appa_stations.iterrows():
        ax.annotate(row['station_code'], (row['longitude'], row['latitude']),
                   fontsize=8, fontweight='bold', ha='center', va='top', color='blue')
    
    # Draw connections for matched stations
    matched_pairs = variable_matching[variable_matching['meteo_station'].notna()].copy()
    # Get unique APPA-meteo pairs
    unique_pairs = matched_pairs[['appa_station', 'meteo_station']].drop_duplicates()
    
    for _, pair in unique_pairs.iterrows():
        appa_code = pair['appa_station']
        meteo_code = pair['meteo_station']
        
        appa_row = appa_stations[appa_stations['station_code'] == appa_code]
        meteo_row = meteo_stations[meteo_stations['code'] == meteo_code]
        
        if len(appa_row) > 0 and len(meteo_row) > 0:
            ax.plot([appa_row.iloc[0]['longitude'], meteo_row.iloc[0]['longitude']],
                   [appa_row.iloc[0]['latitude'], meteo_row.iloc[0]['latitude']],
                   'r--', alpha=0.3, linewidth=0.5, zorder=1)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('APPA and Meteo Trentino Station Locations\n(Red lines show matched pairs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    # Add a simple km scale bar in the bottom-left corner
    try:
        # Approximate conversion: 1 degree longitude ≈ 111 km * cos(latitude)
        # Use mean latitude of all stations for the conversion
        all_lats = pd.concat([appa_stations['latitude'], meteo_stations['latitude']])
        all_lons = pd.concat([appa_stations['longitude'], meteo_stations['longitude']])
        mean_lat = all_lats.mean()
        min_lon = all_lons.min()
        min_lat = all_lats.min()

        km_length = 10  # length of scale bar in km
        km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
        if km_per_deg_lon > 0:
            deg_length = km_length / km_per_deg_lon

            # Place scale bar slightly above the bottom-left corner
            x0 = min_lon + 0.01
            y0 = min_lat + 0.01
            ax.plot([x0, x0 + deg_length], [y0, y0], color='k', linewidth=2, zorder=5)
            ax.text(x0 + deg_length / 2, y0 + 0.01, f'{km_length} km',
                    ha='center', va='bottom', fontsize=9, color='k')
    except Exception:
        # If anything goes wrong, just skip the scale bar
        pass
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / '00_station_map.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved station map to plots/00_station_map.png")
    plt.close()


def plot_weather_station_timeseries(var_dirs: Dict[str, Path],
                                    meteo_stations: pd.DataFrame,
                                    variable_matching: pd.DataFrame,
                                    start_date: str = '2014-01-01',
                                    end_date: str = '2025-12-31',
                                    output_dir: Path = None,
                                    merged_csvs: Optional[Dict[str, Path]] = None):
    """
    Create time series plots for each weather variable at each station.
    Similar to the PM10 plotting scripts.
    
    Args:
        var_dirs: Dictionary mapping variable names to extraction directories
        meteo_stations: DataFrame with Meteo Trentino station info
        variable_matching: DataFrame with station matching results
        start_date: Start date for plots
        end_date: End date for plots
        output_dir: Output directory for plots
    """
    print("\nCreating weather station time series plots...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    var_names = {
        'temperature': 'Temperatura aria',
        'rain': 'Pioggia',
        'wind_speed': 'Veloc. vento media',
        'wind_direction': 'Direzione vento media',
        'pressure': 'Pressione atmosferica',
        'radiation': 'Radiazione solare totale',
        'humidity': 'Umid.relativa aria'
    }
    
    var_display_names = {
        'temperature': 'Temperature (°C)',
        'rain': 'Rain (mm)',
        'wind_speed': 'Wind Speed (m/s)',
        'wind_direction': 'Wind Direction (°)',
        'pressure': 'Pressure (hPa)',
        'radiation': 'Radiation (W/m²)',
        'humidity': 'Humidity (%)'
    }
    
    # Get unique weather stations that are matched
    matched_stations = variable_matching[variable_matching['meteo_station'].notna()]['meteo_station'].unique()
    
    for var_name in var_names.keys():
        print(f"  Plotting {var_name}...")
        
        # Get stations that have this variable
        var_stations = variable_matching[
            (variable_matching['variable'] == var_name) & 
            (variable_matching['meteo_station'].notna())
        ]['meteo_station'].unique()
        
        if len(var_stations) == 0:
            print(f"    No stations found for {var_name}")
            continue
        
        # Create subplot grid
        n_stations = len(var_stations)
        n_cols = 3
        n_rows = (n_stations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_stations > 1 else [axes]
        
        for i, station_code in enumerate(var_stations):
            ax = axes[i]
            
            # Load data
            csv_file = find_variable_file(station_code, var_name, var_dirs, merged_csvs)
            if csv_file is None or not csv_file.exists():
                ax.text(0.5, 0.5, f'No data\nfor {station_code}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{station_code}", fontsize=9)
                continue
            
            # Check if this is a merged CSV
            is_merged = (merged_csvs and var_name in merged_csvs and 
                        merged_csvs[var_name] is not None and
                        csv_file == merged_csvs[var_name])
            df = load_weather_csv(csv_file, variable_name=var_names[var_name],
                                 station_code=station_code if is_merged else None)
            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, f'No data\nfor {station_code}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{station_code}", fontsize=9)
                continue
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
            
            if len(df) == 0:
                ax.text(0.5, 0.5, f'No data in range\nfor {station_code}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{station_code}", fontsize=9)
                continue

            # Separate valid and invalid data (full resolution for statistics)
            valid_data = df[(df['quality'] == 1) & df['value'].notna()]
            invalid_data = df[(df['quality'].isin([140, 145])) & df['value'].notna()]

            # Downsample for plotting only (keep statistics exact)
            # Aim to plot at most ~2000 points per series
            max_points = 2000
            if len(df) > max_points:
                step = max(1, len(df) // max_points)
                df_plot = df.iloc[::step].copy()
                valid_plot = valid_data.iloc[::step].copy() if len(valid_data) > 0 else valid_data
                invalid_plot = invalid_data.iloc[::step].copy() if len(invalid_data) > 0 else invalid_data
            else:
                df_plot = df
                valid_plot = valid_data
                invalid_plot = invalid_data

            # Plot valid data
            if len(valid_plot) > 0:
                ax.plot(valid_plot['datetime'], valid_plot['value'],
                        color='blue', alpha=0.6, linewidth=0.5, label='Valid')

            # Plot invalid data (lighter)
            if len(invalid_plot) > 0:
                ax.scatter(invalid_plot['datetime'], invalid_plot['value'],
                           color='red', alpha=0.3, s=1, label='Invalid')

            # Get station name
            station_info = meteo_stations[meteo_stations['code'] == station_code]
            station_name = station_info['name'].iloc[0] if len(station_info) > 0 else station_code
            
            # Formatting
            ax.set_title(f"{station_code}\n{station_name[:40]}", fontsize=9, fontweight='bold')
            ax.set_xlabel('Date', fontsize=8)
            ax.set_ylabel(var_display_names[var_name], fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='upper right')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.tick_params(axis='x', labelsize=7, rotation=45)
            ax.tick_params(axis='y', labelsize=7)
            
            # Statistics
            total_points = len(df)
            valid_points = len(valid_data)
            invalid_points = len(invalid_data)
            valid_pct = (valid_points / total_points * 100) if total_points > 0 else 0
            
            # Add text box with statistics
            textstr = f'Total: {total_points:,}\nValid: {valid_points:,} ({valid_pct:.1f}%)\nInvalid: {invalid_points:,}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6,
                   verticalalignment='top', bbox=props)
        
        # Hide unused subplots
        for i in range(n_stations, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Meteo Trentino {var_display_names[var_name]} - All Stations', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        var_safe_name = var_name.replace('_', '_')
        plt.savefig(output_dir / 'plots' / f'weather_{var_safe_name}_timeseries.png', 
                   dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved {var_name} plot")
        plt.close()
    
    print("  ✓ Completed all weather variable plots")


def create_visualizations(distance_matrix: pd.DataFrame,
                          coverage_df: pd.DataFrame,
                          summary_df: pd.DataFrame,
                          appa_stations: pd.DataFrame,
                          meteo_stations: pd.DataFrame,
                          variable_matching: pd.DataFrame,
                          var_dirs: Dict[str, Path],
                          output_dir: Path):
    """
    Create comprehensive visualizations for station matching and coverage analysis.
    
    Args:
        distance_matrix: Distance matrix DataFrame
        coverage_df: Detailed coverage analysis DataFrame
        summary_df: Summary DataFrame
        appa_stations: APPA stations DataFrame
        meteo_stations: Meteo Trentino stations DataFrame
        output_dir: Output directory for plots
    """
    print("\n" + "="*80)
    print("Creating visualizations")
    print("="*80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Distance Matrix Heatmap
    print("  [1/6] Creating distance matrix heatmap...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Limit to top stations for readability
    if len(distance_matrix) > 20:
        # Show top 20 closest distances per APPA station
        top_distances = []
        for appa_station in distance_matrix.index:
            top_20 = distance_matrix.loc[appa_station].nsmallest(20)
            top_distances.append(top_20)
        plot_matrix = pd.DataFrame(top_distances, index=distance_matrix.index[:len(top_distances)])
    else:
        plot_matrix = distance_matrix
    
    sns.heatmap(plot_matrix, cmap='YlOrRd', annot=False, fmt='.1f', 
                cbar_kws={'label': 'Distance (km)'}, ax=ax)
    ax.set_title('Distance Matrix: APPA Stations vs Meteo Trentino Stations', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Meteo Trentino Station Code', fontsize=12)
    ax.set_ylabel('APPA Station Code', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / '01_distance_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {plots_dir / '01_distance_matrix_heatmap.png'}")
    
    # 2. Coverage Heatmap by Variable
    print("  [2/6] Creating coverage heatmap by variable...")
    # Pivot coverage data for heatmap (now organized by APPA station and variable)
    coverage_pivot = summary_df.pivot_table(
        index='appa_station',
        columns='variable',
        values='valid_percent',
        aggfunc='first'
    )
    
    if len(coverage_pivot) > 0:
        fig, ax = plt.subplots(figsize=(14, max(8, len(coverage_pivot) * 0.5)))
        sns.heatmap(coverage_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=0, vmax=100, cbar_kws={'label': 'Valid Data Coverage (%)'},
                    ax=ax, linewidths=0.5)
        ax.set_title('Data Coverage by Variable - Variable-Specific Matching', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Weather Variable', fontsize=12)
        ax.set_ylabel('APPA Station', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / '02_coverage_by_variable_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: {plots_dir / '02_coverage_by_variable_heatmap.png'}")
    else:
        print(f"      ⚠ Skipped: No coverage data available")
    
    # 3. Best Matches Bar Chart
    print("  [3/6] Creating variable matching visualization...")
    # Create summary by APPA station showing which station provides each variable
    appa_summary = summary_df.groupby('appa_station').agg({
        'distance_km': 'mean',
        'valid_percent': 'mean',
        'available': 'sum',
        'missing_records': 'sum'
    }).reset_index()
    appa_summary['label'] = appa_summary['appa_station'].astype(str)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Average Distance
    ax = axes[0, 0]
    bars = ax.barh(appa_summary['label'], appa_summary['distance_km'], 
                   color='steelblue')
    ax.set_xlabel('Average Distance (km)', fontsize=11)
    ax.set_title('Average Distance to Matched Weather Stations', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    # Average valid percent
    ax = axes[0, 1]
    bars = ax.barh(appa_summary['label'], appa_summary['valid_percent'],
                   color='green', alpha=0.7)
    ax.set_xlabel('Average Valid Data Coverage (%)', fontsize=11)
    ax.set_title('Average Valid Data Coverage', fontsize=12, fontweight='bold')
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='100% target')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    # Available variables
    ax = axes[1, 0]
    bars = ax.barh(appa_summary['label'], appa_summary['available'],
                   color='orange', alpha=0.7)
    ax.set_xlabel('Number of Available Variables', fontsize=11)
    ax.set_title('Available Variables Count (out of 7)', fontsize=12, fontweight='bold')
    ax.axvline(x=7, color='red', linestyle='--', alpha=0.5, label='All variables')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    # Missing records
    ax = axes[1, 1]
    bars = ax.barh(appa_summary['label'], appa_summary['missing_records'],
                   color='red', alpha=0.7)
    ax.set_xlabel('Total Missing Records', fontsize=11)
    ax.set_title('Total Missing Records', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    plt.suptitle('Variable-Specific Weather Station Matching for Each APPA Station', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / '03_best_matches_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {plots_dir / '03_best_matches_summary.png'}")
    
    # 4. Coverage Distribution by Variable
    print("  [4/6] Creating coverage distribution plots...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    variables = ['temperature', 'rain', 'wind_speed', 'wind_direction', 
                 'pressure', 'radiation', 'humidity']
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        var_data = summary_df[summary_df['variable'] == var]
        
        if len(var_data) > 0:
            ax.hist(var_data['valid_percent'], bins=20, color='skyblue', 
                   edgecolor='black', alpha=0.7)
            ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='100% target')
            ax.set_xlabel('Valid Data Coverage (%)', fontsize=10)
            ax.set_ylabel('Number of Stations', fontsize=10)
            ax.set_title(f'{var.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{var.replace("_", " ").title()}', fontsize=11, fontweight='bold')
    
    # Hide unused subplot
    axes[7].axis('off')
    
    plt.suptitle('Distribution of Valid Data Coverage by Variable', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / '04_coverage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {plots_dir / '04_coverage_distribution.png'}")
    
    # 5. Variable Matching Details
    print("  [5/6] Creating variable matching details...")
    # Show which station provides which variable for each APPA station
    matching_details = summary_df[['appa_station', 'variable', 'meteo_station', 'distance_km', 'valid_percent']].copy()
    matching_details['label'] = matching_details['appa_station'].astype(str) + ' - ' + \
                               matching_details['variable'].str.replace('_', ' ').str.title()
    
    fig, axes = plt.subplots(2, 1, figsize=(16, max(12, len(matching_details) * 0.3)))
    
    # Distance by variable
    ax = axes[0]
    ax.barh(matching_details['label'], matching_details['distance_km'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_title('Distance to Matched Weather Station by Variable', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    # Coverage by variable
    ax = axes[1]
    colors = ['green' if x >= 95 else 'orange' if x >= 80 else 'red' for x in matching_details['valid_percent']]
    ax.barh(matching_details['label'], matching_details['valid_percent'], color=colors, alpha=0.7)
    ax.set_xlabel('Valid Data Coverage (%)', fontsize=11)
    ax.set_title('Valid Data Coverage by Variable', fontsize=12, fontweight='bold')
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='100% target')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    plt.suptitle('Variable-Specific Station Matching Details', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / '05_variable_matching_details.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {plots_dir / '05_variable_matching_details.png'}")
    
    # 6. Summary Dashboard
    print("  [6/6] Creating summary dashboard...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Overall statistics
    ax1 = fig.add_subplot(gs[0, :])
    avg_distance = summary_df['distance_km'].mean()
    avg_coverage = summary_df['valid_percent'].mean()
    perfect_coverage = (summary_df['valid_percent'] >= 100).sum()
    total_variables = len(summary_df)
    available_vars = summary_df['available'].sum()
    
    stats_text = f"""
    Station Matching Analysis Summary (Variable-Specific Matching)
    {'='*80}
    Total APPA Stations: {len(appa_stations)}
    Total Meteo Trentino Stations: {len(meteo_stations)}
    Total Variable-Station Pairs: {total_variables}
    Available Variables: {available_vars} / {total_variables}
    
    Coverage Statistics:
    • Average Distance: {avg_distance:.2f} km
    • Average Valid Coverage: {avg_coverage:.2f}%
    • Variables with 100% coverage: {perfect_coverage} / {total_variables}
    • Average Available Variables per APPA Station: {available_vars / len(appa_stations):.1f} / 7
    """
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.axis('off')
    
    # Distance distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(summary_df['distance_km'], bins=15, 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Distance (km)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Distance Distribution', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Coverage distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(summary_df['valid_percent'], bins=15,
            color='green', edgecolor='black', alpha=0.7)
    ax3.axvline(x=100, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Valid Coverage (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Coverage Distribution', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Available variables per APPA station
    ax4 = fig.add_subplot(gs[1, 2])
    vars_per_station = summary_df.groupby('appa_station')['available'].sum()
    var_counts = vars_per_station.value_counts().sort_index()
    ax4.bar(var_counts.index, var_counts.values, color='orange', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Number of Variables', fontsize=10)
    ax4.set_ylabel('Number of APPA Stations', fontsize=10)
    ax4.set_title('Available Variables per APPA Station', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Variable availability heatmap
    ax5 = fig.add_subplot(gs[2, :])
    if len(summary_df) == 0:
        ax5.text(0.5, 0.5, 'No coverage data\n(availability heatmap skipped)',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_axis_off()
        print("      ⚠ Skipped variable availability heatmap (no summary data).")
    else:
        var_availability = summary_df.pivot_table(
            index='appa_station',
            columns='variable',
            values='available',
            aggfunc='first'
        )
        if var_availability.size == 0:
            ax5.text(0.5, 0.5, 'No availability data\nfor heatmap',
                     ha='center', va='center', transform=ax5.transAxes)
            ax5.set_axis_off()
            print("      ⚠ Skipped variable availability heatmap (empty pivot).")
        else:
            sns.heatmap(var_availability, annot=True, fmt='.0f', cmap='RdYlGn',
                        cbar_kws={'label': 'Available (1=Yes, 0=No)'}, ax=ax5,
                        linewidths=0.5)
            ax5.set_title('Variable Availability by APPA Station', 
                         fontsize=12, fontweight='bold')
            ax5.set_xlabel('Weather Variable', fontsize=11)
            ax5.set_ylabel('APPA Station Code', fontsize=11)
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Station Matching Analysis Dashboard', fontsize=18, 
                fontweight='bold', y=0.995)
    plt.savefig(plots_dir / '06_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {plots_dir / '06_summary_dashboard.png'}")
    
    print(f"\n  ✓ All visualizations saved to: {plots_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Match APPA stations with Meteo Trentino stations and analyze data coverage'
    )
    parser.add_argument(
        '--appa-stations',
        type=Path,
        default=Path('data/appa-data/appa_monitoring_stations.csv'),
        help='Path to APPA stations CSV file'
    )
    parser.add_argument(
        '--meteo-stations-xml',
        type=Path,
        default=Path('data/meteo-trentino/stations.xml'),
        help='Path to Meteo Trentino stations XML file'
    )
    parser.add_argument(
        '--temp-rain-dir',
        type=Path,
        default=Path('data/meteo-trentino/meteo-trentino-storico-completo-temperatura-pioggia'),
        help='Directory containing temperature and rain CSV files'
    )
    parser.add_argument(
        '--wind-pressure-dir',
        type=Path,
        default=Path('data/meteo-trentino/meteo-trentino-storico-completo-vento-pressione-radiazione-umidità'),
        help='Directory containing wind, pressure, radiation, humidity CSV files'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2014-01-01',
        help='Start date for coverage analysis (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-12-31',
        help='End date for coverage analysis (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of closest weather stations to analyze for each APPA station (legacy, not used with variable-specific matching)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs for CSV reading (-1 for all CPUs)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/meteo-trentino-appa-matching'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation of cached intermediate results (distance, coverage)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("APPA - Meteo Trentino Station Matching and Coverage Analysis")
    print("=" * 80)
    print(f"Analysis Period: {args.start_date} to {args.end_date}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    print()
    
    # Load stations
    print("Step 1: Loading station data...")
    appa_stations = load_appa_stations(args.appa_stations)
    meteo_stations = load_meteo_trentino_stations(args.meteo_stations_xml)
    print(f"  ✓ Loaded {len(appa_stations)} APPA stations and {len(meteo_stations)} Meteo Trentino stations")
    print()
    
    # Calculate / load distance matrix
    print("Step 2: Calculating / loading distance matrix...")
    distance_matrix_file = args.output_dir / 'distance_matrix.csv'
    distance_matrix = load_or_compute_distance_matrix(
        appa_stations,
        meteo_stations,
        distance_matrix_file,
        force_recompute=args.force_recompute
    )
    print()
    
    # Extract ZIP files and organize by variable
    print("Step 3: Extracting ZIP files and organizing by variable...")
    var_dirs = extract_all_zip_files(
        args.temp_rain_dir,
        args.wind_pressure_dir,
        extract_dir=None,  # Extract to subdirectories in original dirs
        force_reextract=False  # Skip if already extracted
    )
    
    # Create merged CSVs per variable for faster access (optional but recommended for large datasets)
    print("\nStep 3.5: Creating merged CSV files per variable (for faster access)...")
    merged_csvs = create_merged_csvs_per_variable(
        var_dirs,
        args.output_dir,
        force_recreate=args.force_recompute  # Recreate only if forced
    )
    
    # Build or load global coverage cache (per-station, per-variable)
    print("\nStep 3.6: Building / loading global coverage cache...")
    coverage_cache = build_or_load_coverage_cache(
        var_dirs,
        meteo_stations,
        args.start_date,
        args.end_date,
        args.output_dir,
        n_jobs=args.n_jobs,
        merged_csvs=merged_csvs,
        force_recompute=args.force_recompute
    )
    
    # Find closest stations per variable (preferring 1-to-1 mappings), using cached quality
    print("\nStep 4: Finding closest weather station for each variable...")
    variable_matching = find_closest_stations_per_variable(
        distance_matrix,
        var_dirs,
        args.start_date,
        args.end_date,
        prefer_one_to_one=True,  # Prefer dedicated weather stations
        merged_csvs=merged_csvs,
        coverage_cache=coverage_cache  # Use cached coverage/quality
    )
    
    # Save variable matching
    variable_matching_file = args.output_dir / 'variable_matching.csv'
    variable_matching.to_csv(variable_matching_file, index=False)
    print(f"  ✓ Saved variable matching to {variable_matching_file}")
    print()
    
    # Analyze coverage (slice from cached global coverage)
    print("Step 5: Analyzing data coverage (using cached coverage cache)...")
    coverage_df = analyze_candidate_stations(
        variable_matching,
        var_dirs,
        args.start_date,
        args.end_date,
        output_file=args.output_dir / 'coverage_analysis.csv',
        n_jobs=args.n_jobs,
        merged_csvs=merged_csvs,
        coverage_cache=coverage_cache
    )
    print()
    
    # Generate summary
    print("Step 6: Generating summary report...")
    summary_df = generate_summary_report(
        coverage_df,
        output_file=args.output_dir / 'coverage_summary.csv'
    )
    print()
    
    # Create visualizations
    print("Step 7: Creating visualizations...")
    
    # Create station map
    create_station_map(
        appa_stations,
        meteo_stations,
        variable_matching,
        args.output_dir
    )
    
    # Create weather station time series plots
    plot_weather_station_timeseries(
        var_dirs,
        meteo_stations,
        variable_matching,
        args.start_date,
        args.end_date,
        args.output_dir,
        merged_csvs=merged_csvs  # Use merged CSVs for faster loading
    )
    
    # Create other visualizations
    create_visualizations(
        distance_matrix,
        coverage_df,
        summary_df,
        appa_stations,
        meteo_stations,
        variable_matching,
        var_dirs,
        args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("✓ Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • CSV files: distance_matrix.csv, closest_stations.csv,")
    print(f"               coverage_analysis.csv, coverage_summary.csv")
    print(f"  • Visualizations: plots/ directory (6 plots)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

