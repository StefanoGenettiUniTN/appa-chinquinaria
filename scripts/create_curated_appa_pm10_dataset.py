"""
Script to create a curated PM10 dataset from the raw APPA merged CSV.

This script performs the following operations:
1. Drops all data before 2014 (keeps 2014 onwards)
2. Interpolates all gaps shorter than 7 hours using linear interpolation
3. Copies missing values from the closest monitoring station (excluding Monte Gazza)
4. Creates complete time series for all stations
5. Saves two versions: one with interpolation metadata, one without
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path to import from chinquinaria if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth using Haversine formula."""
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


def find_contiguous_missing_periods(actual_times_set, expected_times):
    """Find all contiguous missing periods and return their start/end indices."""
    missing_periods = []
    if len(expected_times) == 0:
        return missing_periods
    
    missing_indices = []
    for i, t in enumerate(expected_times):
        if t not in actual_times_set:
            missing_indices.append(i)
    
    if not missing_indices:
        return missing_periods
    
    # Group consecutive indices
    start_idx = missing_indices[0]
    end_idx = missing_indices[0]
    
    for i in range(1, len(missing_indices)):
        if missing_indices[i] == missing_indices[i-1] + 1:
            end_idx = missing_indices[i]
        else:
            length = end_idx - start_idx + 1
            missing_periods.append((start_idx, end_idx, length))
            start_idx = missing_indices[i]
            end_idx = missing_indices[i]
    
    # Add the last period
    length = end_idx - start_idx + 1
    missing_periods.append((start_idx, end_idx, length))
    
    return missing_periods


def interpolate_station_data(station_df, max_gap_hours=5):
    """
    Interpolate missing values for a single station.
    
    Args:
        station_df: DataFrame with columns ['datetime', 'pm10', 'station_code', 'station_name']
        max_gap_hours: Maximum gap length to interpolate (strictly smaller than this)
    
    Returns:
        Tuple: (complete_df, interpolated_count, actual_interpolated_count)
    """
    if station_df.empty:
        return pd.DataFrame(), 0, 0
    
    # Get actual measurements
    actual_times = pd.to_datetime(station_df['datetime']).sort_values().unique()
    station_start = actual_times.min()
    station_end = actual_times.max()
    
    # Create expected time series (full years from start to end)
    year_start = pd.Timestamp(station_start.year, 1, 1, 0, 0, 0)
    year_end = pd.Timestamp(station_end.year, 12, 31, 23, 0, 0)
    
    # Create complete hourly time series
    expected_times = pd.date_range(start=year_start, end=year_end, freq='h')
    
    # Create a DataFrame with complete time series
    complete_df = pd.DataFrame({'datetime': expected_times})
    
    # Merge with actual data
    station_df_sorted = station_df.sort_values('datetime').reset_index(drop=True)
    station_df_sorted['datetime'] = pd.to_datetime(station_df_sorted['datetime'])
    
    complete_df = complete_df.merge(
        station_df_sorted[['datetime', 'pm10', 'station_code', 'station_name']],
        on='datetime',
        how='left'
    )
    
    # Forward fill station info and ensure correct types
    complete_df['station_code'] = complete_df['station_code'].ffill().bfill()
    complete_df['station_name'] = complete_df['station_name'].ffill().bfill()
    
    # Identify missing periods
    actual_times_set = set(actual_times)
    missing_periods = find_contiguous_missing_periods(actual_times_set, expected_times)
    
    # Identify which gaps should be interpolated (short gaps only, strictly < max_gap_hours)
    short_gaps = []
    long_gap_indices = set()
    
    for start_idx, end_idx, length in missing_periods:
        if length < max_gap_hours:  # Strictly smaller than max_gap_hours
            # Check if we have values before and after to interpolate
            if start_idx > 0 and end_idx < len(complete_df) - 1:
                before_val = complete_df.loc[start_idx - 1, 'pm10']
                after_val = complete_df.loc[end_idx + 1, 'pm10']
                if (before_val is not None and not pd.isna(before_val) and
                    after_val is not None and not pd.isna(after_val)):
                    short_gaps.append((start_idx, end_idx, length))
        else:
            for idx in range(start_idx, end_idx + 1):
                long_gap_indices.add(idx)
    
    # Create a copy for interpolation
    pm10_series = complete_df['pm10'].copy()
    
    # Initialize interpolation_method column
    complete_df['interpolation_method'] = 'actual'
    
    # Temporarily mark long gaps with a sentinel value
    for idx in long_gap_indices:
        pm10_series.loc[idx] = -999999
    
    # Interpolate (will fill short gaps, but not long gaps due to sentinel)
    # Use max_gap_hours - 1 as limit since we want strictly < max_gap_hours
    pm10_series = pm10_series.interpolate(method='linear', limit_direction='both', limit=max_gap_hours - 1)
    
    # Mark interpolated values
    for start_idx, end_idx, length in short_gaps:
        for idx in range(start_idx, end_idx + 1):
            if not pd.isna(pm10_series.loc[idx]):
                complete_df.loc[idx, 'interpolation_method'] = 'linear'
    
    # Restore long gaps as NaN
    pm10_series[pm10_series == -999999] = np.nan
    
    complete_df['pm10'] = pm10_series
    
    # Count interpolated values (only in short gaps)
    interpolated_count = sum(length for _, _, length in short_gaps)
    actual_interpolated = 0
    for start_idx, end_idx, length in short_gaps:
        gap_values = pm10_series.loc[start_idx:end_idx]
        actual_interpolated += gap_values.notna().sum()
    
    return complete_df, interpolated_count, actual_interpolated


def create_curated_appa_dataset(input_file, stations_file, output_file=None, max_gap_hours=7, start_year=2014):
    """
    Main function to create curated APPA PM10 dataset.
    
    Args:
        input_file: Path to merged_data.csv
        stations_file: Path to appa_monitoring_stations.csv
        output_file: Optional output file path (default: auto-generated)
        max_gap_hours: Maximum gap length to interpolate (strictly smaller than this, default: 5)
        start_year: Year to start from (default: 2014)
    """
    print("=" * 80)
    print("APPA PM10 Dataset Curation")
    print("=" * 80)
    
    # ============================================================================
    # Step 1: Load data
    # ============================================================================
    print(f"\n1. Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Filter for PM10 only
    df = df[df['Inquinante'] == 'PM10'].copy()
    print(f"   Loaded {len(df):,} rows (PM10 only)")
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df['Data']) + pd.to_timedelta(df['Ora'] - 1, unit='h')
    df = df.sort_values(['Stazione', 'datetime'])
    
    # Rename columns
    df = df.rename(columns={
        'Stazione': 'station_name',
        'Valore': 'pm10'
    })
    
    # Convert pm10 to numeric
    df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce')
    df = df[df['pm10'].notna()].copy()
    
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Store the actual maximum datetime from original data (before interpolation creates full years)
    actual_data_end = df['datetime'].max()
    
    # Load station metadata
    print(f"\n   Loading station coordinates from: {stations_file}")
    stations_df = pd.read_csv(stations_file)
    print(f"   Loaded {len(stations_df)} stations")
    
    # Parse coordinates
    def parse_coordinates(pos_str):
        """Parse lat,lon from Posizione string."""
        try:
            pos_str = pos_str.strip().strip('"').strip("'")
            parts = [p.strip() for p in pos_str.split(',')]
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
        except:
            return None, None
    
    stations_df['latitude'] = stations_df['Posizione'].apply(lambda x: parse_coordinates(x)[0])
    stations_df['longitude'] = stations_df['Posizione'].apply(lambda x: parse_coordinates(x)[1])
    
    # Create station_code
    stations_df['station_code'] = stations_df['IT - codice italiano'].fillna(
        stations_df['Nome stazione'].apply(lambda x: x.replace(' ', '_').upper())
    )
    
    # Match station names
    # Note: Data has "Monte Gaza" but we normalize to "Monte Gazza" (correct spelling)
    known_mappings = {
        'TRENTO PSC': 'Parco S. Chiara',
        'TRENTO VBZ': 'Via Bolzano',
        'PIANA ROTALIANA': 'Piana Rotaliana',
        'RIVA GAR': 'Riva del Garda',
        'MONTE GAZA': 'Monte Gazza',  # Remap metadata entry to corrected data name
        'ROVERETO LGP': 'Rovereto',
        'AVIO A22': 'A22 (Avio)',
        'BORGO VAL': 'Borgo Valsugana'
    }
    
    # Fix typo: "Monte Gaza" -> "Monte Gazza"
    if 'Monte Gaza' in df['station_name'].values:
        df.loc[df['station_name'] == 'Monte Gaza', 'station_name'] = 'Monte Gazza'
    
    station_name_mapping = {}
    for _, row in stations_df.iterrows():
        meta_name = row['Nome stazione']
        station_code = row['station_code']
        data_name = known_mappings.get(meta_name)
        
        # Fix typo in metadata name: "MONTE GAZA" -> "MONTE GAZZA"
        if 'MONTE GAZA' in meta_name.upper():
            meta_name = 'MONTE GAZZA'
        
        if data_name and data_name in df['station_name'].values:
            station_name_mapping[data_name] = {
                'station_code': station_code,
                'station_name': meta_name,
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
    
    # Add station_code to data
    df['station_code'] = df['station_name'].map(lambda x: station_name_mapping.get(x, {}).get('station_code', x.replace(' ', '_').upper()))
    
    # Convert station_code to string to avoid type issues
    df['station_code'] = df['station_code'].astype(str)
    stations_df['station_code'] = stations_df['station_code'].astype(str)
    
    print(f"   Found {df['station_code'].nunique()} unique stations in data")
    
    # ============================================================================
    # Step 2: Drop data before start_year
    # ============================================================================
    print(f"\n2. Dropping data before {start_year} (keeping {start_year} onwards)...")
    rows_before = len(df)
    df = df[df['datetime'].dt.year >= start_year].copy()
    rows_after = len(df)
    print(f"   Dropped {rows_before - rows_after:,} rows")
    print(f"   New date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # ============================================================================
    # Step 3: Interpolate gaps shorter than max_gap_hours
    # ============================================================================
    print(f"\n3. Interpolating gaps shorter than {max_gap_hours} hours...")
    
    all_stations_data = []
    total_interpolated = 0
    total_actual_interpolated = 0
    
    for station_code in sorted(df['station_code'].unique()):
        station_df = df[df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        print(f"   Processing {station_code} ({station_name[:50]}...)...", end=' ')
        
        interpolated_df, gap_count, actual_count = interpolate_station_data(station_df, max_gap_hours=max_gap_hours)
        
        if len(interpolated_df) > 0:
            all_stations_data.append(interpolated_df)
            total_interpolated += gap_count
            total_actual_interpolated += actual_count
            print(f"interpolated {actual_count} values")
        else:
            print("no data")
    
    # Combine all stations
    if all_stations_data:
        result_df = pd.concat(all_stations_data, ignore_index=True)
        result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
        
        print(f"\n   Total rows: {len(result_df):,}")
        print(f"   Total gaps interpolated: {total_interpolated:,} hours")
        print(f"   Total values interpolated: {total_actual_interpolated:,}")
    else:
        print("\n   No data to process!")
        return None
    
    # ============================================================================
    # Step 4: Create complete time series for all stations (from start_year to end)
    # ============================================================================
    print(f"\n4. Creating complete time series for all stations ({start_year}-{result_df['datetime'].max().year})...")
    
    # Get global date range
    global_start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    # Use actual maximum datetime from original data (before interpolation)
    # This avoids creating rows for future dates that don't have any data
    global_end = actual_data_end
    
    all_datetimes = pd.date_range(start=global_start, end=global_end, freq='h')
    expected_hours_per_station = len(all_datetimes)
    
    print(f"   Expected hours per station: {expected_hours_per_station:,}")
    
    # Create complete time series for each station
    complete_stations_data = []
    
    for station_code in sorted(result_df['station_code'].unique()):
        station_data = result_df[result_df['station_code'] == station_code].copy()
        station_name = station_data['station_name'].iloc[0]
        
        # Create complete time series
        complete_df = pd.DataFrame({'datetime': all_datetimes})
        complete_df['station_code'] = station_code
        complete_df['station_name'] = station_name
        
        # Merge with existing data
        complete_df = complete_df.merge(
            station_data[['datetime', 'pm10', 'interpolation_method']],
            on='datetime',
            how='left'
        )
        
        # Fill interpolation_method for missing values
        complete_df['interpolation_method'] = complete_df['interpolation_method'].fillna('actual')
        
        complete_stations_data.append(complete_df)
    
    result_df = pd.concat(complete_stations_data, ignore_index=True)
    result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    
    print(f"   Created complete time series: {len(result_df):,} rows")
    print(f"   Missing values remaining: {result_df['pm10'].isna().sum():,}")
    
    # ============================================================================
    # Step 4b: Copy missing values from closest station (excluding Monte Gazza)
    # ============================================================================
    print(f"\n4b. Copying missing values from closest monitoring station (excluding Monte Gazza)...")
    
    # Create a stations dataframe with coordinates for distance calculation
    stations_coords = {}
    for _, row in stations_df.iterrows():
        station_code = row['station_code']
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            stations_coords[station_code] = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'station_name': row['Nome stazione']
            }
    
    # Find Monte Gazza station code (exclude from closest station search)
    monte_gazza_code = None
    # Check in station_name_mapping
    for data_name, mapping_info in station_name_mapping.items():
        if 'MONTE' in mapping_info.get('station_name', '').upper() and 'GAZ' in mapping_info.get('station_name', '').upper():
            monte_gazza_code = mapping_info['station_code']
            break
    
    # If not found, check in stations_coords
    if monte_gazza_code is None:
        for code, info in stations_coords.items():
            if 'MONTE' in info['station_name'].upper() and 'GAZ' in info['station_name'].upper():
                monte_gazza_code = code
                break
    
    print(f"   Monte Gazza station code: {monte_gazza_code}")
    
    # Find closest station for each station (excluding Monte Gazza)
    closest_station_map = {}
    for station_code in sorted(result_df['station_code'].unique()):
        if station_code not in stations_coords:
            continue
        
        min_distance = float('inf')
        closest_code = None
        
        for other_code, other_coords in stations_coords.items():
            if other_code == station_code or other_code == monte_gazza_code:
                continue
            
            distance = haversine_distance(
                stations_coords[station_code]['latitude'],
                stations_coords[station_code]['longitude'],
                other_coords['latitude'],
                other_coords['longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_code = other_code
        
        if closest_code:
            closest_station_map[station_code] = {
                'closest_code': closest_code,
                'distance_km': min_distance,
                'closest_name': stations_coords[closest_code]['station_name']
            }
            print(f"   {station_code} ({stations_coords[station_code]['station_name'][:30]}): closest = {closest_code} ({stations_coords[closest_code]['station_name'][:30]}), distance = {min_distance:.2f} km")
    
    # Copy missing values from closest station using vectorized operations
    print(f"   Copying missing values using vectorized operations...")
    
    # Pivot data for easier access
    df_pivot = result_df.pivot_table(
        index='datetime',
        columns='station_code',
        values='pm10',
        aggfunc='first'
    )
    
    # Create a mapping column for closest station
    result_df['closest_station'] = result_df['station_code'].map(
        lambda x: closest_station_map.get(x, {}).get('closest_code') if x in closest_station_map else None
    )
    
    # Find all missing values
    missing_mask = result_df['pm10'].isna()
    
    if missing_mask.sum() > 0:
        # Create lookup DataFrame: for each (datetime, station_code) pair, get value from closest station
        # Build lookup efficiently by creating a DataFrame with all (datetime, closest_station) -> pm10 mappings
        lookup_rows = []
        
        for station_code, closest_info in closest_station_map.items():
            closest_code = closest_info['closest_code']
            if closest_code in df_pivot.columns:
                # Get all datetimes and values from closest station
                closest_data = df_pivot[[closest_code]].reset_index()
                closest_data.columns = ['datetime', 'closest_pm10']
                closest_data['target_station'] = station_code  # Station that needs this value
                closest_data['closest_station'] = closest_code
                lookup_rows.append(closest_data[['datetime', 'target_station', 'closest_pm10', 'closest_station']])
        
        if lookup_rows:
            lookup_df = pd.concat(lookup_rows, ignore_index=True)
            
            # Get missing values with their station codes
            missing_df = result_df[missing_mask][['datetime', 'station_code', 'closest_station']].copy()
            missing_df = missing_df.reset_index()  # Keep original index
            
            # Merge: match (datetime, station_code) with (datetime, target_station)
            # Note: both DataFrames have 'closest_station', so pandas will add suffixes
            merged = missing_df.merge(
                lookup_df,
                left_on=['datetime', 'station_code'],
                right_on=['datetime', 'target_station'],
                how='left',
                suffixes=('_left', '_right')
            )
            
            # Filter to valid merges (where closest_pm10 is not NaN)
            valid_mask = merged['closest_pm10'].notna()
            if valid_mask.sum() > 0:
                # Get original indices and values
                copied_indices = merged.loc[valid_mask, 'index'].values
                copied_values = merged.loc[valid_mask, 'closest_pm10'].values
                # Use 'closest_station_right' from lookup_df (the one with the actual closest station code)
                # If that doesn't exist, try 'closest_station_right' or just use the value from lookup
                closest_station_col = 'closest_station_right' if 'closest_station_right' in merged.columns else 'closest_station'
                copied_methods = merged.loc[valid_mask, closest_station_col].apply(
                    lambda x: f'copied_from_{x}' if pd.notna(x) else 'actual'
                ).values
                
                # Apply all updates at once using vectorized operations
                result_df.loc[copied_indices, 'pm10'] = copied_values
                result_df.loc[copied_indices, 'interpolation_method'] = copied_methods
                copied_count = len(copied_indices)
            else:
                copied_count = 0
        else:
            copied_count = 0
        
        # Remove temporary column
        result_df = result_df.drop(columns=['closest_station'])
    else:
        copied_count = 0
        result_df = result_df.drop(columns=['closest_station'])
    
    print(f"\n   Copied {copied_count:,} missing values from closest stations")
    print(f"   Missing values remaining after closest-station copy: {result_df['pm10'].isna().sum():,}")
    
    # ============================================================================
    # Step 4c: Distance-weighted interpolation (simple regression) for remaining NaNs
    # ============================================================================
    print(f"\n4c. Distance-weighted interpolation for remaining missing values...")
    remaining_missing_before = result_df['pm10'].isna().sum()
    
    if remaining_missing_before > 0:
        # Build distance matrix between stations using available coordinates
        print("   Building distance matrix between stations...")
        station_codes_list = sorted(stations_coords.keys())
        distance_matrix = {}
        for i, code1 in enumerate(station_codes_list):
            for j, code2 in enumerate(station_codes_list):
                if i == j:
                    distance_matrix[(code1, code2)] = 0.0
                else:
                    lat1 = stations_coords[code1]['latitude']
                    lon1 = stations_coords[code1]['longitude']
                    lat2 = stations_coords[code2]['latitude']
                    lon2 = stations_coords[code2]['longitude']
                    distance_matrix[(code1, code2)] = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Pivot to wide format for easier per-timestamp access
        df_pivot = result_df.pivot_table(
            index='datetime',
            columns='station_code',
            values='pm10',
            aggfunc='first'
        )
        
        filled_distance_weighted = 0
        
        def distance_weighted_fill(target_code, dt):
            """Fill a single (station_code, datetime) using distance-weighted neighbors."""
            if target_code not in df_pivot.columns:
                return None
            if dt not in df_pivot.index:
                return None
            row = df_pivot.loc[dt]
            available = row.dropna()
            if available.empty:
                return None
            
            distances = []
            values = []
            for other_code, val in available.items():
                if other_code == target_code:
                    continue
                key = (target_code, other_code)
                if key not in distance_matrix:
                    key = (other_code, target_code)
                dist = distance_matrix.get(key)
                if dist is None:
                    continue
                distances.append(dist)
                values.append(val)
            
            if not distances:
                return None
            
            scale = 50.0  # km, distance decay scale
            scores = np.exp(-np.array(distances) / scale)
            weights = scores / scores.sum()
            return float(np.dot(weights, np.array(values)))
        
        # Iterate over all NaNs and try to fill them
        for dt in df_pivot.index:
            for station_code in df_pivot.columns:
                if pd.isna(df_pivot.at[dt, station_code]):
                    val = distance_weighted_fill(station_code, dt)
                    if val is not None:
                        df_pivot.at[dt, station_code] = val
                        mask = (result_df['datetime'] == dt) & (result_df['station_code'] == station_code)
                        result_df.loc[mask, 'pm10'] = val
                        # Mark method only where it was 'actual' or missing
                        method_mask = mask & result_df['interpolation_method'].isin(['actual', None, ''])
                        result_df.loc[method_mask, 'interpolation_method'] = 'distance_weighted'
                        filled_distance_weighted += 1
        
        remaining_missing_after = result_df['pm10'].isna().sum()
        print(f"   Filled {filled_distance_weighted:,} values using distance-weighted interpolation.")
        print(f"   Missing values remaining after distance-weighted interpolation: {remaining_missing_after:,}")
    else:
        filled_distance_weighted = 0
        print("   No remaining missing values to fill with distance-weighted interpolation.")
    
    # ============================================================================
    # Step 5: Save curated dataset
    # ============================================================================
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.parent / f"merged_pm10_hourly_curated_{timestamp}.csv"
    
    print(f"\n5. Saving curated dataset to: {output_file}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Reorder columns for output (with interpolation columns)
    output_columns = ['datetime', 'station_code', 'station_name', 'pm10', 'interpolation_method']
    result_df[output_columns].to_csv(output_file, index=False)
    
    print(f"   ✓ Saved {len(result_df):,} rows")
    
    # Save version without interpolation columns
    output_file_no_interp = output_file.parent / f"{output_file.stem}_no_interp_metadata.csv"
    print(f"\n5b. Saving curated dataset (without interpolation metadata) to: {output_file_no_interp}")
    
    # Columns without interpolation metadata
    output_columns_no_interp = ['datetime', 'station_code', 'station_name', 'pm10']
    result_df[output_columns_no_interp].to_csv(output_file_no_interp, index=False)
    
    print(f"   ✓ Saved {len(result_df):,} rows")
    
    # ============================================================================
    # Summary statistics
    # ============================================================================
    print(f"\n6. Summary:")
    print(f"   Original dataset: {len(pd.read_csv(input_file)):,} rows")
    print(f"   Curated dataset: {len(result_df):,} rows")
    print(f"   Stations in curated dataset: {len(result_df['station_code'].unique())}")
    print(f"   Date range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    print(f"   Gaps interpolated (< {max_gap_hours}h, linear): {total_actual_interpolated:,} hours")
    print(f"   Values copied from closest stations: {copied_count:,}")
    print(f"   Values filled via distance-weighted interpolation: {filled_distance_weighted:,}")
    
    # Per-station summary
    print(f"\n   Per-station row counts and completeness:")
    for station_code in sorted(result_df['station_code'].unique()):
        station_name = result_df[result_df['station_code'] == station_code]['station_name'].iloc[0]
        station_data = result_df[result_df['station_code'] == station_code]
        count = len(station_data)
        missing = station_data['pm10'].isna().sum()
        completeness = ((count - missing) / count * 100) if count > 0 else 0.0
        
        # Count interpolated, copied, and distance-weighted values
        interp_count = (station_data['interpolation_method'] == 'linear').sum()
        copied_count_station = station_data['interpolation_method'].str.startswith('copied_from_', na=False).sum()
        distw_count_station = (station_data['interpolation_method'] == 'distance_weighted').sum()
        
        interp_pct = (interp_count / count * 100) if count > 0 else 0.0
        copied_pct = (copied_count_station / count * 100) if count > 0 else 0.0
        distw_pct = (distw_count_station / count * 100) if count > 0 else 0.0
        
        print(
            f"     {station_code} ({station_name[:45]}): {count:,} rows, {missing:,} missing "
            f"({completeness:.2f}% complete), {interp_count:,} interpolated ({interp_pct:.2f}%), "
            f"{copied_count_station:,} copied ({copied_pct:.2f}%), "
            f"{distw_count_station:,} dist-weighted ({distw_pct:.2f}%)"
        )
    
    print("\n" + "=" * 80)
    print("Dataset curation complete!")
    print("=" * 80)
    
    return result_df


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "appa-data" / "merged_data.csv"
    stations_file = project_root / "data" / "appa-data" / "appa_monitoring_stations.csv"
    output_file = project_root / "data" / "appa-data" / "merged_pm10_hourly_curated.csv"
    
    # Run curation (starting from 2014)
    create_curated_appa_dataset(
        input_file, 
        stations_file,
        output_file=output_file, 
        max_gap_hours=7, 
        start_year=2014
    )

