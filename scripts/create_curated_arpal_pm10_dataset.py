"""
Script to create a curated PM10 dataset from the raw ARPAL merged CSV.

This script performs the following operations:
1. Drops all data before 2010 (keeps 2010 onwards)
2. Drops stations: ARPAL_003, ARPAL_009, ARPAL_015, ARPAL_024
3. Interpolates all gaps shorter than 4 hours using linear interpolation
4. Final pass: interpolates remaining gaps shorter than 2 hours
5. Distance-weighted interpolation for ALL remaining missing values using softmax(e^(-distance))
6. Saves the curated dataset with confidence metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

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


def calculate_distance_matrix(stations_df):
    """Calculate distance matrix between stations in kilometers."""
    n_stations = len(stations_df)
    distance_matrix = {}
    station_codes = stations_df['station_code'].values
    
    for i in range(n_stations):
        for j in range(n_stations):
            if i == j:
                distance_matrix[(station_codes[i], station_codes[j])] = 0.0
            else:
                lat1 = stations_df.iloc[i]['latitude']
                lon1 = stations_df.iloc[i]['longitude']
                lat2 = stations_df.iloc[j]['latitude']
                lon2 = stations_df.iloc[j]['longitude']
                distance_matrix[(station_codes[i], station_codes[j])] = haversine_distance(lat1, lon1, lat2, lon2)
    
    return distance_matrix


def softmax(x):
    """Compute softmax values for array x."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def distance_weighted_interpolation(
    target_station_code,
    target_datetime,
    available_stations_data,
    distance_matrix,
    stations_df,
    min_stations=1,
    max_distance_km=200.0
):
    """
    Interpolate missing value using distance-weighted average of nearby stations.
    
    Args:
        target_station_code: Station code for which we're interpolating
        target_datetime: Datetime for which we need a value
        available_stations_data: Dict {station_code: pm10_value} for stations with data at this time
        distance_matrix: Dict mapping (station1, station2) -> distance_km
        stations_df: DataFrame with station coordinates
        min_stations: Minimum number of stations required (default: 1)
        max_distance_km: Maximum distance to consider (default: 200 km)
    
    Returns:
        Tuple: (interpolated_value, confidence_score, min_distance, n_stations_used)
        Returns (None, 0.0, None, 0) if interpolation not possible
    """
    if len(available_stations_data) < min_stations:
        return None, 0.0, None, 0
    
    # Get distances from target station to all available stations
    distances = []
    values = []
    station_codes_used = []
    
    for station_code, pm10_value in available_stations_data.items():
        if station_code == target_station_code:
            continue
        
        # Get distance
        if (target_station_code, station_code) in distance_matrix:
            distance = distance_matrix[(target_station_code, station_code)]
        elif (station_code, target_station_code) in distance_matrix:
            distance = distance_matrix[(station_code, target_station_code)]
        else:
            continue
        
        # Filter by max distance
        if distance <= max_distance_km:
            distances.append(distance)
            values.append(pm10_value)
            station_codes_used.append(station_code)
    
    if len(values) < min_stations:
        return None, 0.0, None, 0
    
    # Calculate scores: e^(-distance)
    # Use a scaling factor to make distances more meaningful
    # For distances in km, e^(-distance/scale) where scale controls decay rate
    # Using scale=50 means: 1km -> weight ~0.98, 10km -> weight ~0.82, 50km -> weight ~0.37
    scale = 50.0  # Distance decay scale in km
    scores = np.exp(-np.array(distances) / scale)
    
    # Apply softmax to get normalized weights
    weights = softmax(scores)
    
    # Calculate weighted average
    interpolated_value = np.average(values, weights=weights)
    
    # Calculate confidence metrics
    min_distance = min(distances)
    sum_weights = weights.sum()
    
    # Confidence score: combination of min distance and number of stations
    # Higher confidence when:
    # - Closer stations (lower min_distance)
    # - More stations contributing
    # Confidence ranges from 0 (low) to 1 (high)
    # Formula: confidence = exp(-min_distance/20) * min(1.0, n_stations/5)
    distance_confidence = np.exp(-min_distance / 20.0)  # Decay: 1km->0.95, 10km->0.61, 50km->0.08
    station_count_confidence = min(1.0, len(station_codes_used) / 5.0)  # Max at 5+ stations
    confidence = distance_confidence * station_count_confidence
    
    return interpolated_value, confidence, min_distance, len(station_codes_used)


def find_contiguous_missing_periods(actual_times_set, expected_times):
    """Find all contiguous missing periods and return their start/end indices."""
    missing_periods = []
    current_start_idx = None
    
    for i, t in enumerate(expected_times):
        if t not in actual_times_set:
            if current_start_idx is None:
                current_start_idx = i
        else:
            if current_start_idx is not None:
                length = i - current_start_idx
                missing_periods.append((current_start_idx, i - 1, length))
                current_start_idx = None
    
    if current_start_idx is not None:
        length = len(expected_times) - current_start_idx
        missing_periods.append((current_start_idx, len(expected_times) - 1, length))
    
    return missing_periods


def process_datetime_group_worker(args):
    """
    Worker function for parallel processing of datetime groups.
    Must be at module level for multiprocessing to work.
    """
    datetime_val, missing_stations, available_data, dist_matrix, stations_dict_data = args
    # Reconstruct stations_df from dict for the interpolation function
    stations_df_local = pd.DataFrame.from_dict(stations_dict_data, orient='index').reset_index()
    stations_df_local.columns = ['station_code', 'latitude', 'longitude']
    
    results = []
    confidence_stats = []
    
    for ms in missing_stations:
        target_station = ms['station_code']
        available_data_for_target = ms['available_data']
        
        # Perform distance-weighted interpolation
        interpolated_value, confidence, min_dist, n_stations = distance_weighted_interpolation(
            target_station,
            datetime_val,
            available_data_for_target,
            dist_matrix,
            stations_df_local,
            min_stations=1,
            max_distance_km=200.0
        )
        
        if interpolated_value is not None:
            results.append({
                'idx': ms['idx'],
                'pm10': interpolated_value,
                'confidence': confidence,
                'method': 'distance_weighted'
            })
            confidence_stats.append({
                'confidence': confidence,
                'min_distance': min_dist,
                'n_stations': n_stations
            })
    
    return results, confidence_stats


def interpolate_station_data(station_df, max_gap_hours=4):
    """
    Interpolate gaps shorter than max_gap_hours for a single station.
    
    Args:
        station_df: DataFrame with datetime and pm10 columns for one station
        max_gap_hours: Maximum gap length to interpolate (default: 4 hours)
    
    Returns:
        Tuple: (complete_df, interpolated_count, actual_interpolated)
    """
    if station_df.empty:
        return station_df.copy(), 0, 0
    
    # Get actual measurement times
    actual_times = pd.to_datetime(station_df['datetime']).sort_values()
    station_start = actual_times.min()
    station_end = actual_times.max()
    
    # Round down to start of year and round up to end of year to ensure we catch boundary gaps
    year_start = pd.Timestamp(station_start.year, 1, 1, 0, 0, 0)
    year_end = pd.Timestamp(station_start.year, 12, 31, 23, 0, 0)
    
    # If station spans multiple years, use the full range
    if station_start.year != station_end.year:
        range_start = pd.Timestamp(station_start.year, station_start.month, station_start.day, 0, 0, 0)
        range_end = pd.Timestamp(station_end.year, station_end.month, station_end.day, 23, 0, 0)
    else:
        range_start = year_start
        range_end = year_end
    
    # Create complete hourly time series
    expected_times = pd.date_range(start=range_start, end=range_end, freq='h')
    
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
    
    # Identify which gaps should be interpolated (short gaps only)
    short_gaps = []
    long_gap_indices = set()
    
    for start_idx, end_idx, length in missing_periods:
        if length <= max_gap_hours:
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
    
    # Temporarily mark long gaps with a sentinel value
    for idx in long_gap_indices:
        pm10_series.loc[idx] = -999999
    
    # Interpolate (will fill short gaps, but not long gaps due to sentinel)
    pm10_series = pm10_series.interpolate(method='linear', limit_direction='both', limit=max_gap_hours)
    
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


def create_curated_arpal_dataset(input_file, stations_file, output_file=None, max_gap_hours=4, start_year=2010, confidence_threshold=0.25):
    """
    Main function to create curated ARPAL PM10 dataset.
    
    Args:
        input_file: Path to merged_pm10_hourly.csv
        stations_file: Path to arpal_pm10_stations.csv
        output_file: Optional output file path (default: auto-generated)
        max_gap_hours: Maximum gap length to interpolate (default: 4 hours)
        start_year: Year to start from (default: 2010)
        confidence_threshold: Minimum confidence threshold for stations (default: 0.25)
                             Stations with confidence below this will be dropped before interpolation
    """
    print("=" * 80)
    print("ARPAL PM10 Dataset Curation")
    print("=" * 80)
    
    # ============================================================================
    # Step 1: Load data
    # ============================================================================
    print(f"\n1. Loading data from: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime'])
    print(f"   Loaded {len(df):,} rows")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    print(f"\n   Loading station coordinates from: {stations_file}")
    stations_df = pd.read_csv(stations_file)
    print(f"   Loaded {len(stations_df)} stations")
    
    unique_stations = df['station_code'].unique()
    print(f"   Found {len(unique_stations)} unique stations in data")
    
    # ============================================================================
    # Step 2: Drop data before start_year
    # ============================================================================
    print(f"\n2. Dropping data before {start_year} (keeping {start_year} onwards)...")
    rows_before = len(df)
    df['datetime'] = pd.to_datetime(df['datetime'])
    cutoff_date = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    df = df[df['datetime'] >= cutoff_date].copy()
    rows_after = len(df)
    print(f"   Dropped {rows_before - rows_after:,} rows")
    print(f"   New date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # ============================================================================
    # Step 3: Drop stations ARPAL_003, ARPAL_009, ARPAL_015, ARPAL_024
    # ============================================================================
    print(f"\n3. Dropping stations: ARPAL_003, ARPAL_009, ARPAL_015, ARPAL_024...")
    stations_to_drop = ['ARPAL_003', 'ARPAL_009', 'ARPAL_015', 'ARPAL_024']
    rows_before = len(df)
    df = df[~df['station_code'].isin(stations_to_drop)].copy()
    rows_after = len(df)
    print(f"   Dropped {rows_before - rows_after:,} rows")
    print(f"   Remaining stations: {sorted(df['station_code'].unique())}")
    
    # Also remove from stations_df
    stations_df = stations_df[~stations_df['station_code'].isin(stations_to_drop)].copy()
    
    # ============================================================================
    # Step 4: Interpolate all gaps shorter than max_gap_hours
    # ============================================================================
    print(f"\n4. Interpolating gaps shorter than {max_gap_hours} hours...")
    
    interpolated_dfs = []
    total_interpolated = 0
    total_actual_interpolated = 0
    
    for station_code in df['station_code'].unique():
        station_df = df[df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        print(f"   Processing {station_code} ({station_name[:40]}...)...", end=" ")
        
        interpolated_df, gap_count, actual_count = interpolate_station_data(
            station_df, 
            max_gap_hours=max_gap_hours
        )
        
        interpolated_dfs.append(interpolated_df)
        total_interpolated += gap_count
        total_actual_interpolated += actual_count
        
        print(f"interpolated {actual_count} values")
    
    # Combine all stations
    print(f"\n5. Combining all stations...")
    result_df = pd.concat(interpolated_dfs, ignore_index=True)
    result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    result_df = result_df[['datetime', 'station_code', 'station_name', 'pm10']]
    
    # Add interpolation method tracking
    result_df['interpolation_method'] = None
    result_df['interpolation_confidence'] = np.nan
    
    print(f"   Total rows: {len(result_df):,}")
    print(f"   Total gaps interpolated: {total_interpolated:,} hours")
    print(f"   Total values interpolated: {total_actual_interpolated:,}")
    
    # Track first pass totals
    first_pass_interpolated = total_actual_interpolated
    
    # ============================================================================
    # Step 6: Final pass - interpolate gaps shorter than 2 hours
    # ============================================================================
    print(f"\n6. Final pass: interpolating remaining gaps shorter than 2 hours...")
    
    final_interpolated_dfs = []
    final_total_interpolated = 0
    final_total_actual_interpolated = 0
    
    for station_code in result_df['station_code'].unique():
        station_df = result_df[result_df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        print(f"   Processing {station_code} ({station_name[:40]}...)...", end=" ")
        
        interpolated_df, gap_count, actual_count = interpolate_station_data(
            station_df, 
            max_gap_hours=2
        )
        
        final_interpolated_dfs.append(interpolated_df)
        final_total_interpolated += gap_count
        final_total_actual_interpolated += actual_count
        
        print(f"interpolated {actual_count} values")
    
    # Combine all stations after final interpolation
    result_df = pd.concat(final_interpolated_dfs, ignore_index=True)
    result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    
    # Ensure interpolation tracking columns exist
    if 'interpolation_method' not in result_df.columns:
        result_df['interpolation_method'] = None
    if 'interpolation_confidence' not in result_df.columns:
        result_df['interpolation_confidence'] = np.nan
    
    # Mark linear interpolation values
    for station_code in result_df['station_code'].unique():
        station_df = result_df[result_df['station_code'] == station_code].copy()
        # Find values that were NaN before but are now filled (linear interpolation)
        # This is approximate - we mark all non-NaN values that don't have a method yet
        mask = (result_df['station_code'] == station_code) & \
               (result_df['pm10'].notna()) & \
               (result_df['interpolation_method'].isna())
        result_df.loc[mask, 'interpolation_method'] = 'linear'
    
    print(f"   Final gaps interpolated: {final_total_interpolated:,} hours")
    print(f"   Final values interpolated: {final_total_actual_interpolated:,}")
    
    # ============================================================================
    # Step 7: Create complete time series for all stations before distance-weighted interpolation
    # ============================================================================
    print(f"\n7. Creating complete time series for all stations (2010-2024)...")
    
    # Define global time range
    global_start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    global_end = pd.Timestamp("2024-12-31 23:00:00")  # End of 2024
    
    # Create complete hourly time series
    all_datetimes = pd.date_range(start=global_start, end=global_end, freq='h')
    print(f"   Expected hours per station: {len(all_datetimes):,}")
    
    # Create complete DataFrame with all stations and all hours
    complete_rows = []
    for station_code in sorted(result_df['station_code'].unique()):
        station_name = result_df[result_df['station_code'] == station_code]['station_name'].iloc[0]
        for dt in all_datetimes:
            complete_rows.append({
                'datetime': dt,
                'station_code': station_code,
                'station_name': station_name,
                'pm10': np.nan,
                'interpolation_method': None,
                'interpolation_confidence': np.nan
            })
    
    complete_df = pd.DataFrame(complete_rows)
    
    # Merge with existing data (this will fill in actual measurements)
    result_df_with_methods = result_df.copy()
    if 'interpolation_method' not in result_df_with_methods.columns:
        result_df_with_methods['interpolation_method'] = None
    if 'interpolation_confidence' not in result_df_with_methods.columns:
        result_df_with_methods['interpolation_confidence'] = np.nan
    
    # Merge on datetime and station_code, keeping existing values
    # Use left join to keep all rows from complete_df, then fill with existing values
    complete_df = complete_df.merge(
        result_df_with_methods[['datetime', 'station_code', 'pm10', 'interpolation_method', 'interpolation_confidence']],
        on=['datetime', 'station_code'],
        how='left',
        suffixes=('', '_existing')
    )
    
    # Fill in existing values (existing takes priority)
    complete_df['pm10'] = complete_df['pm10_existing'].combine_first(complete_df['pm10'])
    complete_df['interpolation_method'] = complete_df['interpolation_method_existing'].combine_first(complete_df['interpolation_method'])
    complete_df['interpolation_confidence'] = complete_df['interpolation_confidence_existing'].combine_first(complete_df['interpolation_confidence'])
    
    # Drop temporary columns
    complete_df = complete_df[['datetime', 'station_code', 'station_name', 'pm10', 'interpolation_method', 'interpolation_confidence']]
    
    # Verify merge worked correctly
    existing_count = result_df_with_methods['pm10'].notna().sum()
    merged_count = complete_df['pm10'].notna().sum()
    print(f"   Existing non-null values: {existing_count:,}")
    print(f"   After merge non-null values: {merged_count:,}")
    if existing_count != merged_count:
        print(f"   ⚠️  Warning: Some values may have been lost in merge!")
    
    # Update result_df
    result_df = complete_df.copy()
    
    print(f"   Created complete time series: {len(result_df):,} rows")
    print(f"   Missing values to fill: {result_df['pm10'].isna().sum():,}")
    
    # ============================================================================
    # Step 7b: Filter stations with 2+ consecutive years of missing data (no records)
    # This must be done BEFORE distance-weighted interpolation so we can identify
    # which years have actual measurements vs NaN
    # ============================================================================
    print(f"\n7b. Filtering stations with 2+ consecutive years of missing data (no records)...")
    
    # Identify stations with consecutive years of missing data
    # A year is considered "missing" if it has no actual measurements (all NaN)
    result_df['year'] = pd.to_datetime(result_df['datetime']).dt.year
    year_min = result_df['year'].min()
    year_max = result_df['year'].max()
    
    stations_to_drop = []
    
    for station_code in sorted(result_df['station_code'].unique()):
        station_df = result_df[result_df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        # Check each year for actual measurements (non-null pm10 values)
        years_with_actual_data = set()
        for year in range(year_min, year_max + 1):
            year_data = station_df[station_df['year'] == year]
            # Check if this year has any non-null pm10 values (actual measurements)
            actual_count = year_data['pm10'].notna().sum()
            
            if actual_count > 0:
                years_with_actual_data.add(year)
        
        # Find consecutive missing years
        consecutive_missing = 0
        max_consecutive_missing = 0
        
        for year in range(year_min, year_max + 1):
            if year not in years_with_actual_data:
                consecutive_missing += 1
                max_consecutive_missing = max(max_consecutive_missing, consecutive_missing)
            else:
                consecutive_missing = 0
        
        # If station has 2+ consecutive years with no actual data, mark for removal
        if max_consecutive_missing >= 2:
            stations_to_drop.append((station_code, station_name, max_consecutive_missing, years_with_actual_data))
    
    if len(stations_to_drop) > 0:
        print(f"   Found {len(stations_to_drop)} stations with 2+ consecutive years of missing data:")
        for station_code, station_name, max_consec_missing, years_with_data in stations_to_drop:
            missing_years = set(range(year_min, year_max + 1)) - years_with_data
            print(f"     {station_code} ({station_name[:50]}): {max_consec_missing} consecutive missing years")
            print(f"       Missing years: {sorted(missing_years)}")
        
        drop_station_codes = [s[0] for s in stations_to_drop]
        
        rows_before = len(result_df)
        result_df = result_df[~result_df['station_code'].isin(drop_station_codes)].copy()
        rows_after = len(result_df)
        print(f"\n   Dropped {len(drop_station_codes)} stations")
        print(f"   Dropped {rows_before - rows_after:,} rows")
        
        # Remove from stations_df
        stations_df = stations_df[~stations_df['station_code'].isin(drop_station_codes)].copy()
        
        print(f"   Remaining stations: {sorted(result_df['station_code'].unique())}")
    else:
        print(f"   No stations found with 2+ consecutive years of missing data")
    
    # Remove temporary 'year' column
    if 'year' in result_df.columns:
        result_df = result_df.drop(columns=['year'])
    
    # ============================================================================
    # Step 8: Calculate and log confidence per station
    # ============================================================================
    print(f"\n8. Calculating confidence per station based on distances...")
    
    # Calculate distance matrix
    print("   Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(stations_df)
    
    # Calculate confidence for each station based on distances to all other stations
    station_confidence_info = []
    for station_code in sorted(result_df['station_code'].unique()):
        # Get distances to all other stations
        distances = []
        for other_station in sorted(result_df['station_code'].unique()):
            if station_code == other_station:
                continue
            if (station_code, other_station) in distance_matrix:
                distances.append(distance_matrix[(station_code, other_station)])
            elif (other_station, station_code) in distance_matrix:
                distances.append(distance_matrix[(other_station, station_code)])
        
        if distances:
            min_distance = min(distances)
            n_stations = len(distances)
            distance_confidence = np.exp(-min_distance / 20.0)
            station_count_confidence = min(1.0, n_stations / 5.0)
            confidence = distance_confidence * station_count_confidence
            
            station_name = result_df[result_df['station_code'] == station_code]['station_name'].iloc[0]
            station_confidence_info.append({
                'station_code': station_code,
                'station_name': station_name,
                'min_distance_km': min_distance,
                'n_nearby_stations': n_stations,
                'confidence': confidence
            })
    
    conf_df = pd.DataFrame(station_confidence_info)
    print("\n   Station confidence (based on distances to other stations):")
    print(conf_df.to_string(index=False))
    
    # ============================================================================
    # Step 8a: Display missing hours per year per station
    # ============================================================================
    print(f"\n8a. Missing hours per year per station:")
    
    # Calculate missing values per year per station
    result_df['year'] = pd.to_datetime(result_df['datetime']).dt.year
    year_min = result_df['year'].min()
    year_max = result_df['year'].max()
    
    missing_per_year_rows = []
    
    for station_code in sorted(result_df['station_code'].unique()):
        station_df = result_df[result_df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        # Get actual NON-NULL measurements per year (not just row count)
        station_df['year'] = pd.to_datetime(station_df['datetime']).dt.year
        # Count only non-null pm10 values, not all rows
        actual_per_year = station_df[station_df['pm10'].notna()].groupby('year').size().to_dict()
        
        # For each year, calculate expected vs actual
        for year in range(year_min, year_max + 1):
            year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
            year_end = pd.Timestamp(f"{year}-12-31 23:00:00")
            expected_hours = len(pd.date_range(start=year_start, end=year_end, freq='h'))
            actual_hours = actual_per_year.get(year, 0)
            missing_hours = expected_hours - actual_hours
            
            missing_per_year_rows.append({
                'year': year,
                'station_code': station_code,
                'station_name': station_name,
                'expected_hours': expected_hours,
                'actual_hours': actual_hours,
                'missing_hours': missing_hours
            })
    
    missing_per_year = pd.DataFrame(missing_per_year_rows)
    
    # Pivot for better visualization
    missing_pivot = missing_per_year.pivot_table(
        index=['station_code', 'station_name'],
        columns='year',
        values='missing_hours',
        fill_value=0
    )
    
    print("\nMissing hours per year per station:")
    print(missing_pivot.to_string())
    
    # Summary by year (all stations combined)
    print("\nMissing values summary by year (all stations combined):")
    yearly_summary = missing_per_year.groupby('year').agg({
        'expected_hours': 'sum',
        'actual_hours': 'sum',
        'missing_hours': 'sum'
    })
    yearly_summary['missing_percentage'] = (yearly_summary['missing_hours'] / 
                                           yearly_summary['expected_hours'] * 100)
    yearly_summary.columns = ['expected_hours', 'actual_hours', 'missing_hours', 'missing_percentage']
    print(yearly_summary.to_string())
    
    # ============================================================================
    # Step 8b: Filter stations by confidence threshold (only stations with missing years)
    # ============================================================================
    if confidence_threshold > 0.0:
        print(f"\n8b. Filtering stations by confidence threshold ({confidence_threshold:.3f})...")
        print(f"   Note: Only stations with missing years are filtered. Complete stations are kept regardless of confidence.")
        
        # Identify stations with missing years (missing data records)
        # A station has missing years if it doesn't have data for all expected hours
        stations_with_missing_years = []
        expected_hours_per_station = len(all_datetimes)  # From step 7
        
        for station_code in sorted(result_df['station_code'].unique()):
            station_data = result_df[result_df['station_code'] == station_code]
            actual_hours = station_data['pm10'].notna().sum()
            
            # If station has fewer hours than expected, it has missing years
            if actual_hours < expected_hours_per_station:
                stations_with_missing_years.append(station_code)
        
        print(f"   Found {len(stations_with_missing_years)} stations with missing years:")
        for station_code in stations_with_missing_years:
            station_name = result_df[result_df['station_code'] == station_code]['station_name'].iloc[0]
            station_data = result_df[result_df['station_code'] == station_code]
            actual_hours = station_data['pm10'].notna().sum()
            missing_hours = expected_hours_per_station - actual_hours
            print(f"     {station_code} ({station_name[:50]}): {missing_hours:,} missing hours")
        
        # Among stations with missing years, find those below confidence threshold
        low_confidence_stations = []
        if conf_df is not None and len(conf_df) > 0 and 'station_code' in conf_df.columns:
            for station_code in stations_with_missing_years:
                station_conf_row = conf_df[conf_df['station_code'] == station_code]
                if len(station_conf_row) > 0:
                    station_conf = station_conf_row['confidence'].iloc[0]
                    if station_conf < confidence_threshold:
                        low_confidence_stations.append(station_code)
        
        if len(low_confidence_stations) > 0:
            print(f"\n   Dropping {len(low_confidence_stations)} stations with missing years and confidence < {confidence_threshold:.3f}:")
            for station_code in low_confidence_stations:
                if conf_df is not None and len(conf_df) > 0 and 'station_code' in conf_df.columns:
                    station_name = conf_df[conf_df['station_code'] == station_code]['station_name'].iloc[0]
                    station_conf = conf_df[conf_df['station_code'] == station_code]['confidence'].iloc[0]
                    print(f"     {station_code} ({station_name[:50]}): confidence = {station_conf:.3f}")
                else:
                    print(f"     {station_code}: confidence < {confidence_threshold:.3f}")
            
            # Remove low confidence stations from result_df
            rows_before = len(result_df)
            result_df = result_df[~result_df['station_code'].isin(low_confidence_stations)].copy()
            rows_after = len(result_df)
            print(f"   Dropped {rows_before - rows_after:,} rows")
            
            # Remove from stations_df
            stations_df = stations_df[~stations_df['station_code'].isin(low_confidence_stations)].copy()
            
            # Remove from conf_df if it exists and has the required column
            if conf_df is not None and len(conf_df) > 0 and 'station_code' in conf_df.columns:
                conf_df = conf_df[~conf_df['station_code'].isin(low_confidence_stations)].copy()
            
            # Recalculate distance matrix with remaining stations
            print(f"   Recalculating distance matrix with {len(stations_df)} remaining stations...")
            distance_matrix = calculate_distance_matrix(stations_df)
            
            print(f"   Remaining stations: {sorted(result_df['station_code'].unique())}")
        else:
            print(f"   No stations with missing years below threshold - all stations kept")
    else:
        print(f"\n8b. Skipping confidence threshold filter (threshold = 0.0)")
    
    # ============================================================================
    # Step 9: Distance-weighted interpolation for remaining missing values
    # ============================================================================
    print(f"\n9. Distance-weighted interpolation for remaining missing values...")
    
    # Pivot data to wide format for easier access
    print("   Preparing data for distance-weighted interpolation...")
    df_pivot = result_df.pivot_table(
        index='datetime',
        columns='station_code',
        values='pm10',
        aggfunc='first'
    )
    
    # Find all missing values
    missing_mask = result_df['pm10'].isna()
    missing_rows = result_df[missing_mask].copy()
    
    print(f"   Found {len(missing_rows):,} missing values to interpolate")
    
    if len(missing_rows) > 0:
        # Group by datetime for efficiency
        missing_by_datetime = missing_rows.groupby('datetime')
        n_datetimes = len(missing_by_datetime)
        
        print(f"   Processing {n_datetimes:,} unique datetimes...")
        
        # Prepare datetime groups as list of tuples
        datetime_groups = []
        for datetime_val, group in missing_by_datetime:
            if datetime_val not in df_pivot.index:
                continue
            available_data = df_pivot.loc[datetime_val].dropna().to_dict()
            if len(available_data) == 0:
                continue
            
            # Prepare missing stations for this datetime
            missing_stations = []
            for idx, row in group.iterrows():
                target_station = row['station_code']
                available_data_for_target = {k: v for k, v in available_data.items() if k != target_station}
                if len(available_data_for_target) > 0:
                    missing_stations.append({
                        'idx': idx,
                        'station_code': target_station,
                        'available_data': available_data_for_target
                    })
            
            if len(missing_stations) > 0:
                datetime_groups.append((datetime_val, missing_stations, available_data))
        
        print(f"   Using {cpu_count()} CPU cores for parallel processing...")
        
        # Prepare stations data as dict for serialization
        stations_dict_data = stations_df.set_index('station_code')[['latitude', 'longitude']].to_dict('index')
        
        # Add distance_matrix and stations_dict to each datetime group
        datetime_groups_with_data = [
            (dt_val, missing_stations, available_data, distance_matrix, stations_dict_data)
            for dt_val, missing_stations, available_data in datetime_groups
        ]
        
        # Process in parallel
        all_updates = []
        all_confidence_stats = []
        
        # Use multiprocessing Pool
        with Pool(processes=cpu_count()) as pool:
            # Process in chunks to show progress
            chunk_size = max(1000, n_datetimes // 100)  # Process in chunks
            total_chunks = (len(datetime_groups_with_data) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(datetime_groups_with_data))
                chunk = datetime_groups_with_data[start_idx:end_idx]
                
                # Process chunk in parallel
                chunk_results = pool.map(process_datetime_group_worker, chunk)
                
                # Collect results
                for results, conf_stats in chunk_results:
                    all_updates.extend(results)
                    all_confidence_stats.extend(conf_stats)
                
                # Show progress
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == total_chunks - 1:
                    print(f"   Processed chunk {chunk_idx + 1}/{total_chunks} ({len(all_updates):,} interpolated so far)")
        
        # Apply all updates to dataframe using vectorized operations
        print(f"   Applying {len(all_updates):,} interpolated values to dataframe...")
        if len(all_updates) > 0:
            # Convert to DataFrame for efficient updates
            updates_df = pd.DataFrame(all_updates)
            updates_df = updates_df.set_index('idx')
            
            # Update in bulk
            result_df.loc[updates_df.index, 'pm10'] = updates_df['pm10']
            result_df.loc[updates_df.index, 'interpolation_confidence'] = updates_df['confidence']
            result_df.loc[updates_df.index, 'interpolation_method'] = updates_df['method']
        
        distance_interpolated_count = len(all_updates)
        confidence_stats = all_confidence_stats
        
        # Print confidence statistics
        if confidence_stats:
            conf_df = pd.DataFrame(confidence_stats)
            print(f"\n   Distance-weighted interpolation statistics:")
            print(f"     Total interpolated: {distance_interpolated_count:,} values")
            print(f"     Mean confidence: {conf_df['confidence'].mean():.3f}")
            print(f"     Median confidence: {conf_df['confidence'].median():.3f}")
            print(f"     Min confidence: {conf_df['confidence'].min():.3f}")
            print(f"     Max confidence: {conf_df['confidence'].max():.3f}")
            print(f"     Mean min distance: {conf_df['min_distance'].mean():.2f} km")
            print(f"     Median min distance: {conf_df['min_distance'].median():.2f} km")
            print(f"     Mean stations used: {conf_df['n_stations'].mean():.1f}")
            
            # Confidence distribution
            print(f"\n   Confidence distribution:")
            conf_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
            for i in range(len(conf_bins) - 1):
                count = ((conf_df['confidence'] >= conf_bins[i]) & 
                        (conf_df['confidence'] < conf_bins[i+1])).sum()
                pct = (count / len(conf_df) * 100) if len(conf_df) > 0 else 0
                print(f"     [{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}): {count:,} ({pct:.1f}%)")
            
            # Distance-based confidence breakdown
            print(f"\n   Confidence by minimum distance:")
            dist_bins = [0, 5, 10, 20, 50, 100, 200]
            for i in range(len(dist_bins) - 1):
                mask = (conf_df['min_distance'] >= dist_bins[i]) & (conf_df['min_distance'] < dist_bins[i+1])
                if mask.sum() > 0:
                    mean_conf = conf_df[mask]['confidence'].mean()
                    count = mask.sum()
                    print(f"     {dist_bins[i]}-{dist_bins[i+1]} km: {count:,} values, mean confidence: {mean_conf:.3f}")
    
    # ============================================================================
    # Step 10: Save curated dataset
    # ============================================================================
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.parent / f"merged_pm10_hourly_curated_{timestamp}.csv"
    
    print(f"\n10. Saving curated dataset to: {output_file}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Reorder columns for output (with interpolation columns)
    output_columns = ['datetime', 'station_code', 'station_name', 'pm10']
    if 'interpolation_method' in result_df.columns:
        output_columns.append('interpolation_method')
    if 'interpolation_confidence' in result_df.columns:
        output_columns.append('interpolation_confidence')
    
    result_df[output_columns].to_csv(output_file, index=False)
    
    print(f"   ✓ Saved {len(result_df):,} rows")
    
    # Save version without interpolation columns
    output_file_no_interp = output_file.parent / f"{output_file.stem}_no_interp_metadata.csv"
    print(f"\n10b. Saving curated dataset (without interpolation metadata) to: {output_file_no_interp}")
    
    # Columns without interpolation metadata
    output_columns_no_interp = ['datetime', 'station_code', 'station_name', 'pm10']
    result_df[output_columns_no_interp].to_csv(output_file_no_interp, index=False)
    
    print(f"   ✓ Saved {len(result_df):,} rows")
    
    # ============================================================================
    # Summary statistics
    # ============================================================================
    print(f"\n12. Summary:")
    print(f"   Original dataset: {len(pd.read_csv(input_file)):,} rows")
    print(f"   Curated dataset: {len(result_df):,} rows")
    print(f"   Stations in curated dataset: {len(result_df['station_code'].unique())}")
    print(f"   Date range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    print(f"   Gaps interpolated (< 4h first pass): {first_pass_interpolated:,} hours")
    print(f"   Gaps interpolated (< 2h final pass): {final_total_actual_interpolated:,} hours")
    print(f"   Gaps interpolated (distance-weighted): {distance_interpolated_count:,} hours")
    print(f"   Total gaps interpolated: {first_pass_interpolated + final_total_actual_interpolated + distance_interpolated_count:,} hours")
    
    # Per-station summary
    print(f"\n   Per-station row counts and completeness:")
    
    # Create a lookup dictionary for station confidence
    station_conf_lookup = {}
    if conf_df is not None and len(conf_df) > 0 and 'station_code' in conf_df.columns:
        station_conf_lookup = dict(zip(conf_df['station_code'], conf_df['confidence']))
    
    for station_code in sorted(result_df['station_code'].unique()):
        station_name = result_df[result_df['station_code'] == station_code]['station_name'].iloc[0]
        station_data = result_df[result_df['station_code'] == station_code]
        count = len(station_data)
        missing = station_data['pm10'].isna().sum()
        completeness = ((count - missing) / count * 100) if count > 0 else 0.0
        
        # Get confidence for this station
        station_conf = station_conf_lookup.get(station_code)
        conf_str = f", confidence: {station_conf:.3f}" if station_conf is not None else ""
        
        print(f"     {station_code} ({station_name[:45]}): {count:,} rows, {missing:,} missing ({completeness:.2f}% complete){conf_str}")
    
    print("\n" + "=" * 80)
    print("Dataset curation complete!")
    print("=" * 80)
    
    return result_df


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "arpal" / "PM10" / "merged_pm10_hourly.csv"
    stations_file = project_root / "data" / "arpal" / "PM10" / "arpal_pm10_stations.csv"
    output_file = project_root / "data" / "arpal" / "PM10" / "merged_pm10_hourly_curated.csv"
    
    # Run curation (starting from 2010)
    # confidence_threshold: stations with confidence below this will be dropped
    # Set to 0.0 to keep all stations, or higher (e.g., 0.3) to filter low-confidence stations
    create_curated_arpal_dataset(
        input_file, 
        stations_file,
        output_file=output_file, 
        max_gap_hours=4, 
        start_year=2012,
        confidence_threshold=0.25
    )
