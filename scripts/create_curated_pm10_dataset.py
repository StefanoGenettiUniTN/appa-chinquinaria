"""
Script to create a curated PM10 dataset from the raw merged CSV.

This script performs the following operations:
1. Drops station Rio Novo (502726) - too many missing years and far from other stations
2. Copies missing years of TV S Agnese (502612) from closest station TV Lancieri (502608)
3. Drops all data before 2012 (keeps 2012 onwards)
4. Interpolates 1 missing hour in VE Tagliamento (502720) in 2012
5. Interpolates all gaps shorter than 4 hours using linear interpolation
6. Final pass: interpolates remaining gaps shorter than 2 hours
7. Saves the curated dataset with an informative filename
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path to import from chinquinaria if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_contiguous_missing_periods(actual_times_set, expected_times):
    """
    Find all contiguous missing periods and return their start/end indices.
    
    Args:
        actual_times_set: Set of actual measurement timestamps
        expected_times: Series of expected hourly timestamps
    
    Returns:
        List of tuples: (start_idx, end_idx, length_in_hours)
    """
    missing_periods = []
    current_start_idx = None
    
    for i, t in enumerate(expected_times):
        if t not in actual_times_set:
            if current_start_idx is None:
                current_start_idx = i
        else:
            if current_start_idx is not None:
                # End of missing period
                length = i - current_start_idx
                missing_periods.append((current_start_idx, i - 1, length))
                current_start_idx = None
    
    # Handle case where series ends with missing values
    if current_start_idx is not None:
        length = len(expected_times) - current_start_idx
        missing_periods.append((current_start_idx, len(expected_times) - 1, length))
    
    return missing_periods


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
    # This ensures that if a station starts at 01:00:00, we still include 00:00:00 in expected_times
    year_start = pd.Timestamp(station_start.year, 1, 1, 0, 0, 0)
    year_end = pd.Timestamp(station_start.year, 12, 31, 23, 0, 0)
    
    # If station spans multiple years, use the full range
    if station_start.year != station_end.year:
        # Multi-year: use actual start/end but round to day boundaries
        range_start = pd.Timestamp(station_start.year, station_start.month, station_start.day, 0, 0, 0)
        range_end = pd.Timestamp(station_end.year, station_end.month, station_end.day, 23, 0, 0)
    else:
        # Single year: use full year range
        range_start = year_start
        range_end = year_end
    
    # Create complete hourly time series
    expected_times = pd.date_range(start=range_start, end=range_end, freq='h')
    
    # Create a DataFrame with complete time series
    complete_df = pd.DataFrame({
        'datetime': expected_times
    })
    
    # Merge with actual data
    station_df_sorted = station_df.sort_values('datetime').reset_index(drop=True)
    station_df_sorted['datetime'] = pd.to_datetime(station_df_sorted['datetime'])
    
    complete_df = complete_df.merge(
        station_df_sorted[['datetime', 'pm10', 'station_code', 'station_name']],
        on='datetime',
        how='left'
    )
    
    # Forward fill station info and ensure correct types
    complete_df['station_code'] = complete_df['station_code'].ffill().bfill().astype(int)
    complete_df['station_name'] = complete_df['station_name'].ffill().bfill().astype(str)
    
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
                # Check that we have valid values before and after
                before_val = complete_df.loc[start_idx - 1, 'pm10']
                after_val = complete_df.loc[end_idx + 1, 'pm10']
                if (before_val is not None and not pd.isna(before_val) and
                    after_val is not None and not pd.isna(after_val)):
                    short_gaps.append((start_idx, end_idx, length))
        else:
            # Mark long gap indices to preserve as NaN
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
        # Count how many values were actually filled in this gap
        gap_values = pm10_series.loc[start_idx:end_idx]
        actual_interpolated += gap_values.notna().sum()
    
    return complete_df, interpolated_count, actual_interpolated


def create_curated_dataset(input_file, output_file=None, max_gap_hours=4):
    """
    Main function to create curated PM10 dataset.
    
    Args:
        input_file: Path to merged_pm10_hourly.csv
        output_file: Optional output file path (default: auto-generated)
        max_gap_hours: Maximum gap length to interpolate (default: 4 hours)
    """
    print("=" * 80)
    print("PM10 Dataset Curation")
    print("=" * 80)
    
    # ============================================================================
    # Step 1: Load data
    # ============================================================================
    print(f"\n1. Loading data from: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime'])
    print(f"   Loaded {len(df):,} rows")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Get unique stations
    unique_stations = df['station_code'].unique()
    print(f"   Found {len(unique_stations)} unique stations: {sorted(unique_stations)}")
    
    # ============================================================================
    # Step 2: Drop station Rio Novo (502726)
    # ============================================================================
    print(f"\n2. Dropping station Rio Novo (502726)...")
    rows_before = len(df)
    df = df[df['station_code'] != 502726].copy()
    rows_after = len(df)
    print(f"   Dropped {rows_before - rows_after:,} rows")
    print(f"   Remaining stations: {sorted(df['station_code'].unique())}")
    
    # ============================================================================
    # Step 3: Copy missing years of TV S Agnese from TV Lancieri
    # ============================================================================
    print(f"\n3. Copying missing years of TV S Agnese (502612) from TV Lancieri (502608)...")
    
    # Identify missing years for TV S Agnese
    tv_s_agnese_df = df[df['station_code'] == 502612].copy()
    tv_lancieri_df = df[df['station_code'] == 502608].copy()
    
    tv_s_agnese_years = set(pd.to_datetime(tv_s_agnese_df['datetime']).dt.year.unique())
    tv_lancieri_years = set(pd.to_datetime(tv_lancieri_df['datetime']).dt.year.unique())
    missing_years = tv_lancieri_years - tv_s_agnese_years
    
    if missing_years:
        print(f"   Missing years for TV S Agnese: {sorted(missing_years)}")
        
        # Get data from TV Lancieri for missing years
        tv_lancieri_missing_years = tv_lancieri_df[
            pd.to_datetime(tv_lancieri_df['datetime']).dt.year.isin(missing_years)
        ].copy()
        
        # Copy and update station info
        copied_data = tv_lancieri_missing_years.copy()
        copied_data['station_code'] = 502612
        copied_data['station_name'] = 'TV S Agnese'
        
        # Append to dataset
        df = pd.concat([df, copied_data], ignore_index=True)
        print(f"   Copied {len(copied_data):,} rows from TV Lancieri to TV S Agnese")
    else:
        print(f"   No missing years found - TV S Agnese already has all years")
    
    # ============================================================================
    # Step 4: Drop data before 2012 (keep 2012 onwards)
    # ============================================================================
    print(f"\n4. Dropping data before 2012 (keeping 2012 onwards)...")
    rows_before = len(df)
    df['datetime'] = pd.to_datetime(df['datetime'])
    cutoff_date = pd.Timestamp("2012-01-01 00:00:00")
    df = df[df['datetime'] >= cutoff_date].copy()
    rows_after = len(df)
    print(f"   Dropped {rows_before - rows_after:,} rows")
    print(f"   New date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # ============================================================================
    # Step 5: Interpolate 1 missing hour in VE Tagliamento (502720) in 2012
    # ============================================================================
    print(f"\n5. Interpolating missing hour in VE Tagliamento (502720) in 2012...")
    
    ve_tagliamento_df = df[df['station_code'] == 502720].copy()
    ve_tagliamento_2012 = ve_tagliamento_df[
        pd.to_datetime(ve_tagliamento_df['datetime']).dt.year == 2012
    ].copy()
    
    # Create expected hourly series for 2012
    year_2012_start = pd.Timestamp("2012-01-01 00:00:00")
    year_2012_end = pd.Timestamp("2012-12-31 23:00:00")
    expected_2012_hours = pd.date_range(start=year_2012_start, end=year_2012_end, freq='h')
    
    # Find missing hour(s)
    actual_2012_times = set(pd.to_datetime(ve_tagliamento_2012['datetime']))
    missing_2012_hours = [t for t in expected_2012_hours if t not in actual_2012_times]
    
    if missing_2012_hours:
        print(f"   Found {len(missing_2012_hours)} missing hour(s) in 2012")
        
        # Create complete 2012 series for VE Tagliamento
        complete_2012_df = pd.DataFrame({'datetime': expected_2012_hours})
        complete_2012_df = complete_2012_df.merge(
            ve_tagliamento_2012[['datetime', 'pm10']],
            on='datetime',
            how='left'
        )
        complete_2012_df['station_code'] = 502720
        complete_2012_df['station_name'] = 'VE Tagliamento'
        
        # Interpolate the missing hour
        complete_2012_df['pm10'] = complete_2012_df['pm10'].interpolate(method='linear', limit_direction='both')
        
        # Remove old 2012 data and add interpolated version
        df = df[~((df['station_code'] == 502720) & 
                  (pd.to_datetime(df['datetime']).dt.year == 2012))].copy()
        df = pd.concat([df, complete_2012_df], ignore_index=True)
        print(f"   Interpolated {len(missing_2012_hours)} missing hour(s)")
    else:
        print(f"   No missing hours found in 2012 for VE Tagliamento")
    
    # ============================================================================
    # Step 6: Interpolate all gaps shorter than max_gap_hours
    # ============================================================================
    print(f"\n6. Interpolating gaps shorter than {max_gap_hours} hours...")
    
    unique_stations = df['station_code'].unique()
    interpolated_dfs = []
    total_interpolated = 0
    total_actual_interpolated = 0
    
    for station_code in unique_stations:
        station_df = df[df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        print(f"   Processing {station_code} ({station_name})...", end=" ")
        
        interpolated_df, gap_count, actual_count = interpolate_station_data(
            station_df, 
            max_gap_hours=max_gap_hours
        )
        
        interpolated_dfs.append(interpolated_df)
        total_interpolated += gap_count
        total_actual_interpolated += actual_count
        
        print(f"interpolated {actual_count} values")
    
    # Combine all stations
    print(f"\n7. Combining all stations...")
    result_df = pd.concat(interpolated_dfs, ignore_index=True)
    result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    
    # Reorder columns to match original format
    result_df = result_df[['datetime', 'station_code', 'station_name', 'pm10']]
    
    print(f"   Total rows in curated dataset: {len(result_df):,}")
    print(f"   Total gaps interpolated: {total_interpolated:,} hours")
    print(f"   Total values interpolated: {total_actual_interpolated:,}")
    
    # Track first pass totals before final pass
    first_pass_interpolated = total_actual_interpolated
    
    # ============================================================================
    # Step 8: Final pass - interpolate gaps shorter than 2 hours
    # ============================================================================
    print(f"\n8. Final pass: interpolating remaining gaps shorter than 2 hours...")
    
    final_interpolated_dfs = []
    final_total_interpolated = 0
    final_total_actual_interpolated = 0
    
    for station_code in result_df['station_code'].unique():
        station_df = result_df[result_df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        print(f"   Processing {station_code} ({station_name})...", end=" ")
        
        interpolated_df, gap_count, actual_count = interpolate_station_data(
            station_df, 
            max_gap_hours=2  # Final pass: only gaps < 2 hours
        )
        
        final_interpolated_dfs.append(interpolated_df)
        final_total_interpolated += gap_count
        final_total_actual_interpolated += actual_count
        
        print(f"interpolated {actual_count} values")
    
    # Combine all stations after final interpolation
    result_df = pd.concat(final_interpolated_dfs, ignore_index=True)
    result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    
    # Reorder columns to match original format
    result_df = result_df[['datetime', 'station_code', 'station_name', 'pm10']]
    
    print(f"   Final gaps interpolated in this pass: {final_total_interpolated:,} hours")
    print(f"   Final values interpolated in this pass: {final_total_actual_interpolated:,}")
    
    # Update total with final pass
    total_actual_interpolated = first_pass_interpolated + final_total_actual_interpolated
    
    # ============================================================================
    # Step 9: Save curated dataset
    # ============================================================================
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.parent / f"merged_pm10_hourly_curated_{timestamp}.csv"
    
    print(f"\n9. Saving curated dataset to: {output_file}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    
    print(f"   ✓ Saved {len(result_df):,} rows")
    
    # ============================================================================
    # Summary statistics
    # ============================================================================
    print(f"\n10. Summary:")
    print(f"   Original dataset: {len(pd.read_csv(input_file)):,} rows")
    print(f"   Curated dataset: {len(result_df):,} rows")
    print(f"   Stations in curated dataset: {sorted(result_df['station_code'].unique())}")
    print(f"   Date range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    print(f"   Gaps interpolated (< 4h first pass): {first_pass_interpolated:,} hours")
    print(f"   Gaps interpolated (< 2h final pass): {final_total_actual_interpolated:,} hours")
    print(f"   Total gaps interpolated: {total_actual_interpolated:,} hours")
    print(f"   Maximum gap length interpolated: {max_gap_hours - 1} hours (first pass), 1 hour (final pass)")
    
    # Per-station summary
    print(f"\n   Per-station row counts:")
    for station_code in sorted(result_df['station_code'].unique()):
        station_name = result_df[result_df['station_code'] == station_code]['station_name'].iloc[0]
        count = len(result_df[result_df['station_code'] == station_code])
        print(f"     {station_code} ({station_name}): {count:,} rows")
    
    print("\n" + "=" * 80)
    print("Dataset curation complete!")
    print("=" * 80)
    
    return result_df


if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "arpav" / "PM10" / "merged_pm10_hourly.csv"
    output_file = project_root / "data" / "arpav" / "PM10" / "merged_pm10_hourly_curated.csv"
    
    # Run curation
    create_curated_dataset(input_file, output_file=output_file, max_gap_hours=4)

